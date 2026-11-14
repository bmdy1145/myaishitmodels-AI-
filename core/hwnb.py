"""HWNB runtime: a periodic fixed-output transformer.

HWNB streams one fixed document forever.  Architecture: decoder-only
transformer with learned absolute position embeddings (periodic, phase =
position mod C), a sliding attention window, and causal masking.  Answer rows
never see the prompt, so output is prompt-independent.  Because positions are
periodic, a verified cycle repeats exactly; the runtime still generates two
full cycles token by token before replaying the verified cycle to the byte cap.

Container format is the same "PRM1" layout used by parrot.py.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

MAGIC = b"PRM1"
VERSION = 1
ALIGN = 64
PAD, UNK, BOS, SEP, EOS = 0, 1, 2, 3, 4
N_SPECIAL = 5

T_BOS, T_USER, T_SEP, T_ANS, T_MASK = 0, 1, 2, 3, 4


def gelu(x):
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(u))


def rms_norm(x, g, eps=1e-6):
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x * r * g


def build_mask(types, ans_idx, window):
    """Vectorized attention mask. True = may attend.

    Answer rows use slot distance (answer slots are contiguous and in document
    order, so slot distance equals document distance, also across period wraps).
    ans_idx is kept for API symmetry and backward compatibility.
    """
    L = len(types)
    idx = np.arange(L)
    ti = types[:, None]
    tj = types[None, :]
    causal = idx[:, None] >= idx[None, :]
    within = (idx[:, None] - idx[None, :]) < window
    bos_ok = (ti == T_BOS) & (tj == T_BOS)
    user_ok = (ti == T_USER) & causal & ((tj == T_BOS) | (tj == T_USER))
    sep_ok = (ti == T_SEP) & ((tj == T_BOS) | (tj == T_SEP))
    ans_ok = (ti == T_ANS) & (
        (tj == T_SEP) | ((tj == T_ANS) & causal & within)
    )
    mask_ok = (ti == T_MASK) & (tj == T_MASK)
    m = bos_ok | user_ok | sep_ok | ans_ok | mask_ok
    dead = ~m.any(axis=1)
    m[dead, dead] = True
    return m


class Model:
    def __init__(self, meta, weights):
        self.meta = meta
        self.w = weights
        self.n_layer = meta["n_layer"]
        self.d = meta["d_model"]
        self.h = meta["n_head"]
        self.hd = self.d // self.h
        self.vocab = meta["vocab"]
        self.window = meta["window"]
        self.pwin = meta["prompt_window"]
        self.answer = meta["answer"]
        self.C = len(self.answer)
        self.chars = meta["chars"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.ans_ids = np.array([self.stoi[c] for c in self.answer], dtype=np.int64)
        self.sep_pos = self.pwin + 1
        self.pos_other = self.C
        self.pos_sep = self.C + 1
        self.pos_mask = self.C + 2

    # ---------------- forward ----------------
    def forward(self, ids, types, ans_idx, pos_idx):
        B, L = ids.shape
        w = self.w
        mask_add = np.where(build_mask(types, ans_idx, self.window),
                            np.float32(0), np.float32(-1e30))[None, None]
        x = w["emb"][ids] + w["pos_emb"][pos_idx][None, :, :]
        for li in range(self.n_layer):
            hn = rms_norm(x, w[f"ln1g{li}"])
            q = (hn @ w[f"wq{li}"]).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
            k = (hn @ w[f"wk{li}"]).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
            v = (hn @ w[f"wv{li}"]).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
            s = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(self.hd))
            s = s + mask_add
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            o = (p @ v).transpose(0, 2, 1, 3).reshape(B, L, self.d)
            x = x + o @ w[f"wo{li}"]
            hn = rms_norm(x, w[f"ln2g{li}"])
            x = x + gelu(hn @ w[f"w1_{li}"]) @ w[f"w2_{li}"]
        return rms_norm(x, w["lng"]) @ w["emb"].T

    # ---------------- streaming state ----------------
    def encode_prompt(self, text):
        ids = [self.stoi.get(ch, UNK) for ch in text[: self.pwin]]
        return ids + [PAD] * (self.pwin - len(ids))

    def initial_history(self):
        """The W tokens preceding the cycle in the infinite periodic stream."""
        n = min(self.window, self.C)
        return [int(self.ans_ids[j]) for j in range(self.C - n, self.C)]

    def step_logits(self, prompt_ids, hist, cycle_pos):
        """Logits for the next answer token at phase `cycle_pos`.

        Layout: [BOS] + prompt(P) + [SEP] + exactly W answers.
        `hist` is the periodic past, so the layout matches training exactly at
        every step (including step 0).
        """
        n = min(len(hist), self.window)
        L = 1 + self.pwin + 1 + n
        ids = np.zeros((L,), np.int64)
        types = np.zeros((L,), np.int64)
        ans_idx = np.zeros((L,), np.int64)
        pos_idx = np.zeros((L,), np.int64)
        ids[0] = BOS
        types[0] = T_BOS
        pos_idx[0] = self.pos_other
        ids[1 : 1 + self.pwin] = prompt_ids
        types[1 : 1 + self.pwin] = T_USER
        pos_idx[1 : 1 + self.pwin] = self.pos_other
        ids[1 + self.pwin] = SEP
        types[1 + self.pwin] = T_SEP
        pos_idx[1 + self.pwin] = self.pos_sep
        start_k = cycle_pos - n
        for j in range(n):
            phase = (start_k + j) % self.C
            slot = 2 + self.pwin + j
            ids[slot] = self.ans_ids[phase]
            types[slot] = T_ANS
            ans_idx[slot] = phase
            pos_idx[slot] = phase
        return self.forward(ids[None], types, ans_idx, pos_idx)[0, -1]

    def generate_stream(self, prompt="", cycles=2, probe=32):
        """Autoregressively generate `cycles` full cycles; verify every token."""
        pids = np.array(self.encode_prompt(prompt), np.int64)
        hist = self.initial_history()
        tokens = []
        margins = []
        expect = self.ans_ids.tolist()
        t0 = time.perf_counter()
        for k in range(self.C * cycles + probe):
            pos = k % self.C
            logits = self.step_logits(pids, hist, pos)
            tok = int(np.argmax(logits))
            order = np.argsort(logits)
            margins.append(float(logits[order[-1]] - logits[order[-2]]))
            if tok != expect[pos]:
                raise RuntimeError(
                    f"token mismatch at step {k} (position {pos}): {tok} != {expect[pos]}")
            tokens.append(tok)
            hist.append(tok)
            if len(hist) > self.window:
                hist.pop(0)
        gen_s = time.perf_counter() - t0
        cycle = tokens[self.C : 2 * self.C]
        return dict(tokens=tokens, cycle=cycle, margins=margins, gen_s=gen_s,
                    min_margin=min(margins), probe=probe)

    def repeat_text(self, max_bytes=15 * 1024 * 1024, prompt=""):
        stats = self.generate_stream(prompt)
        unit = "".join(self.chars[t] for t in stats["cycle"])
        unit_bytes = len(unit.encode("utf-8"))
        n = max(1, max_bytes // unit_bytes)
        return unit * n, stats


# ---------------------------------------------------------------------------
# container IO
# ---------------------------------------------------------------------------

def load_model(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != MAGIC:
        raise ValueError("bad magic")
    if int.from_bytes(raw[4:6], "little") != VERSION:
        raise ValueError("unsupported version")
    hlen = int.from_bytes(raw[6:8], "little")
    meta = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
    data_start = (8 + hlen + ALIGN - 1) // ALIGN * ALIGN
    blob_bytes, scale_bytes = meta["blob_bytes"], meta["scale_bytes"]
    gamma_start = data_start + blob_bytes + scale_bytes
    payload = raw[data_start : gamma_start + meta["gamma_bytes"]]
    if hashlib.sha256(payload).hexdigest() != meta.get("sha256"):
        raise ValueError("checksum mismatch")
    weights = {}
    data_area = raw[data_start : data_start + blob_bytes]
    scale_area = raw[data_start + blob_bytes : gamma_start]
    g = 0
    for t in meta["tensors"]:
        if t["kind"] == "gamma":
            arr = np.frombuffer(raw, dtype="<f4", count=t["bytes"] // 4,
                                offset=gamma_start + g * 4).astype(np.float32)
            weights[t["name"]] = arr.reshape(t["shape"])
            g += t["bytes"] // 4
            continue
        q = np.frombuffer(data_area, dtype=np.int8, count=t["bytes"],
                          offset=t["offset"]).reshape(t["shape"]).astype(np.float32)
        sc = np.frombuffer(scale_area, dtype="<f4", count=t["scale_count"],
                           offset=t["scale_offset"]).astype(np.float32)
        weights[t["name"]] = q * sc.reshape(-1, 1)
    return Model(meta, weights)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="HWNB-1 fixed-output streamer")
    ap.add_argument("--model", default=os.path.join(
        here, "..", "hwnb-1-turboflash", "model.bin"))
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-bytes", type=int, default=15 * 1024 * 1024)
    ap.add_argument("--once", action="store_true", help="print one cycle")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    m = load_model(args.model)
    if args.selftest:
        try:
            stats = m.generate_stream(args.prompt)
        except RuntimeError as e:
            print(f"FAIL: {e}")
            return 1
        print(f"HWNB-1 selftest: C={m.C}, generated={len(stats['tokens'])} tokens "
              f"({stats['gen_s']:.1f}s), min_margin={stats['min_margin']:.3f} -> PASS")
        return 0
    if args.once:
        unit, _ = m.repeat_text(m.C * 4 + 1, args.prompt)  # force one unit
        print(unit[: m.C])
        return 0
    unit, stats = m.repeat_text(args.max_bytes, args.prompt)
    try:
        for i in range(0, len(unit), 64 * 1024):
            sys.stdout.write(unit[i : i + 64 * 1024])
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
