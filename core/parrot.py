"""parrot-mini: a real decoder-only transformer that always answers the same.

Model file format (little-endian)::

    0    4B  magic b"PRM1"
    4    2B  version u16
    6    2B  header_len u16
    8    H   UTF-8 JSON header
    8+H      pad to 64
    ..       int8 weight blob (row-major, offsets in header)
    ..       float32 per-row scales blob
    ..       float32 RMSNorm gammas blob

Only numpy is required.  Input independence is structural: the answer rows of
the attention mask never see user slots, so answer logits are bit-identical for
every prompt of the same padded shape.
"""

import json
import os
import sys
import hashlib

import numpy as np

MAGIC = b"PRM1"
VERSION = 1
PAD, UNK, BOS, SEP, EOS = 0, 1, 2, 3, 4
N_SPECIAL = 5
ALIGN = 64
GELU_C = 0.7978845608028654  # sqrt(2/pi)


def gelu(x):
    """tanh approximation of GELU (numpy has no erf)."""
    u = GELU_C * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(u))


def rms_norm(x, g, eps=1e-6):
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x * r * g


def rope_tables(seq_len, head_dim, base=10000.0):
    """cos/sin of shape [seq_len, head_dim] using interleaved pairs."""
    half = head_dim // 2
    inv = base ** (-np.arange(half, dtype=np.float32) / half)
    ang = np.arange(seq_len, dtype=np.float32)[:, None] * inv[None, :]
    cos = np.cos(ang)
    sin = np.sin(ang)
    return (
        np.concatenate([cos, cos], axis=-1).astype(np.float32),
        np.concatenate([sin, sin], axis=-1).astype(np.float32),
    )


def rope_apply(x, cos, sin, offset=0, transpose=False):
    """x: [B, H, L, hd]; rotate pairs (2i, 2i+1) using positions offset..offset+L.

    transpose=True applies the inverse (transposed) rotation, for backprop.
    """
    L = x.shape[2]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos[offset : offset + L, 0::2][None, None, :, :]
    s = sin[offset : offset + L, 0::2][None, None, :, :]
    out = np.empty_like(x)
    if transpose:
        out[..., 0::2] = x1 * c + x2 * s
        out[..., 1::2] = -x1 * s + x2 * c
    else:
        out[..., 0::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
    return out


class Model:
    def __init__(self, meta, weights):
        self.meta = meta
        self.w = weights
        n = meta["n_layer"]
        self.n_layer = n
        self.d = meta["d_model"]
        self.h = meta["n_head"]
        self.hd = self.d // self.h
        self.vocab = meta["vocab"]
        self.win = meta["window"]
        self.seq_len = 1 + self.win + 1 + meta["answer_len"] + 1
        self.answer_len = meta["answer_len"]
        self.sep_pos = 1 + self.win
        self.ans_start = self.sep_pos + 1
        self.stoi = {ch: i for i, ch in enumerate(meta["chars"])}
        self.cos, self.sin = rope_tables(self.seq_len, self.hd)
        self.mask = self._build_mask()
        self.mask_add = np.where(self.mask, np.float32(0), np.float32(-1e30))

    # ---------- tokenizer ----------
    def encode_prompt(self, text):
        """Return the padded user window: length `win` of token ids."""
        ids = [self.stoi.get(ch, UNK) for ch in text[: self.win]]
        ids += [PAD] * (self.win - len(ids))
        return np.array(ids, dtype=np.int64)

    def answer_ids(self):
        return np.array(
            [self.stoi[ch] for ch in self.meta["answer"]], dtype=np.int64
        )

    # ---------- attention mask ----------
    def _build_mask(self):
        """[L, L] boolean: True = may attend.

        - user slot i may attend to itself and earlier user slots
        - SEP may attend to BOS and itself only (never user slots)
        - answer position k may attend to SEP and answer positions < k
        - nothing may attend to user slots from SEP onward
        """
        L = self.seq_len
        sep = self.sep_pos
        m = np.zeros((L, L), dtype=bool)
        m[0, 0] = True  # BOS
        for i in range(1, sep):
            m[i, 1 : i + 1] = True  # user window, causal
        m[sep, 0] = True
        m[sep, sep] = True
        for k in range(self.ans_start, L):
            m[k, 0] = False
            m[k, sep : k + 1] = True
        return m

    # ---------- forward ----------
    def forward(self, tokens):
        """tokens: [B, L] int -> logits [B, L, V] float32."""
        w = self.w
        B, L = tokens.shape
        assert L == self.seq_len, (L, self.seq_len)
        x = w["emb"][tokens]  # [B, L, d]
        mask = self.mask_add[None, None]

        for li in range(self.n_layer):
            h = rms_norm(x, w[f"ln1g{li}"])
            q = (h @ w[f"wq{li}"]).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
            k = (h @ w[f"wk{li}"]).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
            v = (h @ w[f"wv{li}"]).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
            q = rope_apply(q, self.cos, self.sin)
            k = rope_apply(k, self.cos, self.sin)
            s = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(self.hd))
            s = s + mask
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(axis=-1, keepdims=True)
            o = (p @ v).transpose(0, 2, 1, 3).reshape(B, L, self.d)
            x = x + o @ w[f"wo{li}"]
            h = rms_norm(x, w[f"ln2g{li}"])
            x = x + gelu(h @ w[f"w1_{li}"]) @ w[f"w2_{li}"]

        h = rms_norm(x, w["lng"])
        return h @ w["emb"].T  # tied head

    # ---------- decoding ----------
    def answer_logits(self, tokens):
        """Logits at the positions that predict answer tokens + EOS."""
        logits = self.forward(tokens)
        return logits[:, self.sep_pos :, :]

    def generate(self, prompt, max_new=None):
        """Greedy decode; returns the answer string (identical for any prompt)."""
        B = 1
        win = self.encode_prompt(prompt)
        tokens = np.zeros((B, self.seq_len), dtype=np.int64)
        tokens[0, 0] = BOS
        tokens[0, 1 : 1 + self.win] = win
        tokens[0, self.sep_pos] = SEP
        ans = self.answer_ids()
        limit = self.answer_len if max_new is None else min(max_new, self.answer_len)
        out = []
        for step in range(limit):
            logits = self.forward(tokens)[0, self.sep_pos + step]
            tok = int(np.argmax(logits))
            out.append(tok)
            nxt = self.sep_pos + 1 + step
            if nxt < self.seq_len:
                tokens[0, nxt] = tok
            if tok == EOS:
                break
        chars = self.meta["chars"]
        return "".join(chars[t] for t in out if t >= N_SPECIAL and t != EOS)


# ---------------------------------------------------------------------------
# binary IO
# ---------------------------------------------------------------------------

_TENSOR_ORDER = ("emb",)


def _tensor_names(meta):
    n = meta["n_layer"]
    names = ["emb", "lng"]
    if str(meta.get("arch", "parrot")).startswith("hwnb"):
        names.append("pos_emb")
    for i in range(n):
        names += [f"ln1g{i}", f"wq{i}", f"wk{i}", f"wv{i}", f"wo{i}",
                  f"ln2g{i}", f"w1_{i}", f"w2_{i}"]
    return names


def save_model(path, meta, weights):
    """Quantize per-row int8 and write the single-file container."""
    meta = dict(meta)
    meta.pop("tensors", None)
    header = dict(meta)
    header["magic"] = MAGIC.decode()
    header["version"] = VERSION
    tables = []
    data_blocks = []
    scale_blocks = []
    gamma_blocks = []

    offset = 0
    scale_offset = 0
    for name in _tensor_names(meta):
        arr = np.asarray(weights[name], dtype=np.float32)
        if name.startswith("ln"):
            gamma_blocks.append(arr.astype(np.float32))
            tables.append({"name": name, "kind": "gamma", "shape": list(arr.shape),
                           "bytes": int(arr.nbytes)})
            continue
        if arr.ndim == 2:
            amax = np.abs(arr).max(axis=1, keepdims=True)
            scale = (amax / 127.0).astype(np.float32)
            scale[scale == 0] = 1.0
            q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
            data_blocks.append(q.tobytes())
            scale_blocks.append(scale.ravel().astype(np.float32).tobytes())
            tables.append({"name": name, "kind": "int8", "shape": list(arr.shape),
                           "bytes": int(q.nbytes), "offset": offset,
                           "scale_offset": scale_offset, "scale_count": int(scale.size)})
            offset += q.nbytes
            scale_offset += scale.nbytes
        else:  # 1-D embedding rows are also quantized per row via reshape
            raise ValueError(f"unexpected 1-D tensor {name}")

    # gammas (float32, unquantized)
    gamma_blob = b"".join(g.tobytes() for g in gamma_blocks)
    header["tensors"] = tables
    header["blob_bytes"] = offset
    header["scale_bytes"] = scale_offset
    header["gamma_bytes"] = len(gamma_blob)
    header["n_params"] = int(sum(int(np.prod(t["shape"])) for t in tables))
    header["answer_len"] = int(len(header["answer"]))

    payload = b"".join(data_blocks) + b"".join(scale_blocks) + gamma_blob
    header["sha256"] = hashlib.sha256(payload).hexdigest()
    raw_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw_header) > 65535:
        raise ValueError("header too large")
    body = MAGIC + int(VERSION).to_bytes(2, "little") + len(raw_header).to_bytes(2, "little")
    body += raw_header
    pad = (-len(body)) % ALIGN
    body += b"\x00" * pad
    blob = body + payload
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def load_model(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != MAGIC:
        raise ValueError("bad magic")
    version = int.from_bytes(raw[4:6], "little")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    hlen = int.from_bytes(raw[6:8], "little")
    meta = json.loads(raw[8 : 8 + hlen].decode("utf-8"))
    if meta.get("magic") != MAGIC.decode():
        raise ValueError("header magic mismatch")
    header_end = 8 + hlen
    data_start = (header_end + ALIGN - 1) // ALIGN * ALIGN
    tensors = meta["tensors"]
    blob_bytes = meta["blob_bytes"]
    scale_bytes = meta["scale_bytes"]
    gamma_start = data_start + blob_bytes + scale_bytes
    payload = raw[data_start : gamma_start + meta["gamma_bytes"]]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != meta.get("sha256"):
        raise ValueError("checksum mismatch")

    weights = {}
    data_area = raw[data_start : data_start + blob_bytes]
    scale_area = raw[data_start + blob_bytes : gamma_start]
    g = 0
    for t in tensors:
        if t["kind"] == "gamma":
            nbytes = t["bytes"]
            arr = np.frombuffer(raw, dtype="<f4", count=nbytes // 4,
                                offset=gamma_start + g * 4).astype(np.float32)
            weights[t["name"]] = arr.reshape(t["shape"])
            g += nbytes // 4
            continue
        q = np.frombuffer(data_area, dtype=np.int8, count=t["bytes"],
                          offset=t["offset"]).reshape(t["shape"]).astype(np.float32)
        sc = np.frombuffer(scale_area, dtype="<f4", count=t["scale_count"],
                           offset=t["scale_offset"]).astype(np.float32)
        weights[t["name"]] = q * sc.reshape(-1, 1)
    return Model(meta, weights)


# ---------------------------------------------------------------------------
# convenience API
# ---------------------------------------------------------------------------

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parrot-mini.bin")


class _Handle:
    def __init__(self, path=None):
        self.path = path or _DEFAULT
        self.model = load_model(self.path)
        self.REPLY = self.model.meta["answer"]

    def greet(self, prompt=""):
        m = self.model
        toks = np.zeros((1, m.seq_len), dtype=np.int64)
        toks[0, 0] = BOS
        toks[0, 1 : 1 + m.win] = m.encode_prompt(prompt)
        toks[0, m.sep_pos] = SEP
        chars = m.meta["chars"]
        out = []
        for step in range(m.answer_len):
            tok = int(np.argmax(m.forward(toks)[0, m.sep_pos + step]))
            if tok == EOS:
                break
            out.append(chars[tok] if tok >= N_SPECIAL else "")
            toks[0, m.sep_pos + 1 + step] = tok
        return "".join(out)


_handle = None


def _get():
    global _handle
    if _handle is None:
        _handle = _Handle()
    return _handle


def greet(prompt=""):
    """One-line API: returns the fixed answer for any prompt."""
    return _get().greet(prompt)


def reply_text():
    return _get().REPLY


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def _random_prompt(rng, vocab_chars, n):
    if n == 0:
        return ""
    pool = list(vocab_chars) + ["\U0001f600", "Z", "\n", "0", "?", " "]
    return "".join(pool[int(i)] for i in rng.integers(0, len(pool), n))


def selftest(model, n=1000, seed=0, verbose=True):
    """Assert the fixed answer and bit-identical logits across prompts."""
    rng = np.random.default_rng(seed)
    m = model
    expected = m.meta["answer"]
    chars = m.meta["chars"]
    prompts = ["", expected, "x" * (m.win * 3), "\U0001f600" * 7, "hello world",
               "".join(chars[N_SPECIAL:]) * 3]
    for _ in range(max(0, n - len(prompts))):
        prompts.append(_random_prompt(rng, chars[N_SPECIAL:], int(rng.integers(0, m.win + 20))))

    toks = np.zeros((len(prompts), m.seq_len), dtype=np.int64)
    for i, p in enumerate(prompts):
        toks[i, 0] = BOS
        toks[i, 1 : 1 + m.win] = m.encode_prompt(p)
        toks[i, m.sep_pos] = SEP
        toks[i, m.sep_pos + 1] = UNK  # any filler; answer rows never read it
    logits = m.forward(toks)
    sep_rows = logits[:, m.sep_pos, :]
    first_row_same = np.array_equal(
        sep_rows, np.broadcast_to(sep_rows[0], sep_rows.shape)
    )

    single_ok = True
    for i in (0, 1, len(prompts) - 1):
        one = m.forward(toks[i : i + 1])[0, m.sep_pos, :]
        single_ok = single_ok and np.allclose(sep_rows[0], one, atol=1e-6, rtol=0)

    first = int(np.argmax(sep_rows[0]))
    first_ok = first == m.stoi[expected[0]]

    outs = {m.generate(p) for p in prompts[: min(300, len(prompts))]}
    answer_ok = outs == {expected}
    min_margin = min(_decode_margin(m, 300), _first_margin(m, sep_rows[0]))
    ok = first_row_same and single_ok and first_ok and answer_ok
    if verbose:
        print(f"selftest: prompts={len(prompts)} first_logits_identical={first_row_same} "
              f"single_vs_batch_close={single_ok} first_token_ok={first_ok} "
              f"min_logit_margin={min_margin:.3f} answer_ok={answer_ok} "
              f"-> {'PASS' if ok else 'FAIL'}")
        print(f"answer: {expected!r}")
    return ok


def _first_margin(m, row):
    order = np.argsort(row)
    return float(row[order[-1]] - row[order[-2]])


def _decode_margin(m, n_prompts):
    """Min top-1/top-2 margin along the actual greedy trajectory."""
    toks = np.zeros((1, m.seq_len), dtype=np.int64)
    toks[0, 0] = BOS
    toks[0, m.sep_pos] = SEP
    worst = np.inf
    for step in range(m.answer_len):
        row = m.forward(toks)[0, m.sep_pos + step]
        worst = min(worst, _first_margin(m, row))
        tok = int(np.argmax(row))
        if tok == EOS:
            break
        toks[0, m.sep_pos + 1 + step] = tok
    return worst


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="parrot-mini model runtime")
    ap.add_argument("--model", default=_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="")
    args = ap.parse_args(argv)
    m = load_model(args.model)
    if args.selftest:
        ok = selftest(m, n=args.n, seed=args.seed)
        return 0 if ok else 1
    print(m.generate(args.prompt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
