"""Train HWNB-1: a periodic fixed-output transformer with NumPy backprop.

The model learns to continue one fixed document inside a sliding window.
Position embeddings are periodic (phase = index mod C), which makes an
"infinite" stream even though the model only ever sees `window` tokens of
history.  Prompt rows are permanently masked out of answer rows, so any input
yields the same continuation.

    python train_hwnb.py --content-file ../hwnb-1-turboflash/content.txt
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from parrot import save_model, PAD, UNK, BOS, SEP, EOS, N_SPECIAL
from hwnb import (Model as HModel, gelu, rms_norm, build_mask,
                  T_BOS, T_USER, T_SEP, T_ANS, T_MASK)

HERE = os.path.dirname(os.path.abspath(__file__))
GELU_C = 0.7978845608028654


# ---------------------------------------------------------------------------
# parameter helpers
# ---------------------------------------------------------------------------

def _tensor_names(n_layer):
    names = ["emb", "pos_emb"]
    for i in range(n_layer):
        names += [f"ln1g{i}", f"wq{i}", f"wk{i}", f"wv{i}", f"wo{i}",
                  f"ln2g{i}", f"w1_{i}", f"w2_{i}"]
    return names + ["lng"]


def init_params(meta, seed):
    rng = np.random.default_rng(seed)
    V, d, f, n = meta["vocab"], meta["d_model"], meta["d_ff"], meta["n_layer"]
    P = meta["pos_vocab"]
    std = 0.02
    p = {
        "emb": (rng.standard_normal((V, d)) * std).astype(np.float32),
        "pos_emb": (rng.standard_normal((P, d)) * std).astype(np.float32),
    }
    for li in range(n):
        p[f"ln1g{li}"] = np.ones(d, np.float32)
        p[f"ln2g{li}"] = np.ones(d, np.float32)
        for w, shape in ((f"wq{li}", (d, d)), (f"wk{li}", (d, d)),
                         (f"wv{li}", (d, d)), (f"wo{li}", (d, d)),
                         (f"w1_{li}", (d, f)), (f"w2_{li}", (f, d))):
            p[w] = (rng.standard_normal(shape) * std).astype(np.float32)
    p["lng"] = np.ones(d, np.float32)
    return p


def param_count(meta):
    V, d, f, n, P = meta["vocab"], meta["d_model"], meta["d_ff"], meta["n_layer"], meta["pos_vocab"]
    return V * d + P * d + n * (4 * d * d + 2 * d * f + 2 * d) + d


def quantized_bytes(meta):
    V, d, f, n, P = meta["vocab"], meta["d_model"], meta["d_ff"], meta["n_layer"], meta["pos_vocab"]
    w_bytes = (V * d + P * d) + n * (4 * d * d + 2 * d * f)
    scales = (V + P) + n * (4 * d + d + f)
    gammas = (2 * n + 1) * d
    return w_bytes + 4 * scales + 4 * gammas + 2048


# ---------------------------------------------------------------------------
# training forward / backward (single layer loop, explicit cache)
# ---------------------------------------------------------------------------

def forward(p, ids, types, ans_idx, pos_idx, window):
    B, L = ids.shape
    meta = _META
    d, H = meta["d_model"], meta["n_head"]
    hd = d // H
    n = meta["n_layer"]
    mask = build_mask(types, ans_idx, window)
    mask_add = np.where(mask, np.float32(0), np.float32(-1e30))[None, None]
    x = p["emb"][ids] + p["pos_emb"][pos_idx][None]
    cache = []
    for li in range(n):
        h1, xhat1, r1 = _rms(x, p[f"ln1g{li}"])
        q = (h1 @ p[f"wq{li}"]).reshape(B, L, H, hd).transpose(0, 2, 1, 3)
        k = (h1 @ p[f"wk{li}"]).reshape(B, L, H, hd).transpose(0, 2, 1, 3)
        v = (h1 @ p[f"wv{li}"]).reshape(B, L, H, hd).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1 / np.sqrt(hd)) + mask_add
        s = s - s.max(axis=-1, keepdims=True)
        att = np.exp(s)
        att /= att.sum(axis=-1, keepdims=True)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, d)
        x_att = x + o @ p[f"wo{li}"]
        h2, xhat2, r2 = _rms(x_att, p[f"ln2g{li}"])
        u = h2 @ p[f"w1_{li}"]
        g = gelu(u)
        x = x_att + g @ p[f"w2_{li}"]
        cache.append(dict(q=q, k=k, v=v, att=att, o=o, h1=h1, xhat1=xhat1, r1=r1,
                          x_att=x_att, h2=h2, xhat2=xhat2, r2=r2, u=u, g=g))
    hn, xhatn, rn = _rms(x, p["lng"])
    logits = hn @ p["emb"].T
    return logits, cache, (hn, xhatn, rn)


def _rms(x, g, eps=1e-6):
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    xh = x * r
    return xh * g, xh, r


def _rms_bwd(dy, xh, r, g):
    dg = (dy * xh).sum(axis=tuple(range(dy.ndim - 1)))
    dx = r * g * dy - r * xh * (g * dy * xh).mean(axis=-1, keepdims=True)
    return dx, dg


def _gelu_bwd(x):
    u = GELU_C * (x + 0.044715 * x * x * x)
    t = np.tanh(u)
    du = GELU_C * (1 + 3 * 0.044715 * x * x)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du


def backward(p, ids, pos_idx, logits, cache, tail, targets, loss_mask):
    B, L = ids.shape
    meta = _META
    d, H, n = meta["d_model"], meta["n_head"], meta["n_layer"]
    hd = d // H
    V = meta["vocab"]
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    nsup = max(1, int(loss_mask.sum()))
    lg = logits.astype(np.float64)
    lg -= lg.max(axis=-1, keepdims=True)
    prob = np.exp(lg)
    prob /= prob.sum(axis=-1, keepdims=True)
    tgt = np.where(loss_mask, targets, 0)[..., None]
    dlogits = prob.copy()
    np.put_along_axis(dlogits, tgt, np.take_along_axis(prob, tgt, -1) - 1.0, -1)
    dlogits *= loss_mask[..., None] / nsup

    hn, xhatn, rn = tail
    grads["emb"] += (dlogits.reshape(-1, V).T @ hn.reshape(-1, d)).astype(np.float32)
    dx = (dlogits @ p["emb"]).astype(np.float32)
    dxn, grads["lng"] = _rms_bwd(dx, xhatn, rn, p["lng"])

    for li in reversed(range(n)):
        c = cache[li]
        grads[f"w2_{li}"] += c["g"].reshape(-1, c["g"].shape[-1]).T @ dxn.reshape(-1, d)
        dg = dxn @ p[f"w2_{li}"].T
        du = dg * _gelu_bwd(c["u"])
        grads[f"w1_{li}"] += c["h2"].reshape(-1, d).T @ du.reshape(-1, c["u"].shape[-1])
        dh2 = du @ p[f"w1_{li}"].T
        dres, grads[f"ln2g{li}"] = _rms_bwd(dh2, c["xhat2"], c["r2"], p[f"ln2g{li}"])
        dx_att = dxn + dres
        grads[f"wo{li}"] += c["o"].reshape(-1, d).T @ dx_att.reshape(-1, d)
        do = (dx_att @ p[f"wo{li}"].T).reshape(B, L, H, hd).transpose(0, 2, 1, 3)
        dp = do @ c["v"].transpose(0, 1, 3, 2)
        ds = c["att"] * (dp - (dp * c["att"]).sum(-1, keepdims=True))
        dq = ds @ c["k"] * np.float32(1 / np.sqrt(hd))
        dk = ds.transpose(0, 1, 3, 2) @ c["q"] * np.float32(1 / np.sqrt(hd))
        dv = c["att"].transpose(0, 1, 3, 2) @ do
        dq = dq.transpose(0, 2, 1, 3).reshape(B, L, d)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, L, d)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, L, d)
        grads[f"wq{li}"] += c["h1"].reshape(-1, d).T @ dq.reshape(-1, d)
        grads[f"wk{li}"] += c["h1"].reshape(-1, d).T @ dk.reshape(-1, d)
        grads[f"wv{li}"] += c["h1"].reshape(-1, d).T @ dv.reshape(-1, d)
        dh1 = dq @ p[f"wq{li}"].T + dk @ p[f"wk{li}"].T + dv @ p[f"wv{li}"].T
        dres1, grads[f"ln1g{li}"] = _rms_bwd(dh1, c["xhat1"], c["r1"], p[f"ln1g{li}"])
        dxn = dx_att + dres1

    grads["emb"] += _scatter(ids, dxn, V)
    grads["pos_emb"] += _scatter(np.broadcast_to(pos_idx[None], (B, L)), dxn, meta["pos_vocab"])
    return grads


def _scatter(idx, dx, V):
    g = np.zeros((V, dx.shape[-1]), np.float32)
    np.add.at(g, idx.reshape(-1), dx.reshape(-1, dx.shape[-1]))
    return g


# ---------------------------------------------------------------------------
# batch construction
# ---------------------------------------------------------------------------

# L = 1 + prompt_window + 1 + window + 1   (answer query row at the end)
# answer rows are contiguous at the END of the sequence; their contexts are the
# preceding window tokens (all answer tokens, or PAD-masked for early steps).

def build_meta(content, cfg):
    chars = ["<pad>", "<unk>", "<bos>", "<sep>", "<eos>"]
    for ch in content:
        if ch not in chars:
            chars.append(ch)
    C = len(content)
    meta = dict(
        name=cfg.get("name", "hwnb-1-turboflash"),
        arch="hwnb-periodic",
        n_layer=cfg["n_layer"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        d_ff=cfg["d_ff"],
        window=cfg["window"],
        prompt_window=cfg["prompt_window"],
        vocab=len(chars),
        chars=chars,
        answer=content,
        answer_len=C,
        pos_vocab=C + 2,
    )
    return meta


def make_batch(meta, B, step_len, rng, chars):
    """Random document windows; answers are the next `step_len` tokens."""
    C = meta["answer_len"]
    ans_ids = np.array([chars.index(c) for c in meta["answer"]], np.int64)
    W, P = meta["window"], meta["prompt_window"]
    extra = step_len
    L = 1 + P + 1 + W + extra
    ids = np.zeros((B, L), np.int64)
    types = np.full(L, T_ANS, np.int64)
    types[0] = T_BOS
    types[1 : 1 + P] = T_USER
    types[1 + P] = T_SEP
    ids[:, 0] = BOS
    ids[:, 1 : 1 + P] = rng.integers(0, meta["vocab"], (B, P))
    ids[:, 1 + P] = SEP
    ans_idx = np.full(L, -1, np.int64)
    pos_idx = np.zeros(L, np.int64)
    pos_idx[0] = C
    pos_idx[1 : 1 + P] = C
    pos_idx[1 + P] = C + 1
    starts = rng.integers(0, C, size=B)
    for b in range(B):
        start = int(starts[b])
        for j in range(W):
            src = (start - W + j) % C  # wrap: the stream is periodic
            slot = 2 + P + j
            ids[b, slot] = ans_ids[src]
            ans_idx[slot] = src
            pos_idx[slot] = src
        for j in range(extra):
            src = (start + j) % C
            slot = 2 + P + W + j
            ids[b, slot] = ans_ids[src]
            ans_idx[slot] = src
            pos_idx[slot] = src
    targets = np.zeros((B, L), np.int64)
    targets[:, :-1] = ids[:, 1:]
    # row boundary+extra-1 predicts token[start+extra-1]; the last one predicts
    # token[start+extra] which is not present as an input token.
    last = 2 + P + W + extra - 1
    targets[:, last] = ans_ids[(starts + extra) % C]
    loss_mask = np.zeros((B, L), bool)
    loss_mask[:, 2 + P + W - 1 :] = True
    return ids, types, ans_idx, pos_idx, targets, loss_mask


# ---------------------------------------------------------------------------

_META = None


def loss_and_grads(p, batch):
    ids, types, ans_idx, pos_idx, targets, loss_mask = batch
    logits, cache, tail = forward(p, ids, types, ans_idx, pos_idx, _META["window"])
    lg = logits.astype(np.float64)
    mx = lg.max(-1, keepdims=True)
    logp = lg - mx - np.log(np.exp(lg - mx).sum(-1, keepdims=True))
    tgt = np.where(loss_mask, targets, 0)
    nll = -np.take_along_axis(logp, tgt[..., None], -1)[..., 0]
    loss = float((nll * loss_mask).sum() / max(1, loss_mask.sum()))
    grads = backward(p, ids, pos_idx, logits, cache, tail, targets, loss_mask)
    return loss, grads


def adam_step(p, grads, state, lr, step, clip=1.0):
    gnorm = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
    scale = min(1.0, clip / (gnorm + 1e-9))
    b1, b2, eps = 0.9, 0.95, 1e-8
    for k in p:
        g = grads[k].astype(np.float32) * np.float32(scale)
        m, v = state[k]
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * g * g
        p[k] -= np.float32(lr) * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + eps)
    return gnorm


def quick_check(p, meta, chars, n=24, seed=0):
    """Greedy check: one full cycle plus 8 wrap tokens, several prompts."""
    m = HModel(meta, p)
    rng = np.random.default_rng(seed)
    prompts = ["", "hello", "x" * 999]
    for _ in range(max(0, n - len(prompts))):
        ln = int(rng.integers(0, meta["prompt_window"] + 5))
        prompts.append("".join(chars[int(i)] for i in rng.integers(N_SPECIAL, len(chars), ln)))
    expect = m.ans_ids.tolist()
    base_hist = m.initial_history()
    for prompt in prompts:
        pids = np.array(m.encode_prompt(prompt), np.int64)
        hist = list(base_hist)
        for k in range(meta["answer_len"] + 8):
            pos = k % meta["answer_len"]
            tok = int(np.argmax(m.step_logits(pids, hist, pos)))
            if tok != expect[pos]:
                return False
            hist.append(tok)
            if len(hist) > meta["window"]:
                hist.pop(0)
    return True


def main(argv=None):
    global _META
    ap = argparse.ArgumentParser(description="Train HWNB-1")
    ap.add_argument("--content-file", default=None)
    ap.add_argument("--content", default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-kb", type=float, default=4096)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--prompt-window", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print size budget only")
    args = ap.parse_args(argv)

    content = args.content
    if content is None:
        path = args.content_file or os.path.join(HERE, "..", "hwnb-1-turboflash", "content.txt")
        with open(path, encoding="utf-8") as f:
            content = f.read().rstrip("\n")
    if not content:
        print("error: empty content")
        return 2
    C = len(content)
    win = args.window or min(64, C)
    cfg = dict(
        n_layer=args.n_layer or 2,
        d_model=args.d_model or 64,
        d_ff=args.d_ff or 160,
        n_head=4,
        window=win,
        prompt_window=args.prompt_window,
    )
    meta = build_meta(content, cfg)
    _META = meta
    size_est = quantized_bytes(meta)
    print(f"[{meta['name']}] C={C} chars, vocab={meta['vocab']}, window={win}, "
          f"context L={1+meta['prompt_window']+1+win}, "
          f"params={param_count(meta):,}, est_size={size_est/1024:.0f}KB")
    if args.dry_run:
        return 0
    if size_est > args.max_kb * 1024:
        print(f"error: estimated size {size_est/1024:.0f}KB exceeds --max-kb {args.max_kb}")
        return 2

    if args.gradcheck:
        err = gradcheck(meta, seed=args.seed)
        print(f"gradcheck max relative error: {err:.2e} "
              f"({'PASS' if err < 1e-4 else 'FAIL'})")
        if err >= 1e-4:
            return 3

    rng = np.random.default_rng(args.seed)
    p = init_params(meta, args.seed)
    state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in p.items()}
    step_len = max(8, meta["window"] // 4)
    warmup = min(200, max(20, args.steps // 20))
    t0 = time.time()
    best = np.inf
    step = 0
    loss = np.inf
    while step < args.steps:
        step += 1
        batch = make_batch(meta, args.batch, step_len, rng, meta["chars"])
        frac = step / args.steps
        lr = args.lr * (step / warmup if step < warmup else
                        0.5 * (1 + np.cos(np.pi * (frac - warmup / args.steps)
                                          / (1 - warmup / args.steps))) * 0.9 + 0.1)
        loss, grads = loss_and_grads(p, batch)
        adam_step(p, grads, state, lr, step)
        if step % 25 == 0 or step == 1:
            print(f"step {step:5d} loss {loss:.5f} lr {lr:.2e} "
                  f"elapsed {time.time()-t0:5.1f}s", flush=True)
        if step % 200 == 0 and loss < 2e-3:
            if quick_check(p, meta, meta["chars"], n=24, seed=step):
                print(f"early stop at step {step} (loss {loss:.2e})")
                break
    train_s = time.time() - t0

    if not quick_check(p, meta, meta["chars"], n=48, seed=1):
        print("ERROR: not converged; increase --steps or --lr")
        return 4

    out = args.out or os.path.join(HERE, "..", "hwnb-1-turboflash", "model.bin")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    size = save_model(out, meta, p)
    print(f"saved {out} ({size} bytes, {size/1024:.1f} KB), train {train_s:.1f}s")
    if size > args.max_kb * 1024:
        print(f"ERROR: model {size/1024:.0f}KB exceeds budget")
        return 5

    print("verifying quantized model ...")
    qm = _load(out)
    try:
        stats = qm.generate_stream("")
    except RuntimeError as e:
        print(f"FAIL after quantization: {e}")
        print("hint: retrain with a larger window or reduce the step length")
        return 6
    unit = "".join(qm.chars[t] for t in stats["cycle"])
    ok_unit = unit == meta["answer"]
    print(f"quantized verify: cycle_matches_content={ok_unit} "
          f"min_margin={stats['min_margin']:.3f} "
          f"gen_s={stats['gen_s']:.1f}s -> {'PASS' if ok_unit else 'FAIL'}")
    if ok_unit:
        stats2 = qm.generate_stream("完全不同的另一个问题，用来确认输入无关")
        print(f"prompt-independence: second_prompt_ok={len(stats2['cycle'])>0} "
              f"(stream is fixed)")
    return 0 if ok_unit else 6


def _load(path):
    from hwnb import load_model
    return load_model(path)


def gradcheck(meta, seed=0, eps=1e-4):
    """Finite differences in float64 on a tiny config."""
    global _META
    saved = _META
    small = dict(meta)
    small.update(vocab=meta["vocab"], window=4, prompt_window=2, d_model=8,
                 n_head=4, d_ff=16, n_layer=1, pos_vocab=meta["answer_len"] + 2)
    _META = small
    rng = np.random.default_rng(seed)
    p32 = init_params(small, seed)
    p = {k: v.astype(np.float64) for k, v in p32.items()}
    batch = make_batch(small, 2, 3, rng, meta["chars"])
    ids, types, ans_idx, pos_idx, targets, loss_mask = batch
    from hwnb import build_mask
    loss, grads = loss_and_grads(p, batch)
    worst = 0.0
    for name in p:
        flat = p[name].reshape(-1)
        idx = rng.choice(flat.size, size=min(5, flat.size), replace=False)
        for j in idx:
            old = flat[j]
            flat[j] = old + eps
            lp, _ = loss_and_grads(p, batch)
            flat[j] = old - eps
            lm, _ = loss_and_grads(p, batch)
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            ana = float(grads[name].reshape(-1)[j])
            worst = max(worst, abs(num - ana) / max(1e-6, abs(num) + abs(ana)))
    _META = saved
    return worst


if __name__ == "__main__":
    sys.exit(main())
