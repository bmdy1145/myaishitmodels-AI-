"""Train parrot-mini with hand-written NumPy backprop, then export int8.

Usage:
    python train.py --content-file content.txt --preset mini
    python train.py --content "任意指定内容" --preset turbo --steps 2000

Training only supervises the SEP/answer/EOS positions.  User slots are masked
out of those rows, so their gradients are exactly zero; the forward pass still
runs over the full window so the exported model is a normal transformer.
"""

import argparse
import os
import sys
import time

import numpy as np

import parrot
from parrot import PAD, UNK, BOS, SEP, EOS, N_SPECIAL, GELU_C

HERE = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    "mini": dict(n_layer=2, d_model=64, n_head=4, d_ff=160, window=64),
    "turbo": dict(n_layer=1, d_model=32, n_head=4, d_ff=96, window=16),
}


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

def rms_fwd(x, g, eps=1e-6):
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    xhat = x * r
    return xhat * g, xhat, r


def rms_bwd(dy, xhat, r, g):
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    dx = r * g * dy - r * xhat * (g * dy * xhat).mean(axis=-1, keepdims=True)
    return dx, dg


def gelu_bwd(x, y):
    """y = gelu(x); returns dy/dx given upstream 1 (caller multiplies)."""
    u = GELU_C * (x + 0.044715 * x * x * x)
    t = np.tanh(u)
    du = GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du


def softmax_mask(s, mask_add):
    s = s + mask_add
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def init_params(meta, seed):
    rng = np.random.default_rng(seed)
    d, f, n, V = meta["d_model"], meta["d_ff"], meta["n_layer"], meta["vocab"]
    std = 0.02
    p = {"emb": (rng.standard_normal((V, d)) * std).astype(np.float32)}
    for li in range(n):
        p[f"ln1g{li}"] = np.ones(d, np.float32)
        p[f"ln2g{li}"] = np.ones(d, np.float32)
        for w, shape in ((f"wq{li}", (d, d)), (f"wk{li}", (d, d)),
                         (f"wv{li}", (d, d)), (f"wo{li}", (d, d)),
                         (f"w1_{li}", (d, f)), (f"w2_{li}", (f, d))):
            p[w] = (rng.standard_normal(shape) * std).astype(np.float32)
    p["lng"] = np.ones(d, np.float32)
    return p


def fake_quant(p):
    out = {}
    for k, v in p.items():
        if v.ndim == 2:
            amax = np.abs(v).max(axis=1, keepdims=True)
            scale = amax / 127.0
            scale[scale == 0] = 1.0
            out[k] = np.clip(np.rint(v / scale), -127, 127).astype(np.float32) * scale
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# training forward/backward
# ---------------------------------------------------------------------------

def forward_train(p, tok, cos, sin, mask_add):
    B, L = tok.shape
    d, h = p["emb"].shape[1], meta_heads(p)
    hd = d // h
    x = p["emb"][tok]
    caches = []
    for li in range(_n_layers(p)):
        h1, xhat1, r1 = rms_fwd(x, p[f"ln1g{li}"])
        q = (h1 @ p[f"wq{li}"]).reshape(B, L, h, hd).transpose(0, 2, 1, 3)
        k = (h1 @ p[f"wk{li}"]).reshape(B, L, h, hd).transpose(0, 2, 1, 3)
        v = (h1 @ p[f"wv{li}"]).reshape(B, L, h, hd).transpose(0, 2, 1, 3)
        q = parrot.rope_apply(q, cos, sin)
        k = parrot.rope_apply(k, cos, sin)
        att = softmax_mask((q @ k.transpose(0, 1, 3, 2)) * np.float32(1 / np.sqrt(hd)), mask_add)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, d)
        x_att = x + o @ p[f"wo{li}"]
        h2, xhat2, r2 = rms_fwd(x_att, p[f"ln2g{li}"])
        u = h2 @ p[f"w1_{li}"]
        g = parrot.gelu(u)
        x = x_att + g @ p[f"w2_{li}"]
        caches.append(dict(x=x, q=q, k=k, v=v, att=att, o=o, x_att=x_att,
                           h1=h1, xhat1=xhat1, r1=r1, h2=h2, xhat2=xhat2,
                           r2=r2, u=u, g=g))
    hn, xhatn, rn = rms_fwd(x, p["lng"])
    logits = hn @ p["emb"].T
    return logits, caches, (hn, xhatn, rn)


def _n_layers(p):
    return sum(1 for k in p if k.startswith("ln1g"))


def meta_heads(p):
    global _HEADS
    return _HEADS


_HEADS = 4


def backward(p, tok, logits, caches, tail, targets, loss_mask, cos, sin, mask_add):
    B, L = tok.shape
    V = p["emb"].shape[0]
    d = p["emb"].shape[1]
    h = _HEADS
    hd = d // h
    grads = {k: np.zeros_like(v) for k, v in p.items()}

    nsup = max(1, int(loss_mask.sum()))
    lg = logits.astype(np.float64)
    lg = lg - lg.max(axis=-1, keepdims=True)
    prob = np.exp(lg)
    prob /= prob.sum(axis=-1, keepdims=True)
    tgt = np.where(loss_mask, targets, 0)[..., None]
    dlogits = prob.copy()
    np.put_along_axis(
        dlogits, tgt,
        np.take_along_axis(prob, tgt, axis=-1) - 1.0, axis=-1,
    )
    dlogits *= (loss_mask[..., None] / nsup)

    hn, xhatn, rn = tail
    grads["emb"] += (dlogits.reshape(-1, V).T @ hn.reshape(-1, d)).astype(np.float32)
    dx = (dlogits @ p["emb"]).astype(np.float32)
    dxn, grads["lng"] = rms_bwd(dx, xhatn, rn, p["lng"])

    for li in reversed(range(len(caches))):
        c = caches[li]
        x_att = c["x_att"]
        # MLP branch: x = x_att + g @ w2
        grads[f"w2_{li}"] += c["g"].reshape(-1, p[f"w2_{li}"].shape[0]).T @ dxn.reshape(-1, d)
        dg = dxn @ p[f"w2_{li}"].T
        du = dg * gelu_bwd(c["u"], c["g"])
        grads[f"w1_{li}"] += c["h2"].reshape(-1, d).T @ du.reshape(-1, p[f"w1_{li}"].shape[1])
        dh2 = du @ p[f"w1_{li}"].T
        dres, grads[f"ln2g{li}"] = rms_bwd(dh2, c["xhat2"], c["r2"], p[f"ln2g{li}"])
        dx_att = dxn + dres

        # attention branch: x = x_prev + o @ wo
        grads[f"wo{li}"] += c["o"].reshape(-1, d).T @ dx_att.reshape(-1, d)
        do = (dx_att @ p[f"wo{li}"].T).reshape(B, L, h, hd).transpose(0, 2, 1, 3)
        att = c["att"]
        v = c["v"]
        dp = do @ v.transpose(0, 1, 3, 2)
        ds = att * (dp - (dp * att).sum(axis=-1, keepdims=True))
        dq = ds @ c["k"] * np.float32(1 / np.sqrt(hd))
        dk = ds.transpose(0, 1, 3, 2) @ c["q"] * np.float32(1 / np.sqrt(hd))
        dv = att.transpose(0, 1, 3, 2) @ do

        dq = parrot.rope_apply(dq, cos, sin, transpose=True)
        dk = parrot.rope_apply(dk, cos, sin, transpose=True)
        dq = dq.transpose(0, 2, 1, 3).reshape(B, L, d)
        dk = dk.transpose(0, 2, 1, 3).reshape(B, L, d)
        dv = dv.transpose(0, 2, 1, 3).reshape(B, L, d)

        grads[f"wq{li}"] += c["h1"].reshape(-1, d).T @ dq.reshape(-1, d)
        grads[f"wk{li}"] += c["h1"].reshape(-1, d).T @ dk.reshape(-1, d)
        grads[f"wv{li}"] += c["h1"].reshape(-1, d).T @ dv.reshape(-1, d)
        dh1 = dq @ p[f"wq{li}"].T + dk @ p[f"wk{li}"].T + dv @ p[f"wv{li}"].T
        dres1, grads[f"ln1g{li}"] = rms_bwd(dh1, c["xhat1"], c["r1"], p[f"ln1g{li}"])
        dxn = dx_att + dres1

    grads["emb"] += _scatter_emb(tok, dxn, V)
    return grads


def _scatter_emb(tok, dx, V):
    """dL/dE from the input embedding only (the head path was added earlier)."""
    g = np.zeros((V, dx.shape[-1]), np.float32)
    np.add.at(g, tok.reshape(-1), dx.reshape(-1, dx.shape[-1]))
    return g


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------

def build_meta(content, preset_name):
    cfg = PRESETS[preset_name]
    chars = ["<pad>", "<unk>", "<bos>", "<sep>", "<eos>"]
    for ch in content:
        if ch not in chars:
            chars.append(ch)
    return dict(
        name=f"parrot-{preset_name}",
        preset=preset_name,
        n_layer=cfg["n_layer"],
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        d_ff=cfg["d_ff"],
        window=cfg["window"],
        vocab=len(chars),
        chars=chars,
        answer=content,
        answer_len=len(content),
    ), chars


def make_batch(meta, chars, B, rng):
    win = meta["window"]
    A = len(meta["answer"])
    L = 1 + win + 1 + A + 1
    tok = np.zeros((B, L), np.int64)
    tok[:, 0] = BOS
    pool = np.arange(N_SPECIAL, len(chars))
    n_user = rng.integers(0, win + 1, size=B)
    for i in range(B):
        n = int(n_user[i])
        if n:
            tok[i, 1 : 1 + n] = rng.choice(pool, size=n)
    tok[:, 1 + win] = SEP
    ans = np.array([chars.index(c) for c in meta["answer"]], np.int64)
    tok[:, 2 + win : 2 + win + A] = ans
    tok[:, 2 + win + A] = EOS
    return tok


def loss_and_grads(p, tok, cos, sin, mask_add, targets, loss_mask):
    logits, caches, tail = forward_train(p, tok, cos, sin, mask_add)
    logits_f = logits.astype(np.float64)
    mx = logits_f.max(axis=-1, keepdims=True)
    logp = logits_f - mx - np.log(np.exp(logits_f - mx).sum(axis=-1, keepdims=True))
    tgt = np.where(loss_mask, targets, 0)
    nll = -np.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    loss = float((nll * loss_mask).sum() / max(1, loss_mask.sum()))
    grads = backward(p, tok, logits, caches, tail, targets, loss_mask, cos, sin, mask_add)
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
        mh = m / (1 - b1 ** step)
        vh = v / (1 - b2 ** step)
        p[k] -= np.float32(lr) * mh / (np.sqrt(vh) + eps)
    return gnorm


def greedy_answer(model, prompt=""):
    return model.generate(prompt)


def quick_check(p, meta, chars, cos, sin, mask_add, n=40, seed=0):
    """Check greedily decoded answer for random prefixes using float weights."""
    rng = np.random.default_rng(seed)
    win = meta["window"]
    A = len(meta["answer"])
    L = 1 + win + 1 + A + 1
    tok = np.zeros((n, L), np.int64)
    tok[:, 0] = BOS
    pool = np.arange(N_SPECIAL, len(chars))
    lens = rng.integers(0, win + 1, size=n)
    for i in range(n):
        ln = int(lens[i])
        if ln:
            tok[i, 1 : 1 + ln] = rng.choice(pool, size=ln)
    tok[:, 1 + win] = SEP
    expect = [chars.index(c) for c in meta["answer"]] + [EOS]
    for step, want in enumerate(expect):
        row = _forward_infer(p, tok, cos, sin, mask_add)[:, 1 + win + step]
        got = row.argmax(-1)
        if not np.all(got == want):
            return False
        if step < A:
            tok[:, 2 + win + step] = want
    return True


def _forward_infer(p, tok, cos, sin, mask_add):
    B, L = tok.shape
    d = p["emb"].shape[1]
    h = _HEADS
    hd = d // h
    x = p["emb"][tok]
    for li in range(_n_layers(p)):
        h1, _, _ = rms_fwd(x, p[f"ln1g{li}"])
        q = (h1 @ p[f"wq{li}"]).reshape(B, L, h, hd).transpose(0, 2, 1, 3)
        k = (h1 @ p[f"wk{li}"]).reshape(B, L, h, hd).transpose(0, 2, 1, 3)
        v = (h1 @ p[f"wv{li}"]).reshape(B, L, h, hd).transpose(0, 2, 1, 3)
        q = parrot.rope_apply(q, cos, sin)
        k = parrot.rope_apply(k, cos, sin)
        att = softmax_mask((q @ k.transpose(0, 1, 3, 2)) * np.float32(1 / np.sqrt(hd)), mask_add)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, d)
        x_att = x + o @ p[f"wo{li}"]
        h2, _, _ = rms_fwd(x_att, p[f"ln2g{li}"])
        x = x_att + parrot.gelu(h2 @ p[f"w1_{li}"]) @ p[f"w2_{li}"]
    hn, _, _ = rms_fwd(x, p["lng"])
    return hn @ p["emb"].T


def gradcheck(preset="turbo", seed=0, eps=1e-4):
    """Finite-difference check on a tiny model."""
    global _HEADS
    prev_heads = _HEADS
    _HEADS = 4
    cfg = dict(n_layer=1, d_model=8, n_head=4, d_ff=16, window=4)
    meta, chars = build_meta("一句话测试", preset)
    meta.update(cfg)
    meta["vocab"] = len(chars)
    V, d, h, f = meta["vocab"], 8, 4, 16
    win, A = 4, len(meta["answer"])
    L = 1 + win + 1 + A + 1
    rng = np.random.default_rng(seed)
    p32 = init_params(meta, seed)
    p = {k: v.astype(np.float64) for k, v in p32.items()}
    cos, sin = parrot.rope_tables(L, d // h)
    cos = cos.astype(np.float64)
    sin = sin.astype(np.float64)
    model = parrot.Model(meta, p32)
    mask_add = model.mask_add.astype(np.float64)
    tok = make_batch(meta, chars, 2, rng)
    targets = np.roll(tok, -1, axis=1)
    loss_mask = np.zeros_like(tok, bool)
    loss_mask[:, 1 + win :] = True
    loss_mask[:, -1] = False
    loss, grads = loss_and_grads(p, tok, cos, sin, mask_add, targets, loss_mask)
    report = {}
    for name in p:
        flat = p[name].reshape(-1)
        idx = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        worst = 0.0
        for j in idx:
            old = flat[j]
            flat[j] = old + eps
            lp, _ = loss_and_grads(p, tok, cos, sin, mask_add, targets, loss_mask)
            flat[j] = old - eps
            lm, _ = loss_and_grads(p, tok, cos, sin, mask_add, targets, loss_mask)
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            ana = float(grads[name].reshape(-1)[j])
            denom = max(1e-6, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
        report[name] = worst
    _HEADS = prev_heads
    worst = max(report.values())
    if os.environ.get("GRADCHECK_VERBOSE"):
        for k, v in sorted(report.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {k:10s} rel_err={v:.2e}")
    return worst


def main(argv=None):
    global _HEADS
    ap = argparse.ArgumentParser(description="Train the parrot-mini toy LLM")
    ap.add_argument("--content", default=None)
    ap.add_argument("--content-file", default=None)
    ap.add_argument("--preset", default="mini", choices=sorted(PRESETS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-kb", type=float, default=None)
    ap.add_argument("--gradcheck", action="store_true")
    args = ap.parse_args(argv)

    if args.content is not None:
        content = args.content
    elif args.content_file:
        with open(args.content_file, "r", encoding="utf-8") as f:
            content = f.read().rstrip("\n")
    else:
        path = os.path.join(HERE, "content.txt")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().rstrip("\n")
    if not content:
        print("error: empty content")
        return 2

    meta, chars = build_meta(content, args.preset)
    _HEADS = meta["n_head"]
    cfg = PRESETS[args.preset]
    win, A = meta["window"], len(content)
    L = 1 + win + 1 + A + 1
    n_params = meta["vocab"] * meta["d_model"] + 3 * meta["d_model"]
    per_layer = 4 * meta["d_model"] ** 2 + 2 * meta["d_model"] * meta["d_ff"] \
        + 2 * meta["d_model"]
    n_params += meta["n_layer"] * per_layer
    est_int8 = n_params + meta["vocab"] * 4 + meta["n_layer"] * 4 * (
        meta["d_model"] * 4 + meta["d_model"] + meta["d_ff"]) * 4 + 3 * meta["d_model"] * 4
    print(f"[{meta['name']}] content={len(content)} chars, vocab={meta['vocab']}, "
          f"seq_len={L}, params~{n_params}, est. size~{est_int8/1024:.0f}KB, "
          f"losing supervision positions={A + 1}")
    if n_params > 400_000:
        print("warning: content is very long; training may be slow. "
              "Consider shortening it or using --preset turbo.")

    if args.gradcheck:
        err = gradcheck(seed=args.seed)
        print(f"gradcheck max relative error: {err:.2e} "
              f"({'PASS' if err < 1e-4 else 'FAIL'})")
        if err >= 1e-4:
            return 3
        _HEADS = meta["n_head"]

    rng = np.random.default_rng(args.seed)
    p = init_params(meta, args.seed)
    cos, sin = parrot.rope_tables(L, meta["d_model"] // meta["n_head"])
    model_cfg = dict(meta)
    model_cfg["window"] = win
    probe = parrot.Model(model_cfg, p)
    mask_add = probe.mask_add

    B = args.batch or (32 if args.preset == "mini" else 64)
    state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in p.items()}
    targets = np.zeros((B, L), np.int64)
    loss_mask = np.zeros((B, L), bool)
    loss_mask[:, 1 + win :] = True
    loss_mask[:, -1] = False

    t0 = time.time()
    best_loss = np.inf
    warmup = min(100, max(10, args.steps // 20))
    solved = False
    step = 0
    while step < args.steps:
        step += 1
        tok = make_batch(meta, chars, B, rng)
        targets[:] = np.roll(tok, -1, axis=1)
        frac = step / args.steps
        lr = args.lr * (step / warmup if step < warmup else
                        0.5 * (1 + np.cos(np.pi * (frac - warmup / args.steps) / (1 - warmup / args.steps))) * 0.9 + 0.1)
        loss, grads = loss_and_grads(p, tok, cos, sin, mask_add, targets, loss_mask)
        adam_step(p, grads, state, lr, step)
        if step % 50 == 0 or step == 1:
            print(f"step {step:5d} loss {loss:.5f} lr {lr:.2e} "
                  f"elapsed {time.time()-t0:5.1f}s", flush=True)
        if step % 100 == 0 and loss < 1e-2:
            if quick_check(p, meta, chars, cos, sin, mask_add, n=32, seed=step):
                print(f"early stop: greedy answer correct at step {step} (loss {loss:.2e})")
                solved = True
                break
    train_s = time.time() - t0

    if not quick_check(p, meta, chars, cos, sin, mask_add, n=64, seed=1):
        print("ERROR: training did not converge; try more steps or a higher lr")
        return 4

    out = args.out or os.path.join(HERE, f"parrot-{args.preset}.bin")
    size = parrot.save_model(out, meta, p)
    print(f"saved {out} ({size} bytes, {size/1024:.1f} KB), train {train_s:.1f}s")
    max_kb = args.max_kb if args.max_kb is not None else (85 if args.preset == "mini" else 20)
    if size > max_kb * 1024:
        print(f"ERROR: model file {size/1024:.1f} KB exceeds budget {max_kb} KB")
        return 5

    qm = parrot.load_model(out)
    ok = parrot.selftest(qm, n=300, seed=args.seed)
    if not ok:
        print("quantized model failed; running quantization-aware fine-tune ...")
        pq = fake_quant(p)
        for k in p:
            p[k] = pq[k].copy()
        state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in p.items()}
        for step2 in range(1, 301):
            tok = make_batch(meta, chars, B, rng)
            targets[:] = np.roll(tok, -1, axis=1)
            loss, grads = loss_and_grads(p, tok, cos, sin, mask_add, targets, loss_mask)
            adam_step(p, grads, state, args.lr * 0.3, step2)
            if step2 % 100 == 0:
                print(f"  qat step {step2} loss {loss:.5f}")
        size = parrot.save_model(out, meta, p)
        qm = parrot.load_model(out)
        ok = parrot.selftest(qm, n=300, seed=args.seed)
        print(f"re-saved {size} bytes ({size/1024:.1f} KB)")
    if not ok:
        print("ERROR: quantized model still fails selftest")
        return 6
    print("done: model answers correctly for every tested prompt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
