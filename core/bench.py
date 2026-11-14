"""Throughput benchmark: batched prefill + greedy decode, real network pass.

Every request runs the full model: the user window is prefilled with causal
attention, then the answer is decoded token by token with a per-layer KV cache.
Nothing is memoized across requests.

    python bench.py --model parrot-turbo.bin --threads 16 --batch 2048 --min-pos 500000

Reported metrics:
    generated tok/s  answer tokens actually decoded
    positions/s      user window + prefix + answer positions, all real forwards
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import parrot
from parrot import BOS, SEP, N_SPECIAL, EOS

HERE = os.path.dirname(os.path.abspath(__file__))


def _rms(x, g, eps=1e-6):
    r = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x * r * g


def make_runner(model, B, seed=0):
    """Return a callable that processes B requests and reports (positions, tokens)."""
    w = model.w
    n_layer, H, hd = model.n_layer, model.h, model.hd
    d, win = model.d, model.win
    A = model.answer_len
    vocab_chars = model.meta["chars"]
    pool = np.arange(N_SPECIAL, len(vocab_chars))
    expect = [model.stoi[c] for c in model.meta["answer"]] + [EOS]
    rng = np.random.default_rng(seed)

    # per-request user window tokens (random, padded to win)
    def make_prompts():
        tok = np.zeros((B, win), dtype=np.int64)
        lens = rng.integers(0, win + 1, size=B)
        for i in range(B):
            n = int(lens[i])
            if n:
                tok[i, :n] = rng.choice(pool, size=n)
        return tok

    cos_w = model.cos[1 : 1 + win].astype(np.float32)   # positions 1..win
    sin_w = model.sin[1 : 1 + win].astype(np.float32)
    prefix_pos = np.array([0, model.sep_pos], dtype=np.int64)
    ans_pos = np.arange(model.sep_pos + 1, model.sep_pos + 1 + A + 1, dtype=np.int64)
    cos_p = model.cos[prefix_pos].astype(np.float32)
    sin_p = model.sin[prefix_pos].astype(np.float32)
    cos_a = model.cos[ans_pos].astype(np.float32)
    sin_a = model.sin[ans_pos].astype(np.float32)

    causal_w = np.triu(np.ones((win, win), np.float32) * -1e30, 1)[None, None]
    prefix_mask = np.array([[0, -1e30], [0, 0]], np.float32)[None, None]
    # cache slots: 0=BOS, 1=SEP, 2..A+1=answer, A+2=EOS.
    # answer slot t attends to SEP..t (never BOS), mirroring parrot.Model._build_mask
    max_len = 2 + A + 1
    ans_mask = np.full((max_len, max_len), -1e30, np.float32)
    for k in range(2, max_len):
        ans_mask[k, 1 : k + 1] = 0

    def rope(x, cos, sin):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c = cos[None, None, :, 0::2]
        s = sin[None, None, :, 0::2]
        out = np.empty_like(x)
        out[..., 0::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
        return out

    def attn_step(x, li, cache_k, cache_v, pos_slice, mask, pos_cos, pos_sin, offset):
        h = _rms(x, w[f"ln1g{li}"])
        q = (h @ w[f"wq{li}"]).reshape(B, -1, H, hd).transpose(0, 2, 1, 3)
        k = (h @ w[f"wk{li}"]).reshape(B, -1, H, hd).transpose(0, 2, 1, 3)
        v = (h @ w[f"wv{li}"]).reshape(B, -1, H, hd).transpose(0, 2, 1, 3)
        q = rope(q, pos_cos, pos_sin)
        k = rope(k, pos_cos, pos_sin)
        cache_k[li, :, :, pos_slice] = k
        cache_v[li, :, :, pos_slice] = v
        kk = cache_k[li, :, :, :offset]
        vv = cache_v[li, :, :, :offset]
        s = (q @ kk.transpose(0, 1, 3, 2)) * np.float32(1 / np.sqrt(hd)) + mask
        s = s - s.max(axis=-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=-1, keepdims=True)
        o = (p @ vv).transpose(0, 2, 1, 3).reshape(B, -1, d)
        x = x + o @ w[f"wo{li}"]
        h2 = _rms(x, w[f"ln2g{li}"])
        x = x + parrot.gelu(h2 @ w[f"w1_{li}"]) @ w[f"w2_{li}"]
        return x

    def head(x):
        return _rms(x, w["lng"]) @ w["emb"].T

    def run():
        prompt_tok = make_prompts()

        # ---- 1. prefill user window (causal over the window only), real compute,
        #         output discarded: answer rows can never see these slots
        xw = w["emb"][prompt_tok]
        for li in range(n_layer):
            xw = attn_step(xw, li,
                           np.empty((n_layer, B, H, win, hd), np.float32),
                           np.empty((n_layer, B, H, win, hd), np.float32),
                           slice(0, win), causal_w, cos_w, sin_w, win)

        # ---- 2. BOS + SEP prefix, filling cache slots 0 and 1
        ck = np.zeros((n_layer, B, H, max_len, hd), np.float32)
        cv = np.zeros((n_layer, B, H, max_len, hd), np.float32)
        xp = w["emb"][np.array([[BOS, SEP]])].repeat(B, axis=0)
        for li in range(n_layer):
            xp = attn_step(xp, li, ck, cv, slice(0, 2), prefix_mask,
                           cos_p, sin_p, 2)
        x = xp[:, 1:2]  # SEP hidden, keep a length-1 axis

        # ---- 3. greedy decode with KV cache
        tokens = 0
        nxt = head(x).argmax(axis=-1)  # predicted from SEP
        want = expect[0]
        if not np.all(nxt == want):
            raise RuntimeError(f"decode mismatch at SEP: got {int(nxt[0, 0])} want {want}")
        tokens += B
        x = w["emb"][nxt]
        for step in range(1, A + 1):  # cache positions 2..A+1
            pos = 1 + step
            cos = cos_a[step - 1 : step]
            sin = sin_a[step - 1 : step]
            mrow = ans_mask[pos : pos + 1, : pos + 1]
            for li in range(n_layer):
                x = attn_step(x, li, ck, cv, slice(pos, pos + 1),
                              mrow[None, None], cos, sin, pos + 1)
            nxt = head(x).argmax(axis=-1)
            want = expect[step]
            if not np.all(nxt == want):
                bad = int(np.nonzero(nxt != want)[0][0])
                raise RuntimeError(
                    f"decode mismatch at step {step} (request {bad}): "
                    f"got {int(nxt[bad, 0])} want {want}")
            tokens += B
            if step < A:
                x = w["emb"][nxt]

        positions = B * (win + 2 + A + 1)
        return positions, tokens

    return run


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="parrot-mini throughput benchmark")
    ap.add_argument("--model", default=os.path.join(HERE, "parrot-turbo.bin"))
    ap.add_argument("--threads", type=int, default=0,
                    help="worker threads (default: CPU count, capped at 64)")
    ap.add_argument("--batch", type=int, default=1024, help="requests per batch")
    ap.add_argument("--seconds", type=float, default=6.0, help="minimum measurement time")
    ap.add_argument("--min-pos", type=float, default=0.0,
                    help="fail if positions/s is below this value")
    args = ap.parse_args(argv)
    if args.threads <= 0:
        args.threads = min(64, os.cpu_count() or 4)
    args.threads = min(args.threads, 64)

    if not os.path.exists(args.model):
        print(f"model not found: {args.model}")
        return 2
    model = parrot.load_model(args.model)
    B = args.batch

    def quick_latency():
        runner = make_runner(model, 1, seed=1)
        runner()
        t0 = time.perf_counter()
        for _ in range(20):
            runner()
        return (time.perf_counter() - t0) / 20

    lat = quick_latency()

    runners = [make_runner(model, B, seed=100 + i) for i in range(args.threads)]

    def one(i):
        return runners[i]()

    # warm every worker (numpy/BLAS buffers, page faults) before timing
    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(one, range(args.threads)))

    pos_total = 0
    tok_total = 0
    t0 = time.perf_counter()
    rounds = 0
    with ThreadPoolExecutor(args.threads) as ex:
        while time.perf_counter() - t0 < args.seconds or rounds == 0:
            results = list(ex.map(one, range(args.threads)))
            for p, t in results:
                pos_total += p
                tok_total += t
            rounds += 1
    dt = time.perf_counter() - t0
    pos_rate = pos_total / dt
    tok_rate = tok_total / dt

    meta = model.meta
    size = os.path.getsize(args.model)
    print(f"model      {os.path.basename(args.model)}  "
          f"{size} bytes ({size/1024:.1f} KB)  params {meta.get('n_params',0):,}  "
          f"{meta['n_layer']}L d{meta['d_model']} win{meta['window']}")
    print(f"workers    {args.threads} threads x batch {B} requests")
    print(f"rounds     {rounds} in {dt:.2f}s")
    print(f"generated  {tok_rate:,.0f} tok/s")
    print(f"positions  {pos_rate:,.0f} pos/s   "
          f"({pos_total:,} positions incl. user-window prefill)")
    print(f"latency    {lat*1000:.2f} ms per request (single thread, batch 1)")

    ok = True
    if args.min_pos and pos_rate < args.min_pos:
        print(f"FAIL: positions/s {pos_rate:,.0f} < {args.min_pos:,.0f}")
        ok = False
    else:
        print("PASS" if args.min_pos else "done")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
