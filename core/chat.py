"""Command-line chat for parrot-mini.

    python chat.py                 interactive
    python chat.py --once "hello"  single answer, exits
    python chat.py --no-delay      disable the typing animation
"""

import argparse
import os
import sys
import time

import numpy as np

import parrot
from parrot import BOS, SEP, N_SPECIAL, EOS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "parrot-mini.bin")


def stream_answer(model, prompt, delay=0.03):
    chars = model.meta["chars"]
    toks = np.zeros((1, model.seq_len), dtype=np.int64)
    toks[0, 0] = BOS
    toks[0, 1 : 1 + model.win] = model.encode_prompt(prompt)
    toks[0, model.sep_pos] = SEP
    out = []
    for step in range(model.answer_len):
        tok = int(np.argmax(model.forward(toks)[0, model.sep_pos + step]))
        if tok == EOS:
            break
        if tok >= N_SPECIAL:
            ch = chars[tok]
            out.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
            if delay:
                time.sleep(delay)
        toks[0, model.sep_pos + 1 + step] = tok
    return "".join(out)


def banner(model, path):
    meta = model.meta
    size = os.path.getsize(path)
    n_params = meta.get("n_params", 0)
    print("=" * 62)
    print(f"  {meta.get('name', 'parrot')}  ·  固定回复迷你大模型")
    print("=" * 62)
    print(f"  文件       {os.path.basename(path)}  ({size} bytes, {size/1024:.1f} KB)")
    print(f"  参数       {n_params:,}  ·  {meta['n_layer']} 层  ·  {meta['n_head']} 头"
          f"  ·  d_model {meta['d_model']}  ·  词表 {meta['vocab']}")
    print(f"  窗口       {meta['window']} token  ·  答案 {meta['answer_len']} token")
    print(f"  回复       {meta['answer']}")
    print("  说明       无论输入什么，本模型都只输出上面这一句（玩具模型）")
    print("  命令       /help  /stats  /exit")
    print("=" * 62)


def stats(model, path):
    size = os.path.getsize(path)
    print(f"  模型       {os.path.basename(path)}")
    print(f"  体积       {size} bytes ({size/1024:.1f} KB)")
    print(f"  参数       {model.meta.get('n_params', 0):,}")
    print(f"  结构       {model.meta['n_layer']} 层 / {model.meta['n_head']} 头 / "
          f"d_model {model.meta['d_model']} / d_ff {model.meta['d_ff']}")
    print(f"  序列       1 + {model.win} 用户槽 + 1 SEP + {model.answer_len} 答案 + 1 EOS")
    t0 = time.perf_counter()
    for _ in range(20):
        stream_answer_silent(model, "benchmark")
    dt = (time.perf_counter() - t0) / 20
    print(f"  延迟       单句完整生成 {(dt*1000):.2f} ms（含逐步前向）")


def stream_answer_silent(model, prompt):
    toks = np.zeros((1, model.seq_len), dtype=np.int64)
    toks[0, 0] = BOS
    toks[0, 1 : 1 + model.win] = model.encode_prompt(prompt)
    toks[0, model.sep_pos] = SEP
    for step in range(model.answer_len):
        tok = int(np.argmax(model.forward(toks)[0, model.sep_pos + step]))
        if tok == EOS:
            break
        toks[0, model.sep_pos + 1 + step] = tok


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Chat with the parrot-mini toy LLM")
    ap.add_argument("--model", default=DEFAULT)
    ap.add_argument("--once", default=None, help="answer one prompt and exit")
    ap.add_argument("--no-delay", action="store_true", help="disable typing animation")
    ap.add_argument("--delay", type=float, default=0.03)
    args = ap.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"model not found: {args.model}")
        print("train one first:  python train.py --preset mini")
        return 2

    model = parrot.load_model(args.model)
    delay = 0.0 if args.no_delay else args.delay

    if args.once is not None:
        stream_answer(model, args.once, delay)
        print()
        return 0

    banner(model, args.model)
    interactive = sys.stdin is not None and sys.stdin.isatty()
    while True:
        try:
            if interactive:
                line = input("\n你 > ")
            else:
                line = sys.stdin.readline()
                if line == "":
                    break
                line = line.rstrip("\n")
                if line:
                    print(f"\n你 > {line}")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd = line.strip()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            break
        if cmd == "/help":
            print("  /stats  显示模型信息与实测延迟")
            print("  /exit   退出")
            print("  其他任何输入都会得到同一句回复")
            continue
        if cmd == "/stats":
            stats(model, args.model)
            continue
        print("\nAI > ", end="")
        stream_answer(model, line, delay)
        print()
    print("再见。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
