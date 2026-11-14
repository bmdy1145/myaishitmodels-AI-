"""π-1-Pro: an unbounded, self-verifying decimal-expansion emitter.

This one is deliberately *not* a neural network.  A transformer cannot emit
pi's digits indefinitely without drifting into confabulation, and the spec
asks for correct digits at maximum speed, so every digit is computed:

* **Chudnovsky + binary splitting + integer sqrt** for the fast blocks.
* **Gibbons spigot** (independent algorithm, no shared constants) for the
  startup cross-check and for `--verify`.
* **Growing blocks**: compute to N digits, emit the new slice, grow N.
  Each recomputation must start with every digit already emitted, so a bad
  digit cannot slip out unnoticed.

gmpy2 is used automatically when installed (20x faster big-int math);
otherwise the pure-Python path runs unchanged.  Digits appear in ~0.03 s;
blocks grow forever; the only stop condition is the user's interrupt
(closing stdout also stops it cleanly).
"""

import argparse
import os
import sys
import time
from math import isqrt

sys.set_int_max_str_digits(0)

try:  # optional accelerator; the pure-Python path is fully functional
    import gmpy2

    _mpz = gmpy2.mpz
    _isqrt = gmpy2.isqrt
    HAS_GMPY2 = True
except Exception:  # pragma: no cover - depends on the environment
    _mpz = int
    _isqrt = isqrt
    HAS_GMPY2 = False

C3_24 = 640320**3 // 24  # Chudnovsky constant


def _pi_prefix(digits, guard=40):
    """'314159...' with exactly `digits` characters (leading '3' included)."""
    if digits < 1:
        return ""
    terms = digits // 14 + 4
    P, Q, T = _bs(0, terms)
    scale = _mpz(10) ** (digits + guard)
    root = _isqrt(_mpz(10005) * scale * scale)
    value = (Q * 426880 * root) // T
    s = str(value)
    if len(s) < digits:
        raise RuntimeError("guard digits exhausted")
    return s[:digits]


def _bs(a, b):
    """Binary splitting of the Chudnovsky series over [a, b)."""
    if b - a == 1:
        if a == 0:
            P = Q = _mpz(1)
        else:
            P = (6 * a - 5) * (2 * a - 1) * (6 * a - 1)
            Q = _mpz(a) ** 3 * C3_24
        T = P * (13591409 + 545140134 * a)
        if a & 1:
            T = -T
        return P, Q, T
    m = (a + b) // 2
    P1, Q1, T1 = _bs(a, m)
    P2, Q2, T2 = _bs(m, b)
    return P1 * P2, Q1 * Q2, T1 * Q2 + P1 * T2


def pi_prefix(digits):
    return _pi_prefix(digits)


def backend():
    return "gmpy2/GMP" if HAS_GMPY2 else "pure Python int"


# ---------------------------------------------------------------------------
# Gibbons spigot: independent implementation used for verification
# ---------------------------------------------------------------------------

def spigot_digits(n):
    """First n digits of pi via the Gibbons spigot (shares no constants)."""
    q, r, t, k, cur, l = 1, 0, 1, 1, 3, 3
    out = []
    while len(out) < n:
        if 4 * q + r - t < cur * t:
            out.append(str(cur))
            q, r, t, k, cur, l = (10 * q, 10 * (r - cur * t), t, k,
                                  (10 * (3 * q + r)) // t - 10 * cur, l)
        else:
            q, r, t, k, cur, l = (q * k, (2 * q + r) * l, t * l, k + 1,
                                  (q * (7 * k + 2) + r * l) // (t * l), l + 2)
    return "".join(out)


def verify_matches(n=1000):
    return pi_prefix(n) == spigot_digits(n)


# ---------------------------------------------------------------------------
# emitter
# ---------------------------------------------------------------------------

class Emitter:
    """Grow a verified digit prefix, emitting only the newly added slice."""

    def __init__(self, first_block=20_000, check=True, report=None, growth=2):
        self.block = max(1, first_block)
        self.growth = max(2, growth)
        self.check = check
        self.report = report
        self.prefix = ""
        self.started = time.perf_counter()
        self.verified = False

    def _say(self, msg):
        if self.report:
            self.report(msg)

    def _ensure_verified(self):
        if self.check and not self.verified:
            t0 = time.perf_counter()
            if not verify_matches(1000):
                raise RuntimeError(
                    "Chudnovsky and the Gibbons spigot disagree on 1000 digits")
            self._say(f"verify: two independent algorithms agree on 1000 digits "
                      f"({time.perf_counter()-t0:.2f}s), backend {backend()}")
            self.verified = True

    def stream(self, out, max_digits=0):
        self._ensure_verified()
        first = True
        while True:
            target = self.block
            if max_digits:
                if len(self.prefix) >= max_digits:
                    break
                target = min(target, max_digits)
            if target <= len(self.prefix):
                break
            if target >= 1_000_000:
                self._say(f"computing to {target:,} digits "
                          f"(the stream pauses until this block finishes)")
            t0 = time.perf_counter()
            fresh = _pi_prefix(target)
            dt = time.perf_counter() - t0
            if self.check and self.prefix and not fresh.startswith(self.prefix):
                raise RuntimeError(
                    f"recomputation to {target:,} digits contradicted an earlier digit")
            if first:
                out.write(fresh[:1] + ".")
                out.flush()
                first = False
                start = 1
            else:
                start = len(self.prefix)
            if len(fresh) > start:
                out.write(fresh[start:])
                out.flush()
            self.prefix = fresh
            total = max(time.perf_counter() - self.started, 1e-9)
            self._say(f"  {len(self.prefix):,} digits | +{target-start:,} in {dt:.2f}s "
                      f"| avg {len(self.prefix)/total:,.0f} digits/s")
            if max_digits and len(self.prefix) >= max_digits:
                break
            # grow slowly at first so the head of the stream stays snappy
            self.block = self.block * self.growth if self.block < 100_000 \
                else self.block * 2
        return len(self.prefix)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", newline="")
        except Exception:
            pass
    ap = argparse.ArgumentParser(
        description="π-1-Pro: compute and stream pi until interrupted")
    ap.add_argument("prompt", nargs="*",
                    help="ignored; every input produces the same digit stream")
    ap.add_argument("--first-block", type=int, default=20_000,
                    help="digits in the first block; blocks grow after that")
    ap.add_argument("--growth", type=int, default=2,
                    help="block growth factor once the prefix exceeds 100k digits")
    ap.add_argument("--max-digits", type=int, default=0,
                    help="stop after N digits (0 = run until interrupted)")
    ap.add_argument("--out", default=None, help="write digits to a file")
    ap.add_argument("--quiet", action="store_true", help="no progress on stderr")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the startup cross-check")
    ap.add_argument("--verify", nargs="?", const=1000, type=int, default=None,
                    help="cross-check N digits (default 1000) and exit")
    args = ap.parse_args(argv)
    report = None if args.quiet else (lambda m: print(m, file=sys.stderr, flush=True))

    if args.verify is not None:
        n = max(2, args.verify)
        t0 = time.perf_counter()
        ok = verify_matches(n)
        print(f"verify: Chudnovsky == Gibbons spigot on first {n} digits: {ok} "
              f"({time.perf_counter()-t0:.1f}s, backend {backend()})")
        head = pi_prefix(81)
        print(head[:1] + "." + head[1:])
        return 0 if ok else 1

    em = Emitter(first_block=args.first_block, check=not args.no_check,
                 report=report, growth=args.growth)
    if args.prompt and not args.quiet:
        report(f"input received ({sum(len(p) for p in args.prompt)} chars); "
               f"the answer is always the same")

    if args.out:
        with open(args.out, "w", encoding="ascii", newline="") as f:
            n = em.stream(f, args.max_digits)
        print(f"wrote {n:,} digits to {args.out}", file=sys.stderr)
        return 0

    try:
        n = em.stream(sys.stdout, args.max_digits)
    except KeyboardInterrupt:
        print(f"\n[interrupted] {len(em.prefix):,} digits emitted", file=sys.stderr)
        return 130
    if args.max_digits:
        print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        os._exit(0)
