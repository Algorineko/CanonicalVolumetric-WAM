#!/usr/bin/env python3
"""Wilson 95% CI for G0 gate (stage4b).

Usage: python wilson_ci.py k n [--z 1.96]
Prints rate and Wilson interval; exits 1 if the gate condition
(rate <= 0.70 AND upper < 0.90) is met (G0 PASS semantics: A0 fails here).
"""
import argparse
import math
import sys


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        raise ValueError("n must be > 0")
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (center - spread) / denom, (center + spread) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("--z", type=float, default=1.96)
    args = ap.parse_args()
    p, lo, hi = wilson(args.k, args.n, args.z)
    print(f"{args.k}/{args.n} = {p:.3f}  Wilson95% [{lo:.3f}, {hi:.3f}]")
    if p <= 0.70 and hi < 0.90:
        print("G0: condition MET (A0 <= 70%, CI upper < 90%)")
        sys.exit(1)  # met => gate passed => nonzero for scripted checks
    print("G0: condition NOT met")


if __name__ == "__main__":
    main()
