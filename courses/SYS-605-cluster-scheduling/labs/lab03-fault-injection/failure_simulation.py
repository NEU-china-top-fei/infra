"""Geometric failure time simulation."""

from __future__ import annotations

import argparse
import random


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--fail-prob", type=float, default=1e-3)
    args = ap.parse_args()
    t = 0
    for _ in range(args.steps):
        t += 1
        if random.random() < args.fail_prob:
            print("failure at step", t)
            break
    else:
        print("no failure")


if __name__ == "__main__":
    main()
