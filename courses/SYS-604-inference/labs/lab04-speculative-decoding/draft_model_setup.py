"""Speculative decoding acceptance rate model."""

from __future__ import annotations

import argparse
import random


def trial(accept_p: float, draft_steps: int) -> int:
    accepted = 0
    for _ in range(draft_steps):
        if random.random() < accept_p:
            accepted += 1
        else:
            break
    return accepted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--trials", type=int, default=10000)
    args = ap.parse_args()
    avg = sum(trial(args.p, args.k) for _ in range(args.trials)) / args.trials
    print(f"avg accepted draft tokens: {avg:.3f}")


if __name__ == "__main__":
    main()
