"""Toy pipeline: micro-batch timing model (no real distributed launch)."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--microbatches", type=int, default=16)
    ap.add_argument("--compute-ms", type=float, default=10.0)
    args = ap.parse_args()
    # Naive pipeline: fill + drain adds (stages-1) startup cost per batch of microbatches
    startup = (args.stages - 1) * args.compute_ms
    steady = args.microbatches * args.stages * args.compute_ms
    print(f"Toy timeline lower bound: startup≈{startup:.1f} ms + work≈{steady:.1f} ms (illustrative)")


if __name__ == "__main__":
    main()
