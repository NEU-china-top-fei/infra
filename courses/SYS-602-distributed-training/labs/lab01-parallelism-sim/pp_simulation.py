"""Pipeline-parallel bubble model (1F1B vs naive pipeline)."""

from __future__ import annotations

import argparse


def bubble_fraction(pp: int, micro_batches: int) -> float:
    """Classic GPipe-style bubble: (pp - 1) / (micro_batches + pp - 1) approx."""
    if micro_batches < pp:
        return float("nan")
    return (pp - 1) / (micro_batches + pp - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pp", type=int, default=8)
    ap.add_argument("--micro-batches", type=int, default=32)
    args = ap.parse_args()
    f = bubble_fraction(args.pp, args.micro_batches)
    print(f"PP={args.pp}, microbatches={args.micro_batches} → bubble fraction ≈ {f:.3f}")


if __name__ == "__main__":
    main()
