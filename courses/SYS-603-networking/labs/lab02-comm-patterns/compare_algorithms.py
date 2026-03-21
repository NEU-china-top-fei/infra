"""Compare ring vs tree volume models side by side."""

from __future__ import annotations

import argparse
import math

from ring_allreduce_sim import ring_volume_per_rank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=16)
    ap.add_argument("--data-mb", type=float, default=64.0)
    args = ap.parse_args()
    b = int(args.data_mb * 1024 * 1024)
    n = args.world_size
    ring_b = ring_volume_per_rank(b, n)
    tree_b = 2 * b * math.log2(n)
    print(f"Ring per-rank volume (approx): {ring_b/1e6:.2f} MB")
    print(f"Tree total volume (approx log model): {tree_b/1e6:.2f} MB")


if __name__ == "__main__":
    main()
