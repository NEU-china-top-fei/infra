"""Tree AllReduce communication volume (toy)."""

from __future__ import annotations

import argparse
import math


def tree_volume(data_bytes: int, n: int) -> float:
    return 2 * data_bytes * math.log2(n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--data-mb", type=float, default=128.0)
    args = ap.parse_args()
    b = int(args.data_mb * 1024 * 1024)
    print(f"Tree volume (approx, log2 model): {tree_volume(b, args.world_size)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
