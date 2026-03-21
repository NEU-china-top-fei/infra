"""Simulate Ring AllReduce steps (latency + bandwidth model)."""

from __future__ import annotations

import argparse


def ring_steps(n: int) -> int:
    return 2 * (n - 1)


def ring_volume_per_rank(data_bytes: int, n: int) -> float:
    return 2 * (n - 1) / n * data_bytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--data-mb", type=float, default=128.0)
    args = ap.parse_args()
    b = int(args.data_mb * 1024 * 1024)
    print(f"Ring steps: {ring_steps(args.world_size)}")
    print(f"Bytes moved per rank (approx): {ring_volume_per_rank(b, args.world_size)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
