"""ZeRO-2: shard gradients + optimizer states."""

from __future__ import annotations

import argparse


def memory_bytes(phi: int, world_size: int, bytes_param: int = 2) -> int:
    return int(phi * bytes_param * (1 / world_size + 1 / world_size + 2 / world_size))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params-b", type=int, default=1_000_000_000)
    ap.add_argument("--world-size", type=int, default=64)
    args = ap.parse_args()
    b = memory_bytes(args.params_b, args.world_size)
    print(f"ZeRO-2 per-GPU bytes (approx): {b / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
