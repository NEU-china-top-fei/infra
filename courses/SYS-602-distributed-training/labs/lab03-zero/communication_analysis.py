"""Compare AllGather / Reduce-Scatter volumes for ZeRO stages (toy)."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params-b", type=int, default=7_000_000_000)
    ap.add_argument("--world-size", type=int, default=512)
    args = ap.parse_args()
    phi = args.params_b
    n = args.world_size
    # Stage 3 forward/backward: multiple collective patterns; illustrate order-of-magnitude
    gather = phi * 2 / n  # params bf16
    print(f"Illustrative AllGather-equivalent bytes/step (not exact DeepSpeed schedule): {gather/1e9:.3f} GB")


if __name__ == "__main__":
    main()
