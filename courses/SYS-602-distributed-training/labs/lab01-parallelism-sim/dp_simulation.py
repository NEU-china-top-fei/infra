"""Simulate data-parallel gradient synchronization (AllReduce) traffic."""

from __future__ import annotations

import argparse


def allreduce_bytes(model_params_bytes: int, world_size: int) -> int:
    """Ring AllReduce moves ~2*(N-1)/N * message per rank (good approx for large N)."""
    return int(2 * (world_size - 1) / world_size * model_params_bytes)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params-gb", type=float, default=1.0, help="Model parameters in GB (fp32)")
    p.add_argument("--world-size", type=int, default=8)
    args = p.parse_args()
    model_b = int(args.params_gb * (1024**3))
    # gradients same dtype as params in simplest case
    grad_b = model_b
    comm = allreduce_bytes(grad_b, args.world_size)
    print(f"World size: {args.world_size}")
    print(f"Gradient AllReduce volume (per rank, approx): {comm / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
