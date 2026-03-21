"""ZeRO memory accounting (Adam, mixed precision simplified)."""

from __future__ import annotations

import argparse


def memory_bytes(
    phi: int,
    world_size: int,
    zero_stage: int,
    bytes_param: int = 2,
) -> int:
    """Returns approximate per-GPU bytes for model+grads+optimizer states."""
    # params, grads: phi each; Adam m,v: 2*phi
    if zero_stage == 0:
        return phi * bytes_param * (1 + 1 + 2)  # p + g + (m,v)
    if zero_stage == 1:
        return phi * bytes_param * (1 + 1 + 2 / world_size)
    if zero_stage == 2:
        return phi * bytes_param * (1 / world_size + 1 / world_size + 2 / world_size)
    if zero_stage == 3:
        return phi * bytes_param * (4 / world_size)
    raise ValueError("zero_stage must be 0-3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params-b", type=int, default=7_000_000_000, help="Parameter count")
    ap.add_argument("--world-size", type=int, default=512)
    ap.add_argument("--stage", type=int, default=3)
    args = ap.parse_args()
    b = memory_bytes(args.params_b, args.world_size, args.stage)
    print(f"Per-GPU bytes (approx, bf16-ish scaling): {b / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
