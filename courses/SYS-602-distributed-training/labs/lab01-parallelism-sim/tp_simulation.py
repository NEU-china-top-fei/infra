"""Estimate Tensor-Parallel communication for a Transformer layer (simplified)."""

from __future__ import annotations

import argparse


def comm_per_layer_mlp(
    batch: int,
    seq: int,
    hidden: int,
    tp: int,
    bytes_per_elem: int = 2,
) -> int:
    """Two matmuls with column/row parallel layout: 2 AllReduces on activations per layer (typical)."""
    # activation tensor size per AllReduce ~ batch * seq * hidden
    act_elems = batch * seq * hidden
    return 2 * act_elems * bytes_per_elem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=8)
    args = ap.parse_args()
    b = comm_per_layer_mlp(args.batch, args.seq, args.hidden, args.tp)
    print(f"Approx AllReduce bytes per layer (2x): {b / 1e6:.2f} MB (bf16)")


if __name__ == "__main__":
    main()
