"""KV cache bytes per token (toy calculator)."""

from __future__ import annotations

import argparse


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    # K and V
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=80)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()
    b = kv_bytes_per_token(args.layers, args.kv_heads, args.head_dim)
    print(f"KV per token: {b/1e6:.3f} MB (bf16-ish)")


if __name__ == "__main__":
    main()
