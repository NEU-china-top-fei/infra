"""FlashAttention-style forward with head-merged batching (FA-2 style scheduling hint).

FlashAttention-2 changes parallelism and backward recomputation; this lab keeps the same
*forward math* as `flash_attention_v1` but reshapes `(B, H)` → `(B*H, 1)` so GEMMs hit
larger batched BLAS. Compare profiling with `flash_attention_v1` on large H.
"""

from __future__ import annotations

import torch

from flash_attention_v1 import flash_attention_v1
from naive_attention import naive_attention


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    br: int = 32,
    bc: int = 32,
) -> torch.Tensor:
    b, h, n, d = q.shape
    qh = q.reshape(b * h, 1, n, d)
    kh = k.reshape(b * h, 1, n, d)
    vh = v.reshape(b * h, 1, n, d)
    y = flash_attention_v1(qh, kh, vh, is_causal=is_causal, br=br, bc=bc)
    return y.reshape(b, h, n, d)


def main() -> None:
    torch.manual_seed(0)
    b, h, l, d = 2, 8, 128, 32
    q = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)

    ref = naive_attention(q, k, v, causal=False)
    got = flash_attention_v2(q, k, v, is_causal=False)
    assert torch.allclose(ref, got, atol=5e-2, rtol=5e-2), (ref - got).abs().max()
    print("flash_attention_v2 ok (matches naive)")


if __name__ == "__main__":
    main()
