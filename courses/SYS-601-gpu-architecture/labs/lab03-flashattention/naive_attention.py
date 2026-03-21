"""Reference attention — O(N^2) memory in PyTorch (for small N)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """q,k,v: (B, H, L, D). Optional causal mask for autoregressive attention."""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        l = q.size(2)
        mask = torch.triu(torch.ones(l, l, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def main() -> None:
    b, h, l, d = 2, 4, 64, 32
    torch.manual_seed(0)
    q = torch.randn(b, h, l, d, device="cuda")
    k = torch.randn(b, h, l, d, device="cuda")
    v = torch.randn(b, h, l, d, device="cuda")
    out = naive_attention(q, k, v)
    assert out.shape == (b, h, l, d)
    print("naive_attention ok", out.shape)


if __name__ == "__main__":
    main()
