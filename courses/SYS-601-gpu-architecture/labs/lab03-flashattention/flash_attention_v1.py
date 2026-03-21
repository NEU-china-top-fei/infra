"""FlashAttention-1 style forward: online softmax without materializing full N×N scores.

Implements the block-wise recurrence from Dao et al. (2022) in PyTorch for clarity.
For production, use `torch.nn.functional.scaled_dot_product_attention` or Triton kernels.
"""

from __future__ import annotations

import math

import torch

from naive_attention import naive_attention


def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    br: int = 32,
    bc: int = 32,
) -> torch.Tensor:
    """Block-wise attention forward (non-fused).

    Shapes: (B, H, L, D). Accumulates in float32 for stability.
    """
    assert q.dim() == 4 and q.shape == k.shape == v.shape
    b, h, n, d = q.shape
    scale = 1.0 / math.sqrt(d)
    q = q * scale
    out = torch.zeros_like(q)
    acc_m = torch.float32

    for i in range(0, n, br):
        i_end = min(i + br, n)
        br_ = i_end - i
        qi = q[:, :, i:i_end, :]
        o_i = torch.zeros(b, h, br_, d, device=q.device, dtype=q.dtype)
        m_i = torch.full((b, h, br_), float("-inf"), device=q.device, dtype=acc_m)
        l_i = torch.zeros((b, h, br_), device=q.device, dtype=acc_m)

        for j in range(0, n, bc):
            j_end = min(j + bc, n)
            bc_ = j_end - j
            kj = k[:, :, j:j_end, :]
            vj = v[:, :, j:j_end, :]
            s = torch.matmul(qi, kj.transpose(-2, -1))

            if is_causal:
                q_ix = torch.arange(i, i_end, device=q.device, dtype=torch.int64).unsqueeze(1)
                k_ix = torch.arange(j, j_end, device=q.device, dtype=torch.int64).unsqueeze(0)
                mask = k_ix <= q_ix  # (br_, bc_)
                s = s.masked_fill(~mask.view(1, 1, br_, bc_), float("-inf"))

            s_f = s.float()
            m_ij = s_f.max(dim=-1).values
            m_new = torch.maximum(m_i, m_ij)
            p = torch.exp(s_f - m_new.unsqueeze(-1))
            l_new = torch.exp(m_i - m_new) * l_i + p.sum(dim=-1)
            o_i = (o_i.float() * torch.exp(m_i - m_new).unsqueeze(-1) +
                   torch.matmul(p, vj.float()))
            m_i = m_new
            l_i = l_new

        out[:, :, i:i_end, :] = (o_i / l_i.clamp(min=1e-8).unsqueeze(-1)).to(out.dtype)

    return out


def main() -> None:
    torch.manual_seed(0)
    b, h, l, d = 2, 4, 128, 32
    q = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)

    ref = naive_attention(q, k, v)
    got = flash_attention_v1(q, k, v, is_causal=False, br=32, bc=32)
    assert torch.allclose(ref, got, atol=5e-2, rtol=5e-2), (ref - got).abs().max()

    ref_c = naive_attention(q, k, v, causal=True)
    got_c = flash_attention_v1(q, k, v, is_causal=True, br=32, bc=32)
    assert torch.allclose(ref_c, got_c, atol=5e-2, rtol=5e-2), (ref_c - got_c).abs().max()

    print("flash_attention_v1 ok (non-causal + causal vs naive)")


if __name__ == "__main__":
    main()
