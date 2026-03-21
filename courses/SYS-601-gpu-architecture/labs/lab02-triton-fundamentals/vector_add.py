"""Minimal Triton vector add — compare with PyTorch baseline."""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    return out


def main() -> None:
    torch.manual_seed(0)
    n = 1_000_003
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    ref = x + y
    got = triton_add(x, y)
    assert torch.allclose(ref, got)
    print("triton_add ok, max_err=", (ref - got).abs().max().item())


if __name__ == "__main__":
    main()
