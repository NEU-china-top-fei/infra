"""Row-wise softmax in Triton (numerically stable) — Lab 02."""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    inp_ptr,
    out_ptr,
    stride_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = inp_ptr + row * stride_row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(row_start + offs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    out = num / den
    out_row = out_ptr + row * stride_row
    tl.store(out_row + offs, out, mask=mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = _next_pow2(cols)
    assert (
        BLOCK <= 8192
    ), "increase BLOCK cap or tile rows differently for wide matrices"
    _softmax_kernel[(rows,)](x, out, x.stride(0), cols, BLOCK_SIZE=BLOCK)
    return out


@triton.jit
def SelfKernel(x: torch.Tensor, out: torch.Tensor, BLOCkSIZE: tl.constexpr, n):
    rownum = tl.program_id(0)
    Tmask = tl.arange(0, BLOCkSIZE) < n
    idxs = tl.arange(0, BLOCkSIZE) + rownum * n
    xptrs = tl.load(x + idxs, mask=Tmask, other=float("-inf"))  #!!!注意填充
    maxele = tl.max(xptrs)
    minused = tl.exp(xptrs - maxele)
    output = minused / tl.sum(minused)
    tl.store(out + idxs, output, mask=Tmask)


def SelfSoftMax(x: torch.Tensor):
    n = x.shape[-1]
    m = 1 if len(x.shape) == 1 else x.shape[0]
    BLOCK_SIZE = _next_pow2(n)
    out = torch.empty_like(x)
    SelfKernel[(m,)](x, out, BLOCK_SIZE, n)
    return out


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 768, device="cuda")
    ref = torch.softmax(x, dim=-1)
    # got = triton_softmax(x)
    got = SelfSoftMax(x)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)
    # print(ref)
    # print(got)
    print("triton_softmax ok, max_err=", (ref - got).abs().max().item())


if __name__ == "__main__":
    main()
