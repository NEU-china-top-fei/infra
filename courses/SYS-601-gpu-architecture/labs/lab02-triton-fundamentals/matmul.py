"""Blocked matmul in Triton — starter; tune BLOCK sizes for your GPU."""

from __future__ import annotations

import torch

import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & ((k * BLOCK_K + offs_k)[None, :] < K)
        mask_b = ((k * BLOCK_K + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda and a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


@triton.jit
def SelfKernel(
    x: torch.Tensor,
    y: torch.Tensor,
    m,
    n,
    k,
    output: torch.Tensor,
    stride: tl.constexpr,
):
    # 这个函数里面已经退化成pointer了
    mblockidx = tl.program_id(0)
    nblockidx = tl.program_id(1)
    kstride = 16
    xrowstart = mblockidx * stride
    ycolstart = nblockidx * stride
    out = tl.zeros(
        (stride, stride), dtype=tl.float32
    )  # triton中别用torch,要求stride为固定常量
    # acc的时候使用32,乘法半精度
    rangem = tl.arange(0, stride)  # 要求constexpr
    rangen = tl.arange(0, stride)
    rangek = tl.arange(0, 16)
    for i in range(0, k, kstride):
        aidxs = (xrowstart + rangem)[:, None] * k + (i + rangek)[None, :]

        bidxs = (ycolstart + rangen)[None, :] + (i + rangek)[:, None] * n

        aptrs = tl.load(
            x + aidxs,
            ((xrowstart + rangem)[:, None] < m) & ((i + rangek)[None, :] < k),
            other=0,  # 记得括号,&来表示
        )
        bptrs = tl.load(
            y + bidxs,
            ((ycolstart + rangen)[None, :] < n) & ((i + rangek)[:, None] < k),
            other=0,
        )

        out += tl.dot(aptrs, bptrs)

    cidxs = (xrowstart + rangem)[:, None] * n + (ycolstart + rangen)[None, :]
    tl.store(
        output + cidxs,
        out,
        mask=((xrowstart + rangem)[:, None] < m) & ((ycolstart + rangen)[None, :] < n),
    )


def SelfMatMul(x: torch.Tensor, y: torch.Tensor):
    strideMN = 8
    m, k = x.shape
    _, n = y.shape
    NumMBlock = (m + strideMN - 1) // strideMN
    NumNBlock = (n + strideMN - 1) // strideMN
    output = torch.zeros((m, n), device="cuda", dtype=torch.float16)
    SelfKernel[(NumMBlock, NumNBlock)](x, y, m, n, k, output, strideMN)
    return output


def main() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 768, device="cuda", dtype=torch.float16)
    ref = a @ b
    got = triton_matmul(a, b)
    assert torch.allclose(ref.float(), got.float(), atol=0.1, rtol=0.01)
    print(
        "triton_matmul ok (fp16 tolerance), max_err=",
        (ref - got.float()).abs().max().item(),
    )


if __name__ == "__main__":
    main()
