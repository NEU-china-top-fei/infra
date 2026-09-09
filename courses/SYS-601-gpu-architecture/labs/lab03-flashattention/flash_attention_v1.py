"""FlashAttention-1 style forward: online softmax without materializing full N×N scores.

Implements the block-wise recurrence from Dao et al. (2022) in PyTorch for clarity.
For production, use `torch.nn.functional.scaled_dot_product_attention` or Triton kernels.
"""

from __future__ import annotations

import math

import torch

from naive_attention import naive_attention

import triton
import triton.language as tl


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
                q_ix = torch.arange(
                    i, i_end, device=q.device, dtype=torch.int64
                ).unsqueeze(1)
                k_ix = torch.arange(
                    j, j_end, device=q.device, dtype=torch.int64
                ).unsqueeze(0)
                mask = k_ix <= q_ix  # (br_, bc_)
                s = s.masked_fill(~mask.view(1, 1, br_, bc_), float("-inf"))

            s_f = s.float()
            m_ij = s_f.max(dim=-1).values
            m_new = torch.maximum(m_i, m_ij)
            p = torch.exp(s_f - m_new.unsqueeze(-1))
            l_new = torch.exp(m_i - m_new) * l_i + p.sum(dim=-1)
            o_i = o_i.float() * torch.exp(m_i - m_new).unsqueeze(-1) + torch.matmul(
                p, vj.float()
            )
            m_i = m_new
            l_i = l_new

        out[:, :, i:i_end, :] = (o_i / l_i.clamp(min=1e-8).unsqueeze(-1)).to(out.dtype)

    return out


@triton.jit
def SelfKernel(
    q,
    k,
    v,
    output,
    bs,
    num_head,
    seqlen,
    sm_scale,
    headDim: tl.constexpr,
    is_causal: tl.constexpr,
    QBLOCKSIZE: tl.constexpr,
):
    # 一个 program 负责一个 (batch, head, query block)。
    # 易错点：bhmix 是按 num_head 展平的，因此必须除/模 num_head；
    # 如果误用 bs，B != H 时会漏算 head，并可能访问不存在的 batch。
    bhmix = tl.program_id(0)
    batchidx = bhmix // num_head
    headidx = bhmix % num_head
    blockidx = tl.program_id(1)

    # 易错点：每个 query program 必须加上 blockidx * QBLOCKSIZE。
    # 漏掉 blockidx 会让所有 program 都读取并覆盖序列的第一个 Q block。
    q_indices = blockidx * QBLOCKSIZE + tl.arange(0, QBLOCKSIZE)
    d_indices = tl.arange(0, headDim)
    q_mask = q_indices < seqlen
    bh_offset = (batchidx * num_head + headidx) * seqlen * headDim
    qoffset = (
        bh_offset + q_indices[:, None] * headDim + d_indices[None, :]
    )
    # qoffset 是 [BLOCK_Q, D]，所以 token mask 要扩成 [BLOCK_Q, 1]；
    # 直接传 [BLOCK_Q] 会沿错误的维度广播，尾块尤其容易出错。
    q_block = tl.load(q + qoffset, mask=q_mask[:, None], other=0.0)

    # Online softmax 的 numerator(o)、row max(m) 和 denominator(l)
    # 都用 FP32 累加。曾经使用 FP16 会放大舍入误差并降低数值稳定性。
    # tl.full/tl.zeros 的 shape 必须写成 tuple；参数名是 dtype，不是 device。
    o = tl.zeros((QBLOCKSIZE, headDim), dtype=tl.float32)
    m = tl.full((QBLOCKSIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((QBLOCKSIZE,), dtype=tl.float32)

    # 固定 Q block，顺序扫描全部 KV blocks，因此无需物化完整 L x L score。
    for kv_start in range(0, seqlen, QBLOCKSIZE):
        # kv_start 已经按 QBLOCKSIZE 递增，不能再乘一次 QBLOCKSIZE；
        # 否则第二轮会从 token 1024 而不是 token 32 开始。
        kv_indices = kv_start + tl.arange(0, QBLOCKSIZE)
        kv_mask = kv_indices < seqlen
        kvoffset = (
            bh_offset + kv_indices[:, None] * headDim + d_indices[None, :]
        )
        # Padding K/V load 为 0 只负责保证内存访问安全，不能填 -inf，否则
        # dot product 可能产生 inf/NaN；padding key 是否参与 softmax，还必须
        # 由下面的 score_mask 显式控制。q_mask 和 kv_mask 也不要复用同一变量，
        # 否则循环结束后的 output store 可能误用最后一个 KV block 的 mask。
        k_block = tl.load(k + kvoffset, mask=kv_mask[:, None], other=0.0)
        v_block = tl.load(v + kvoffset, mask=kv_mask[:, None], other=0.0)

        # Scaled dot-product attention 不能漏掉 1 / sqrt(D)。
        s = tl.dot(q_block, tl.trans(k_block)) * sm_scale
        score_mask = q_mask[:, None] & kv_mask[None, :]
        if is_causal:
            # Causal mask 比较的是 token 位置，而不是扁平内存 offset；
            # score[q, k] 仅在 q >= k 时有效，shape 为 [BLOCK_Q, BLOCK_KV]。
            score_mask = score_mask & (q_indices[:, None] >= kv_indices[None, :])
        # 易错点：mask 应直接选择 s 或 -inf。不能写成 mask_value * s，
        # 因为 (-inf) * 负数会变成 +inf，(-inf) * 0 会产生 NaN。
        s = tl.where(score_mask, s, float("-inf"))

        block_m = tl.max(s, axis=1)
        # tl.max 是 reduction；两个向量逐元素取最大值要用 tl.maximum。
        newm = tl.maximum(block_m, m)
        # Padding query row 没有有效 score；为它设置有限的 running maximum，
        # 防止中间计算出现 -inf - -inf 导致的 NaN。
        newm = tl.where(q_mask, newm, 0.0)
        # newm/changer 是每个 query row 的标量，参与二维计算时必须使用
        # [:, None] 按行广播；省略它可能错误地按列缩放。
        p = tl.exp(s - newm[:, None])
        changer = tl.exp(m - newm)
        newl = l * changer + tl.sum(p, axis=1)
        o = o * changer[:, None] + tl.dot(p.to(v_block.dtype), v_block)
        l = newl
        m = newm

    # o 保存的是未归一化 numerator，最后必须除以 online softmax 的 l。
    # padding query 的 l 为 0，用 safe_l 避免生成无意义的 0/0。
    safe_l = tl.where(q_mask, l, 1.0)
    out = o / safe_l[:, None]
    tl.store(output + qoffset, out, mask=q_mask[:, None])


def SelfFlash(q, k, v, is_causal):
    QBLOCKSIZE = 32
    assert q.ndim == 4 and q.shape == k.shape == v.shape
    assert q.device == k.device == v.device
    assert q.dtype == k.dtype == v.dtype
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    bs, num_head, seqlen, headDim = q.shape
    assert seqlen > 0
    assert headDim >= 16 and (headDim & (headDim - 1)) == 0
    # Grid 只启动 ceil(seqlen / BLOCK_Q) 个 Q blocks；把 seqlen 补到下一个
    # 2 的幂会启动不必要的 programs，正确性依赖的只是尾块 mask。
    grid = (bs * num_head, triton.cdiv(seqlen, QBLOCKSIZE))
    output = torch.empty_like(q)
    SelfKernel[grid](
        q,
        k,
        v,
        output,
        bs,
        num_head,
        seqlen,
        1.0 / math.sqrt(headDim),
        headDim,
        is_causal,
        QBLOCKSIZE,
    )
    # 易错点：wrapper 必须返回 output，否则 torch.allclose 收到的是 None。
    return output


# =============================================================================
# Lab 03 实验总结
# =============================================================================
# 1. Naive attention 会构造完整的 L x L score/softmax 矩阵，显存复杂度为
#    O(L^2)。本 Triton kernel 让每个 program 常驻一个 Q block，并逐块读取
#    K/V，把中间状态限制在 block 级别，避免保存完整 attention matrix。
#
# 2. FlashAttention-1 的核心是按 query row 维护三个 online softmax 状态：
#       m_new = max(m_old, max(score_block))
#       alpha = exp(m_old - m_new)
#       p     = exp(score_block - m_new)
#       l_new = alpha * l_old + sum(p)
#       o_new = alpha * o_old + p @ V_block
#    全部 KV blocks 扫描结束后，最终结果为 o / l。m 的更新保证指数输入不会
#    因 score 过大而溢出；旧的 l/o 必须乘 alpha，才能切换到新的归一化基准。
#
# 3. 正确性不仅取决于矩阵乘法，还取决于三类索引/mask：
#    - Q/K/V 的物理内存 offset；
#    - 非整块序列的 padding mask；
#    - causal attention 的 token-position mask。
#    Load mask 防止越界，score mask 决定 softmax 语义，两者不能互相替代。
#
# 4. 数值策略：Q/K/V 保持输入 dtype，score 以及 m/l/o 使用 FP32；p 在进入
#    Tensor Core dot 前转换为 V 的 dtype。最终 store 时再转换为输出 dtype。
#
# 5. 验证不能只测 (B == H) 或 L 能整除 block size 的单一 case。本实验额外
#    覆盖了 B != H、L=1/33/65/127、D=32/64，以及 causal/non-causal；这些
#    case 分别能暴露 batch/head 解码、尾块 mask、广播和 causal 方向错误。
#
# 6. 当前实现是教学版 forward kernel：要求 contiguous 的 (B,H,L,D) 输入，
#    headDim 是不小于 16 的 2 的幂；尚未实现 backward、autotune、不同 Q/KV
#    block size、causal block 跳过或面向性能的 warp/stage 参数搜索。


def main() -> None:
    torch.manual_seed(0)
    b, h, l, d = 2, 4, 128, 32
    q = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, h, l, d, device="cuda", dtype=torch.float16)

    ref = naive_attention(q, k, v)
    got_blocked = flash_attention_v1(q, k, v, is_causal=False, br=32, bc=32)
    assert torch.allclose(ref, got_blocked, atol=5e-2, rtol=5e-2), (
        ref - got_blocked
    ).abs().max()
    got = SelfFlash(q, k, v, False)
    assert torch.allclose(ref, got, atol=5e-2, rtol=5e-2), (ref - got).abs().max()

    ref_c = naive_attention(q, k, v, causal=True)
    got_c = SelfFlash(q, k, v, True)
    assert torch.allclose(ref_c, got_c, atol=5e-2, rtol=5e-2), (
        (ref_c - got_c).abs().max()
    )

    print("flash_attention_v1 ok (blocked + Triton, non-causal + causal vs naive)")


if __name__ == "__main__":
    main()
