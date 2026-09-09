# FlashAttention-2 in Triton：从算法到 Forward Kernel

> 课程：SYS-601 GPU Architecture and Operator Optimization  
> 主参考：[How I Wrote FlashAttention-2 from Scratch in Custom Triton Kernels](https://medium.com/@katherineolowookere/how-i-wrote-flashattention-2-from-scratch-in-custom-triton-kernels-885cac1da357)，Katherine Oluwadarasimi Olowookere，2026-02-03  
> 作者代码：[MyDarapy/triton — `flash_attention.py`](https://github.com/MyDarapy/triton/blob/main/flash_attention%20/flash_attention.py)  
> 官方对照：[Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)  
> 本地前置笔记：[Triton GPU 编程基础](triton-programming-fundamentals.md)  
> 官方源码详解：[Triton `06-fused-attention.py` Source Code Study](source-code-study-06-fused-attention.md)  
> 公式使用 `$...$` / `$$...$$`，兼容 VS Code Markdown 数学渲染。

## 1. 阅读定位

这篇文章实现的是 FlashAttention-2 风格的 forward kernel，重点是：

- 将 Q、K、V 按 tile 切分；
- 每个 Triton program 固定一个 Q tile；
- 流式读取 K/V tiles；
- 用 online softmax 合并局部结果；
- 不在 HBM 中物化完整 attention score/probability；
- 支持 causal 和 non-causal；
- 用 autotune 选择 block size、warps 和 pipeline stages；
- 与 PyTorch reference 对比正确性和性能。

它没有完整覆盖：

- backward；
- dropout；
- attention bias；
- GQA/MQA；
- variable-length sequences；
- paged KV cache；
- FP8 backward；
- 生产环境中的全部 layout、shape 和架构分支。

所以它最适合用作：

> 从 PyTorch blocked prototype 过渡到真实 Triton Attention kernel 的实现教材。

---

## 2. 一页总览

### 2.1 数学目标

对每个 batch 和 attention head：

$$
S=sQK^\top,
$$

$$
P=\operatorname{softmax}(S),
$$

$$
O=PV,
$$

通常：

$$
s=\frac{1}{\sqrt d}.
$$

输入输出逻辑 shape：

```text
Q, K, V, O: [B, H, N, D]
```

### 2.2 kernel 调度

```text
grid axis 0 -> Q block
grid axis 1 -> flattened (batch, head)

一个 program:
    固定 Q[BM, D]
    保持 m[BM], l[BM], acc[BM, D]
    for each K/V block:
        S_tile = Q_tile @ K_tile.T       [BM, BN]
        online softmax update
        acc += P_tile @ V_tile           [BM, D]
    O_tile = acc / l
    store O_tile
```

### 2.3 性能本质

FlashAttention 没有把精确 Attention 的算术复杂度从 $O(N^2D)$ 降下来。它改变的是计算调度和 IO：

```text
标准 Attention:
    生成 N×N S -> 写 HBM
    读 S -> softmax -> 写 N×N P
    读 P 和 V -> 输出

FlashAttention:
    只生成 BM×BN 的局部 S/P tile
    用完即丢弃
    只写最终 O 和少量每行统计量
```

所谓 memory complexity 从 $O(N^2)$ 降到 $O(N)$，准确地说是：避免了 $N^2$ 级别的显式中间 attention tensor；Q/K/V/O 本身仍占 $O(ND)$，片上临时状态由 tile size 决定。

---

## 3. 为什么标准 Attention 受 HBM 限制

标准分步实现通常对应多个 kernels：

```text
QKᵀ
  -> scale/mask
  -> row max
  -> exp
  -> row sum
  -> normalize
  -> PV
```

如果中间结果没有融合，score 和 probability 会在 HBM 中多次读写。

单个 head 的 score 元素数是：

$$
N^2.
$$

当 $N=16384$：

$$
N^2=268{,}435{,}456.
$$

若用 FP16，仅一份矩阵约为：

$$
268{,}435{,}456\times2\ \text{bytes}
\approx512\ \text{MiB}.
$$

这还只是一个 batch、一个 head、一份中间量。真正实现还可能涉及 FP32 softmax、mask、反向状态等。

关键矛盾：

- Tensor Core 计算吞吐增长很快；
- HBM 带宽和容量增长跟不上中间数据规模；
- 若算子反复搬运 $N^2$ 中间量，计算单元会等待数据；
- 仅减少 FLOPs 不一定解决瓶颈，必须优化 memory traffic。

FlashAttention 因此是 IO-aware exact attention，而不是近似 Attention。

---

## 4. Blocked Attention 的算法结构

设：

```text
BM = BLOCK_SIZE_Q
BN = BLOCK_SIZE_KV
```

把 Q 按行切成 Q tiles，把 K/V 按 token 维切成 K/V tiles。

```python
for each batch b:
    for each head h:
        parallel for each Q block i:
            q = Q[b, h, i:i+BM, :]
            m = -inf
            l = 0
            acc = 0

            for each K/V block j:
                k = K[b, h, j:j+BN, :]
                v = V[b, h, j:j+BN, :]
                scores = scale * q @ k.T
                update m, l, acc with online softmax

            O[b, h, i:i+BM, :] = acc / l
```

单次 inner iteration 的 tile shapes：

| 值 | Shape |
|---|---|
| `Q_block` | `[BM, D]` |
| `K_block` | `[D, BN]`，已按 dot 需要转置寻址 |
| `QK_block` | `[BM, BN]` |
| `P_block` | `[BM, BN]` |
| `V_block` | `[BN, D]` |
| `O_block/acc` | `[BM, D]` |

两次局部矩阵乘：

$$
[B_M,D]\times[D,B_N]\to[B_M,B_N],
$$

$$
[B_M,B_N]\times[B_N,D]\to[B_M,D].
$$

注意：文章正文有一处把 K 和 V pointers 都概括成 `[HEAD_DIM, BLOCK_SIZE_KV]`。实际只有 K 为 `[D,BN]`；V 必须为 `[BN,D]`，否则 `P_block @ V_block` 的 shape 不成立。

---

## 5. 最难部分：Online Softmax

### 5.1 为什么普通 softmax 不能直接套用

普通稳定 softmax 假设整行 score 同时可见：

$$
m_i=\max_jS_{ij},
$$

$$
P_{ij}=\frac{e^{S_{ij}-m_i}}
{\sum_ke^{S_{ik}-m_i}}.
$$

FlashAttention 一次只看到 `[BM,BN]` score tile，无法提前知道整行全局最大值和分母。因此每个 Q row 需要维护可合并状态。

### 5.2 三个状态

对每个正在处理的 query row：

| 状态 | 含义 | Shape |
|---|---|---|
| `m_i` | 已处理 score 的运行最大值 | `[BM]` |
| `l_i` | 以当前最大值为基准的指数和 | `[BM]` |
| `O_block` / `acc` | 同一尺度下的未归一化加权 V 和 | `[BM,D]` |

处理旧 tiles 后保持不变量：

$$
m_i=\max_{j\in\text{seen}}S_{ij},
$$

$$
l_i=\sum_{j\in\text{seen}}e^{S_{ij}-m_i},
$$

$$
acc_i=\sum_{j\in\text{seen}}e^{S_{ij}-m_i}V_j.
$$

若全部 K/V 已处理，则：

$$
O_i=\frac{acc_i}{l_i}.
$$

### 5.3 合并一个新 tile

新 score tile 的行最大值：

$$
m_{tile}=\max_{j\in\text{new tile}}S_{ij}.
$$

合并最大值：

$$
m_{new}=\max(m_{old},m_{tile}).
$$

旧状态原来以 $m_{old}$ 为指数基准，必须重标定到 $m_{new}$：

$$
\alpha=e^{m_{old}-m_{new}}.
$$

新 tile 的局部指数：

$$
P_{tile}=e^{S_{tile}-m_{new}}.
$$

更新分母：

$$
l_{new}=\alpha l_{old}
+\operatorname{rowsum}(P_{tile}).
$$

更新输出累加器：

$$
acc_{new}=\alpha acc_{old}+P_{tile}V_{tile}.
$$

最后：

$$
m_{old}\leftarrow m_{new},\qquad
l_{old}\leftarrow l_{new},\qquad
acc_{old}\leftarrow acc_{new}.
$$

### 5.4 为什么必须缩放旧 accumulator

这是最常见的实现错误。

旧 `acc` 中的权重是：

$$
e^{S_{ij}-m_{old}}.
$$

新 tile 到来后，所有权重必须改成：

$$
e^{S_{ij}-m_{new}}
=e^{S_{ij}-m_{old}}e^{m_{old}-m_{new}}.
$$

因此旧 `l` 和旧 `acc` 都必须乘相同的 `alpha`。只缩放分母、不缩放 `acc`，输出会混用两个指数坐标。

### 5.5 初值

数学上最直观：

```text
m = -inf
l = 0
acc = 0
```

文章/官方源码的某些版本使用 `l = 1`。第一次更新时：

$$
\alpha=e^{-\infty-m_{new}}=0,
$$

所以初始 1 会被乘成 0，不影响结果。学习和自己实现时，`l=0` 更容易直接对应数学不变量；阅读具体源码时则按其实际初始化理解。

---

## 6. Triton Grid 与工作划分

host wrapper 中的 grid：

```python
grid = lambda meta: (
    triton.cdiv(Lq, meta["BLOCK_SIZE_Q"]),
    B * H,
    1,
)
```

kernel 内：

```python
block_index_q = tl.program_id(axis=0)
index_batch_head = tl.program_id(axis=1)
```

解码 batch/head：

```python
index_batch = index_batch_head // NUM_HEADS
index_head = index_batch_head % NUM_HEADS
```

一个 program 的所有权：

```text
一个 batch
× 一个 head
× 一个 Q token block
× 对应的一个 O token block
```

严谨纠正：不是“一个 head 只有一个 program”。同一 `(batch,head)` 上有：

$$
\left\lceil\frac{L_q}{B_M}\right\rceil
$$

个 Q-block programs。这正是 FlashAttention-2 相比只在 batch/head 上并行的重要改进之一：即使 batch/head 数不大，也能沿 sequence/Q blocks 提供更多并行度。

programs 不需要互相同步，因为不同 Q blocks 写不重叠的 O rows。

---

## 7. Stride 与 Pointer Arithmetic

Q 的逻辑索引：

```text
Q[b, h, n, d]
```

对应地址：

$$
Q_{ptr}
+b\cdot stride_{qb}
+h\cdot stride_{qh}
+n\cdot stride_{qn}
+d\cdot stride_{qd}.
$$

### 7.1 先定位 batch/head

```python
q_head_offset = (
    index_batch * q_stride_b
    + index_head * q_stride_h
)
```

K/V/O 应分别使用各自 stride，不能假定它们与 Q 相同。

### 7.2 Q block pointer

```python
offs_q = block_index_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
offs_d = tl.arange(0, HEAD_DIM)

q_ptrs = (
    q_ptr
    + q_head_offset
    + offs_q[:, None] * q_stride_n
    + offs_d[None, :] * q_stride_d
)
```

得到 `[BM,D]` pointer block。

### 7.3 K block pointer

为了直接执行 `Q @ K.T`，K pointer 按 `[D,BN]` 构造：

```python
kv_positions = start_kv + offs_kv

k_ptrs = (
    k_ptr
    + k_head_offset
    + offs_d[:, None] * k_stride_d
    + kv_positions[None, :] * k_stride_n
)
```

### 7.4 V block pointer

V 保持 `[BN,D]`：

```python
v_ptrs = (
    v_ptr
    + v_head_offset
    + kv_positions[:, None] * v_stride_n
    + offs_d[None, :] * v_stride_d
)
```

### 7.5 Shape 自检

```text
q       [BM,D]
k       [D,BN]
scores  [BM,BN]
p       [BM,BN]
v       [BN,D]
acc     [BM,D]
```

每次写 pointer 前先写这个 shape chain，能避免大多数转置错误。

---

## 8. Outer Kernel 的职责

文章把 forward 拆成：

```text
fwd_flash_attn_kernel
    └─ _attn_fwd_inner
```

outer kernel 负责：

1. 读取 program ids；
2. 解码 batch/head；
3. 构造 Q/O tile pointers；
4. 初始化 `m/l/acc`；
5. load Q tile；
6. 根据 causal 选择 inner 调用范围；
7. 最终归一化；
8. store O；
9. 可选保存 log-sum-exp 供 backward。

inner function 负责：

1. 确定 K/V 扫描区间；
2. 循环 load K/V tiles；
3. 计算 score tile；
4. 执行 causal mask；
5. 更新 online softmax；
6. 累积 `P @ V`。

这种拆分有利于把 program mapping 与数学 inner loop 分开阅读。编译后它们不一定是两个独立 GPU kernel；被 `@triton.jit` helper 调用的 inner 通常会内联进主 kernel。文章所说“split into two kernels”更准确地应理解为“拆成主 kernel 与 JIT helper”。

---

## 9. Causal Attention 的两段式处理

causal 条件：

$$
P_{ij}=0\quad\text{if }j>i.
$$

对于当前 Q block，可把 K/V 区域分成：

```text
左侧 off-band:
    整个 tile 都合法，不需要逐元素 mask

对角 on-band:
    tile 穿过 causal diagonal，需要 mask

右侧 future region:
    全部非法，不访问、不计算
```

### 9.1 wrapper 的 stage

```python
stage = 3 if causal else 1
```

这不是 `num_stages`。二者含义完全不同：

| 参数 | 含义 |
|---|---|
| `STAGE` | 算法控制：扫描全序列，还是 causal 的 off-band/on-band |
| `num_stages` | 编译器软件流水深度 |

### 9.2 inner 的 stage 语义

```text
inner STAGE=1:
    lo=0
    hi=block_index_q * BLOCK_Q
    严格对角线左侧，无 mask

inner STAGE=2:
    lo=block_index_q * BLOCK_Q
    hi=(block_index_q+1) * BLOCK_Q
    对角带，需要 mask

inner STAGE=3:
    lo=0
    hi=SEQ_LEN
    non-causal 全序列，无 mask
```

outer 的控制看起来绕：

```python
if STAGE & 1:
    inner_stage = 4 - STAGE

if STAGE & 2:
    inner_stage = 2
```

展开：

| 模式 | outer STAGE | 第一次 inner | 第二次 inner |
|---|---:|---:|---:|
| non-causal | 1 | 3：全序列 | 无 |
| causal | 3 | 1：左侧 | 2：对角带 |

重要纠正：文章代码注释将第二次 causal 调用描述为“对角线右侧”。实际它处理的是对角 block/on-band；真正位于对角线右侧的 future tiles 不应计算。

### 9.3 对角 mask

```python
causal_mask = (
    offs_q[:, None]
    >= start_kv + offs_kv[None, :]
)
scores = scores * scale + tl.where(causal_mask, 0.0, -1.0e6)
```

`-1e6` 在指数后数值上成为 0，近似 `-inf`。只有穿过对角线的 tile 需要这一步，从而避免在大部分合法区域执行逐元素 mask。

---

## 10. Inner Loop 逐步映射

一轮 K/V tile 的核心逻辑：

```python
k = tl.load(k_ptrs, mask=k_mask, other=0.0)    # [D,BN]
v = tl.load(v_ptrs, mask=v_mask, other=0.0)    # [BN,D]

scores = tl.dot(q, k)                          # [BM,BN]
scores = scores * scale
scores = apply_causal_mask_if_needed(scores)

m_new = tl.maximum(m, tl.max(scores, axis=1))  # [BM]
p = tl.exp(scores - m_new[:, None])            # [BM,BN]
alpha = tl.exp(m - m_new)                      # [BM]

l = l * alpha + tl.sum(p, axis=1)
acc = acc * alpha[:, None]
acc = tl.dot(p.to(input_dtype), v, acc)
m = m_new
```

### 10.1 为什么 P 转回低精度

`p` 和 online softmax 状态通常以 FP32 计算，以保证稳定性。但第二次 `tl.dot(p,v)` 为获得 Tensor Core 吞吐，常把 P tile 转为 FP16/BF16，同时保留 FP32 accumulator。

精度策略：

```text
score accumulation: implementation/hardware dependent
max/l/alpha:        FP32
p before dot:       FP16/BF16
output accumulator: FP32
stored O:           input/output dtype
```

### 10.2 为什么状态更新顺序重要

正确顺序应确保旧 `m` 在计算 `alpha` 时仍可用：

```text
compute m_new
compute alpha from old m and m_new
rescale old l/acc
add new tile contribution
assign m = m_new
```

若过早执行 `m = m_new`，则 `alpha` 会恒等于 1。

### 10.3 为什么 Q 保持不变、K/V 流动

每个 program 独占一个输出 Q block。固定 Q：

- Q tile 只需 load 一次；
- `m/l/acc` 始终对应固定 Q rows；
- 扫描 K/V 即可完成该输出 tile；
- 不需要跨 program 合并 partial O。

若把 K block 固定而把不同 Q blocks 分给 programs，最终可能需要对输出进行额外归并，复杂度更高。

---

## 11. Final Normalization 与 LSE

所有 K/V tiles 处理结束后：

```python
O_block = acc / l[:, None]
```

若计划实现 backward，forward 还可保存每行 log-sum-exp：

$$
LSE_i=m_i+\log l_i.
$$

因为：

$$
e^{S_{ij}-LSE_i}
=\frac{e^{S_{ij}}}{\sum_ke^{S_{ik}}}
=P_{ij}.
$$

backward 可只保存 LSE，而不保存 $N^2$ 的 P，再通过重算 score tiles 恢复 P。

文章的 forward-only 代码分配并写入 `M`，但没有 backward 消费它。若只做 inference forward，M 是可删除的额外输出；若要扩展训练 backward，则它是关键状态。

---

## 12. `exp` 与 `exp2` 两条实现路线

文章实现使用自然指数：

```python
p = tl.exp(scores - m_new)
alpha = tl.exp(m_old - m_new)
LSE = m + tl.log(l)
```

当前 Triton 官方 fused attention 常使用 base-2 指数：

```python
qk_scale = sm_scale / ln(2)
p = tl.exp2(qk * qk_scale - m_new)
alpha = tl.exp2(m_old - m_new)
LSE = m + tl.log2(l)
```

依据：

$$
e^x=2^{x/\ln2}.
$$

两者都正确，但不能混用尺度：

- 若 score 使用 `exp`，M/LSE 使用自然对数坐标；
- 若 score 使用 `exp2`，score scale、M/LSE 和 backward 重算都必须使用 base-2 坐标；
- 最危险的 bug 是 forward 保存一种 LSE，backward 用另一种指数解释。

先实现自然指数版本保证正确，再进行 `exp2` 优化更容易调试。

---

## 13. 什么使它更接近 FlashAttention-2

FlashAttention-2 不只是“FlashAttention 用 Triton 重写”。其核心改进包括：

1. 增加 sequence 方向并行：不同 Q blocks 可独立成为 programs；
2. 改善不同 warps 的工作划分，减少 shared-memory 通信；
3. 减少非矩阵乘 FLOPs；
4. 提高 Tensor Core 工作占比和 occupancy；
5. forward/backward 使用更适合各自输出所有权的分块方向。

文章实现最清晰体现的是第一点：grid 同时沿 Q block 和 batch-head 展开。

但文章仅实现 forward，且没有完整展示官方 FA2 backward 的 dQ 与 dK/dV 工作分区。因此称其为“FA2-style forward”比称为“完整训练级 FA2”更准确。

---

## 14. Autotune

文章源码搜索：

```text
BLOCK_SIZE_Q  ∈ {128, 256, 512}
BLOCK_SIZE_KV ∈ {32, 64, 128}
num_stages    ∈ {1, 2, 3, 4}
num_warps     ∈ {2, 4, 8}
```

共 $3\times3\times4\times3=108$ 个候选，key 为 sequence length 和 head dimension。

### 14.1 参数权衡

| 参数增大 | 潜在收益 | 潜在代价 |
|---|---|---|
| `BLOCK_Q` | K/V tile 被更多 Q rows 复用 | `acc[BM,D]`、score `[BM,BN]` 更大，寄存器压力上升 |
| `BLOCK_KV` | inner loop 次数减少，dot 更大 | score/P tile 增大，mask tail 浪费增加 |
| `num_warps` | program 内并行度提高 | 单 program 资源增大，驻留 program 数下降 |
| `num_stages` | load/compute overlap 改善 | staging memory 与寄存器使用增加 |

### 14.2 为什么要剪枝

108 个配置全部编译/测量会导致明显首次运行开销，而且部分组合可能：

- 片上资源过大；
- 对小 N 严重浪费；
- 不适合特定 HEAD_DIM；
- 引发 register spill；
- 不满足 causal block 对齐关系；
- 编译失败或性能显著不合理。

建议先限制：

```text
BLOCK_Q <= SEQ_LEN（或妥善支持 Q tail）
BLOCK_KV <= 合理的 HEAD_DIM/架构约束
causal 时保证对角切分可整齐覆盖
排除估算会造成极端 register pressure 的大 tile
```

### 14.3 autotune key

除 `SEQ_LEN`、`HEAD_DIM` 外，应考虑：

- causal/STAGE；
- dtype；
- Q/K sequence lengths（cross-attention 时可能不同）；
- warp specialization 模式；
- 目标架构。

如果 causal 和 non-causal 共享相同 autotune key，可能复用由另一种工作量选出的配置。JIT specialization 与 autotune cache 的行为依 Triton 版本而异，应该显式验证而非假设。

---

## 15. 正确性验证

### 15.1 PyTorch reference

```python
scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
scores *= 1.0 / math.sqrt(head_dim)

if causal:
    mask = torch.triu(
        torch.ones(n, n, device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(mask, -float("inf"))

p = torch.softmax(scores, dim=-1)
expected = torch.matmul(p, v.float()).to(q.dtype)
```

### 15.2 必测维度

```text
B: 1, 2, 4
H: 1, 8, 32
N: 1, 31, 127, 128, 129, 511, 1024, 4096
D: 32, 64, 128
causal: False, True
dtype: FP16, BF16（若实现支持）
```

先测规则的 N，再测非整 tile N。作者示例的 Q/O 读写没有完整 tail mask，所以非整 `BLOCK_Q` 的 N 是必须新增的测试。

### 15.3 数值压力测试

- Q/K 全零：non-causal 输出应为 V 的行均值；
- causal Q/K 全零：第 i 行输出应为 `V[:i+1]` 的均值；
- Q/K 放大 10 倍或 100 倍：检查 NaN/Inf；
- 第一 causal row：只能关注 token 0；
- 最后 causal row：可关注全部 tokens；
- V 为单位模式：更容易观察 probability 是否正确；
- 固定随机种子，统计 max/mean absolute error。

### 15.4 容差

FP16 初始可用：

```python
torch.testing.assert_close(
    actual,
    expected,
    atol=5e-2,
    rtol=5e-2,
)
```

但容差不是“让测试通过”的旋钮。应记录误差分布，并检查误差是否随 N、D 或输入尺度异常增长。

---

## 16. Benchmark 与 TFLOPS

non-causal forward 有两次主矩阵乘：

```text
QKᵀ: 2 B H N² D FLOPs
PV:  2 B H N² D FLOPs
```

总计：

$$
F_{fwd}\approx4BHN^2D.
$$

causal 只处理下三角，近似：

$$
F_{causal}\approx2BHN^2D.
$$

若测得延迟为 $t$ 秒：

$$
\text{TFLOPS}=\frac{F}{t}\times10^{-12}.
$$

### 16.1 公平 benchmark

- 相同 B/H/N/D；
- 相同 dtype；
- 相同 causal 语义；
- 相同输入 layout；
- 只测 forward 时不要把某个 baseline 的 backward 包含进去；
- warm up JIT/autotune；
- 排除首次 allocator/cache 影响；
- 用 `triton.testing.do_bench` 或 CUDA events；
- 报告 GPU、频率模式、PyTorch/Triton/CUDA 版本；
- 验证 PyTorch SDPA 实际选中了哪个 backend。

### 16.2 怎样理解文章结果

文章报告 A100 SXM 上长序列时自定义 Triton forward 接近或略超 PyTorch SDPA，并达到约 180–190 TFLOPS。这个结果只说明作者给定 shape、版本、layout 和硬件下的表现，不能直接外推到 H100、Blackwell、AMD 或另一 Triton 版本。

长 N 吞吐提高的常见原因：

- 计算规模足以摊薄 launch overhead；
- tile dot 占比提高；
- Tensor Core 利用更充分；
- 不物化 score/P 后，HBM 中间 traffic 不随标准路径那样恶化；
- kernel 逐渐从 launch/IO 限制转向 compute-bound。

但“FlashAttention 一定 compute-bound”也不是绝对结论。小 batch/head、短序列、小 head dimension、差 tile、spill 或非理想 layout 都可能形成其他瓶颈。

---

## 17. 作者实现的工程审计

文章很适合建立思路，但其示例代码应视为教学/实验实现。直接复用前应修正以下问题。

### 17.1 Q/O tail mask 缺失

源码中 Q load 和 O store 没有完整使用：

```text
offs_q < SEQ_LEN
```

当 `SEQ_LEN % BLOCK_Q != 0` 或 autotune 选择 `BLOCK_Q > SEQ_LEN` 时，最后一个 program 可能越界。

修改原则：

```python
q_mask = (offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < HEAD_DIM)
q = tl.load(q_ptrs, mask=q_mask, other=0.0)
tl.store(o_ptrs, out, mask=q_mask)
```

### 17.2 shape 被覆盖而未校验

作者 wrapper 连续解包 Q/K/V shape，会覆盖 B/H/D 变量，却没有 assert 三者一致。应该显式检查：

```text
Q.ndim == K.ndim == V.ndim == 4
Q.shape == K.shape == V.shape（self-attention 简化版）
Q/K head dim 一致
K/V sequence length 一致
device/dtype 一致
支持的 HEAD_DIM
stride/layout 约束
```

### 17.3 `torch.autograd.Function` 使用不完整

作者类定义了 `flash_attention` 静态方法，而不是标准的 `forward(ctx, ...)`，也没有 `backward`。因此它并不是完整可用的 autograd Function。

forward-only 实验可直接写普通函数；若要接入 autograd，应实现：

```python
class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        ...

    @staticmethod
    def backward(ctx, do):
        ...

attention = FlashAttention.apply
```

### 17.4 runtime 值不应滥标 `tl.constexpr`

作者 inner helper 把部分 stride、batch/head offset 等标为 `tl.constexpr`，但其中一些值来自 `program_id` 或 runtime tensor stride。应只把真正编译期可知的 shape/meta 参数标为 constexpr，例如：

```text
HEAD_DIM
BLOCK_Q
BLOCK_KV
STAGE
```

动态 pointer offset 和一般 runtime stride 不应假装成编译期常量。具体可接受签名随 Triton 版本演进，必须以当前编译器测试为准。

### 17.5 `M` 在 forward-only 中未被使用

若不实现 backward，可不分配/写 M，避免额外 HBM traffic。若计划 backward，则应明确它保存的是 LSE，并保证 log base 与反向的 exp 一致。

### 17.6 causal 注释错误

第二阶段是 diagonal/on-band，不是“对角线右侧”。右侧 future blocks 应完全跳过。

### 17.7 K/V shape 描述错误

K 为 `[D,BN]`，V 为 `[BN,D]`。这应由两次 dot 的 shape 直接验证。

### 17.8 autotune 搜索空间过宽

包含 `BLOCK_Q=512` 的配置可能对部分 head dimension/架构产生极大 score/accumulator 状态。应做 early pruning，并在 benchmark 后检查 register spill，而不是只看 autotuner 是否能运行。

### 17.9 `BATCH_SIZE` 参数未使用

若 kernel 不需要它，应删掉；若用于边界或 grid 验证，则实际使用。无用 runtime 参数增加理解和接口成本。

### 17.10 “on-chip SRAM” 是抽象描述

Triton tensor value 最终可能映射到 registers、shared memory 或编译器安排的 staging。不要从一条 `tl.load` 直接推断具体物理存储，应检查 TTGIR/PTX 和 profiler。

---

## 18. 与本地实现的对应关系

### 18.1 Naive reference

本地：[`../labs/lab03-flashattention/naive_attention.py`](../labs/lab03-flashattention/naive_attention.py)

它显式计算：

```text
scores [B,H,N,N]
attn   [B,H,N,N]
out    [B,H,N,D]
```

适合 correctness reference，不适合长序列 memory benchmark。

### 18.2 Blocked PyTorch prototype

本地：[`../labs/lab03-flashattention/flash_attention_v1.py`](../labs/lab03-flashattention/flash_attention_v1.py)

| PyTorch prototype | Triton kernel |
|---|---|
| 外层 Q block Python loop | grid axis 0 并行 programs |
| `qi = q[..., i:i_end, :]` | Q pointer block + `tl.load` |
| 内层 K/V Python loop | `_attn_fwd_inner` 的 `tl.range` |
| `s = qi @ kj.T` | `tl.dot(Q_block, K_block)` |
| `m_new` | `tl.maximum(m_i, rowmax(scores))` |
| `exp(m_old-m_new)` | `alpha` |
| `o_i = alpha*o_i + p@v` | FP32 `O_block` accumulator |
| `o_i/l_i` | final normalization |

### 18.3 本地 v2 的边界

[`../labs/lab03-flashattention/flash_attention_v2.py`](../labs/lab03-flashattention/flash_attention_v2.py) 只把 `(B,H)` 合并为更大的 batch，以观察批量 GEMM 调度，并不等价于完整 Triton FA2。真正的实现还包括 program-level Q-block parallelism、融合、online state、causal stage、autotune 和自定义 backward。

---

## 19. 从零实现的推荐顺序

不要一次写完整 kernel。按以下里程碑推进。

### Milestone 1：PyTorch blocked forward

- [ ] non-causal；
- [ ] `m/l/acc` 全部 FP32；
- [ ] 与 naive reference 对齐；
- [ ] 打印每轮 online state；
- [ ] 用极端 score 验证稳定性。

### Milestone 2：最小 Triton non-causal forward

- [ ] 固定 B=H=1；
- [ ] 固定 N 为 tile 整数倍；
- [ ] 固定 D=64；
- [ ] 一个 program 一个 Q tile；
- [ ] K/V streaming；
- [ ] 不分配 N×N 中间 tensor。

### Milestone 3：通用 batch/head

- [ ] grid axis 1 展平 B×H；
- [ ] 分别传 Q/K/V/O strides；
- [ ] 验证 program ownership；
- [ ] 增加 B/H 测试组合。

### Milestone 4：Boundary safety

- [ ] Q/O tail mask；
- [ ] K/V tail mask；
- [ ] 非整 N；
- [ ] 支持 D 集合；
- [ ] 明确 contiguous/stride 契约。

### Milestone 5：Causal

- [ ] off-band 无 mask；
- [ ] diagonal masked；
- [ ] future region 跳过；
- [ ] 第一行、最后一行专门测试。

### Milestone 6：性能优化

- [ ] 自然 exp 版本作为 correctness baseline；
- [ ] 尝试 exp2；
- [ ] autotune BM/BN/warps/stages；
- [ ] config pruning；
- [ ] profiler 检查 Tensor Core、spill、occupancy、HBM/L2；
- [ ] 与 PyTorch SDPA 公平比较。

### Milestone 7：Backward

- [ ] forward 保存 LSE；
- [ ] 预计算 $\Delta_i=\langle O_i,dO_i\rangle$；
- [ ] 重算 P tiles；
- [ ] dQ 与 dK/dV 分开选择扫描方向；
- [ ] 与 PyTorch autograd 对比 gradients。

对应练习：[Lab 02 P6 Mini Attention Forward](../labs/lab02-triton-fundamentals/EXERCISES.md#p6mini-attention-forward)。

---

## 20. Profiler 检查表

### Memory

- [ ] 是否出现 $N^2$ global allocation？
- [ ] Q 是否每个 program 只显式 load 一次？
- [ ] K/V 是否按 tile 流式访问？
- [ ] O 是否只在最终归一化后 store？
- [ ] forward-only 是否仍无必要写 M？
- [ ] L2 hit rate 是否随 program ordering 改善？

### Compute

- [ ] 两次 `tl.dot` 是否使用目标 Tensor Core 指令？
- [ ] 非 matmul FLOPs 占比是否过高？
- [ ] causal mask 是否只在 diagonal tile 执行？
- [ ] Tensor Core utilization 是否随 N 增大？

### Resources

- [ ] registers per thread/program；
- [ ] shared memory/staging 使用；
- [ ] occupancy；
- [ ] local-memory spill；
- [ ] `BLOCK_Q × BLOCK_KV` score tile 是否过大；
- [ ] `BLOCK_Q × D` accumulator 是否成为资源瓶颈。

### Scheduling

- [ ] grid 是否提供足够 Q-block parallelism？
- [ ] 小 B/H 时是否仍能占满 SM？
- [ ] `num_warps` 是否过大/过小？
- [ ] `num_stages` 是否真正改善 load/compute overlap？

---

## 21. 常见错误速查

| 症状 | 常见原因 |
|---|---|
| 只有第一个 K/V tile 正确 | pointer 没在 inner loop 前进 |
| N 为 tile 整数倍时正确，其他 N 崩溃 | Q/O 或 K/V tail mask 缺失 |
| causal 第一行错误 | 对角 mask 方向反了 |
| causal 与 non-causal 输出相同 | STAGE 传递或 mask 分支未生效 |
| 输出随 K/V tile 顺序变化很大 | online softmax rescale 错误 |
| 概率和不为 1 | `l` 更新或最终归一化错误 |
| 后续 tile 最大值变大时误差爆炸 | 忘记 `acc *= alpha` |
| `tl.dot(P,V)` 编译失败 | V tile shape 写成 `[D,BN]` |
| 小 shape 越界 | autotune 选择的 BLOCK_Q 大于 N 且无 mask |
| 编译期报 constexpr 错误 | runtime stride/offset 被错误标成 `tl.constexpr` |
| 正确但很慢 | P 未转低精度、tile 太大、spill、warps/stages 不合适 |
| benchmark 首次异常慢 | JIT/autotune 时间被计入 |

---

## 22. 面试问答

### Q1：FlashAttention 为什么更快？

它通过 tiling、fusion 和 online softmax 避免在 HBM 中物化并多次读写 $N^2$ 的 score/probability，使更多时间花在 Tensor Core 计算而不是中间数据搬运上。

### Q2：FlashAttention 是近似算法吗？

不是。忽略浮点运算次序带来的数值差异，它计算与标准 scaled dot-product attention 相同的结果。

### Q3：它降低了计算复杂度吗？

没有。精确 dense Attention 的主 FLOPs 仍为 $O(N^2D)$。主要降低的是显式中间显存和 HBM IO。

### Q4：online softmax 保存哪三个量？

每个 query row 的运行最大值 `m`、运行指数和 `l`、未归一化输出 accumulator `acc`。

### Q5：为什么旧 accumulator 要乘 alpha？

新 tile 可能提高运行最大值，旧贡献与新贡献的指数基准不同。`alpha=exp(m_old-m_new)` 把旧贡献重标定到新最大值坐标。

### Q6：为什么一个 program 固定 Q tile？

这样它独占对应 O tile，可让 Q 和 online state 保持片上，仅流式扫描 K/V，不需要跨 program 合并输出。

### Q7：causal 为什么拆两段？

严格下三角的 tiles 全部合法，不需 mask；只有 diagonal tile 需要逐元素 mask；未来 tiles 完全跳过。这样减少分支、比较和无效 dot。

### Q8：M/LSE 有什么作用？

forward 保存每行 LSE 后，backward 可通过 `exp(score-LSE)` 重算概率，无需保存完整 P。

### Q9：为什么大 tile 不一定更快？

大 tile 提高数据复用和 dot 效率，但会扩大 score/P/acc 活跃状态，增加寄存器/shared-memory 压力、降低 occupancy，甚至 spill。

### Q10：这份文章代码为什么不能直接当生产实现？

它只有 forward，且存在 tail mask、shape validation、autograd 接口、constexpr 标注、autotune pruning 等工程缺口；生产实现还需覆盖更多 dtype、layout、mask 和训练功能。

---

## 23. 最终复习版

```text
目标:
    O = softmax(scale * QKᵀ) V

grid:
    pid0 -> Q block
    pid1 -> flattened batch-head

program state:
    Q_tile [BM,D]
    m      [BM]
    l      [BM]
    acc    [BM,D]

inner tile:
    K [D,BN]
    V [BN,D]
    S = Q @ K       [BM,BN]

online update:
    m_new = max(m_old, rowmax(S))
    alpha = exp(m_old - m_new)
    P = exp(S - m_new)
    l = alpha*l + rowsum(P)
    acc = alpha*acc + P@V
    m = m_new

final:
    O = acc / l
    optional LSE = m + log(l)

causal:
    left/off-band -> no mask
    diagonal      -> mask
    future/right  -> skip

performance:
    no N×N HBM intermediate
    Q stays, K/V stream
    FP32 online state/acc
    low-precision dot operands
    autotune BM/BN/warps/stages

engineering gaps to check:
    Q/O tail mask
    K/V tail mask
    shape/dtype/stride validation
    correct constexpr usage
    fair benchmark
    backward/autograd contract
```

## 参考资料

- [How I Wrote FlashAttention-2 from Scratch in Custom Triton Kernels](https://medium.com/@katherineolowookere/how-i-wrote-flashattention-2-from-scratch-in-custom-triton-kernels-885cac1da357)
- [作者公开实现：MyDarapy/triton](https://github.com/MyDarapy/triton/blob/main/flash_attention%20/flash_attention.py)
- [Triton 官方 Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
