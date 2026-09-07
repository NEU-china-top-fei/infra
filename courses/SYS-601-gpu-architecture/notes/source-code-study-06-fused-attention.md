# Source Code Study：Triton `06-fused-attention.py`

> 课程：SYS-601 GPU Architecture and Operator Optimization  
> 主线源码：[`triton/python/tutorials/06-fused-attention.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)  
> 官方渲染版：[Fused Attention — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)  
> 阅读基线：`triton-lang/triton` 的 `main` 分支，2026-09-07 查阅。`main` 会变化，复现实验时应额外记录实际 commit SHA。  
> 相关本地代码：[`../labs/lab03-flashattention/flash_attention_v1.py`](../labs/lab03-flashattention/flash_attention_v1.py)

## 1. 先给结论

这个文件实现的是 FlashAttention-2 风格的融合 Attention：

$$
O=\operatorname{softmax}(sQK^\top)V,
$$

其中 `s = sm_scale`，通常取 $1/\sqrt d$。它的关键并不是减少 Attention 的理论计算量——时间复杂度仍然是 $O(N^2d)$——而是：

1. 不把完整的 $N\times N$ score 矩阵和 probability 矩阵写回 HBM；
2. 每个 Triton program 固定一个 Q tile，在片上反复扫描 K/V tile；
3. 用 online softmax 在只看到一部分列的情况下稳定地更新 softmax；
4. 把 `QKᵀ → mask → softmax → PV` 融合进一个前向 kernel；
5. 反向不保存 $P$，而是保存每行 log-sum-exp，随后重算 $P$；
6. 通过 autotune、Tensor Descriptor、warp specialization 和硬件分支适配不同 GPU。

一句话概括代码结构：

```text
PyTorch autograd wrapper
  ├─ forward
  │   └─ _attn_fwd                         每个 program 负责一个 Q 行块
  │       └─ _attn_fwd_inner               扫描 K/V 列块并做 online softmax
  └─ backward
      ├─ _attn_bwd_preprocess              计算 delta_i = <O_i, dO_i>
      └─ _attn_bwd
          ├─ _attn_bwd_dkdv                固定 K/V 块，扫描 Q/dO
          └─ _attn_bwd_dq                  固定 Q/dO 块，扫描 K/V
```

阅读时最重要的三个状态是：

| 变量 | 每行含义 | 形状 |
|---|---|---|
| `m_i` | 已处理 score 的行最大值，代码使用 base-2 指数坐标 | `[BLOCK_M]` |
| `l_i` | 相对当前最大值平移后的指数和 | `[BLOCK_M]` |
| `acc` | 尚未除以 `l_i` 的加权 V 累加器 | `[BLOCK_M, HEAD_DIM]` |

只要能独立推导这三个变量的更新式，就抓住了整个前向 kernel。

---

## 2. 数学、张量与代码命名

### 2.1 输入输出

源码约定 Q、K、V 的逻辑形状均为：

```text
[Z, H, N_CTX, HEAD_DIM]
```

| 符号/源码名 | 含义 |
|---|---|
| `Z` | batch size，本文也写作 $B$ |
| `H` | attention head 数 |
| `N_CTX` | sequence length，本文也写作 $N$ |
| `HEAD_DIM` | 每个 head 的维度，本文写作 $d$ |
| `q`, `k`, `v` | Q/K/V 输入 |
| `o` | Attention 输出 O |
| `M` | 前向保存的每行 log-sum-exp，供反向重算概率 |
| `do` | 上游梯度 $dO$ |
| `dq`, `dk`, `dv` | Q/K/V 的梯度 |

对单个 batch、单个 head，朴素 Attention 是：

$$
S=sQK^\top,\qquad
P=\operatorname{softmax}(S),\qquad
O=PV.
$$

causal 模式还要求：

$$
S_{ij}=-\infty\quad\text{if }j>i.
$$

### 2.2 tile 命名

前向中：

- `BLOCK_M`：一次处理多少行 Q；
- `BLOCK_N`：内层循环一次处理多少行 K/V，即 score 的多少列；
- `HEAD_DIM`：归约维度，通常是 64 或 128；
- `start_m`：当前 Q tile 的编号，而不是绝对 token 下标；
- `start_n`：当前 K/V tile 的绝对起始 token 下标。

一个前向 program 的主要矩阵形状为：

```text
q       [BLOCK_M, HEAD_DIM]
k.T     [HEAD_DIM, BLOCK_N]
qk      [BLOCK_M, BLOCK_N]
p       [BLOCK_M, BLOCK_N]
v       [BLOCK_N, HEAD_DIM]
acc     [BLOCK_M, HEAD_DIM]
```

这正好构成两次 Tensor Core 友好的矩阵乘：

```text
qk  = q @ k.T
acc = p @ v + acc
```

---

## 3. 为什么普通 Attention 撞上 memory wall

朴素实现往往拆成多个 kernel：

```text
QKᵀ → 写 S 到 HBM
读 S → mask/softmax → 写 P 到 HBM
读 P 和 V → PV → 写 O
```

当 $N$ 很大时，S 和 P 都有 $N^2$ 个元素。例如单个 head、`N=8192`、FP16 时，一个 $N^2$ 矩阵约为 128 MiB；多 batch、多 head 后中间量迅速膨胀。

融合 kernel 改成：

```text
load Q tile once
for each K/V tile:
    load K/V tile
    compute local QKᵀ
    update online softmax state
    immediately accumulate P @ V
store O tile once
```

因此完整 S/P 从未成为 HBM 中的实体。代价是 K/V 会被不同 Q tile 重读，但避免了更昂贵的 $N^2$ 中间矩阵落盘，并提升了片上数据复用。

需要严谨地区分：源码注释会说 Q、K、V “stay in SRAM”，但 Triton 层面的 tensor value 最终具体落在寄存器、shared memory 还是由编译器以其他方式调度，取决于 lowering 和目标架构。学习时可把它理解为“保持在片上存储层级，不写回 HBM”，不要机械等同于 CUDA 源码中的显式 `__shared__` 数组。

---

## 4. 从入口开始读：`_attention.forward`

建议先从 `class _attention(torch.autograd.Function)` 读，而不是一上来钻进 inner loop。

### 4.1 形状与输出

forward 首先检查 Q/K/V 的 head dimension 一致，并限制：

```python
HEAD_DIM_K in {16, 32, 64, 128, 256}
```

输出 `o = torch.empty_like(q)`，另分配 FP32 的 `M`：

```text
M.shape = [Z, H, N_CTX]
```

`M` 不是普通的行最大值；前向结束时它已经变成 base-2 表示下的 log-sum-exp。反向用它直接恢复归一化概率。

### 4.2 `stage = 3 if causal else 1`

这里的 `STAGE` 不是 pipeline 的 `num_stages`。二者必须分清：

| 名字 | 含义 |
|---|---|
| `STAGE` | 算法控制位：是否运行 causal 对角区和非对角区 |
| `num_stages` | Triton 编译/软件流水参数，影响异步 load 的 pipeline 深度 |

`STAGE` 被当作 bit mask：

- 非 causal：外层传 `STAGE=1`，只运行一次 inner，覆盖全部 K/V；
- causal：外层传 `STAGE=3`，先跑严格下三角的无 mask 区，再跑对角带的 mask 区。

这样做的目的，是让大部分 causal 区域不执行逐元素 mask。只有穿过对角线的 tile 才承担 mask 分支。

### 4.3 launch grid

```python
grid = (
    ceil_div(N_CTX, BLOCK_M),
    Z * H,
    1,
)
```

所以一个 program 对应：

```text
(一个 Q token 行块, 一个 batch-head 对)
```

程序内：

```python
start_m = tl.program_id(0)
off_hz  = tl.program_id(1)
off_z   = off_hz // H
off_h   = off_hz % H
```

这体现了 Triton 的 block-level 编程模型：程序员描述“一个 program 怎样操作一组 tile”，不是逐 CUDA thread 编写标量逻辑。tile 内怎样映射到 threads/warps 由编译器和 `num_warps` 等 meta-parameters 决定。

### 4.4 为什么展平 `Z × H × N_CTX`

源码令：

```python
y_dim = Z * H * N_CTX
```

并把常规 Q/K/O 描述成 `[y_dim, HEAD_DIM]`。某个 batch-head 的 token 起点为：

$$
\text{offset\_y}=z(NH)+hN.
$$

当前 Q tile 的起点再加：

$$
\text{qo\_offset\_y}=\text{offset\_y}+\text{start\_m}\cdot B_M.
$$

这利用了 `[Z,H,N,D]` contiguous tensor 中最后两维连续的事实，将 batch/head 寻址与二维 tile load 解耦。

### 4.5 Tensor Descriptor 路径

源码同时支持：

1. host 侧创建 `TensorDescriptor`；
2. 直接把 tensor/pointer 传进 kernel，再由 `_maybe_make_tensor_desc` 创建设备侧 descriptor。

`_host_descriptor_pre_hook` 会在 autotune config 确定后，把各 descriptor 的 `block_shape` 改成当前 tile：

| Tensor | block shape |
|---|---|
| Q | `[BLOCK_M, HEAD_DIM]` |
| K | `[BLOCK_N, HEAD_DIM]` |
| V（普通） | `[BLOCK_N, HEAD_DIM]` |
| O | `[BLOCK_M, HEAD_DIM]` |

Hopper/Blackwell 相关分支是实现细节，不应与 FlashAttention 的核心数学混在一起背。先掌握 tile 和 online softmax，再看 descriptor、TMA、warp specialization。

---

## 5. 前向外层：`_attn_fwd`

### 5.1 初始化三个在线状态

```python
m_i = -inf                      # [BLOCK_M]
l_i = 1.0                       # [BLOCK_M]
acc = 0.0                       # [BLOCK_M, HEAD_DIM]
```

数学上更自然的初值是 `l_i = 0`。源码用 `1` 也成立，因为第一次更新时：

$$
\alpha=2^{-\infty-m_{new}}=0,
$$

旧 `l_i` 会被乘成 0，因此不会污染结果。这一写法有利于生成简单、稳定的代码。

全部状态用 FP32 累积，最后才转回输出 dtype。

### 5.2 为什么乘 `1/log(2)`

源码执行：

```python
qk_scale = sm_scale * 1.44269504  # 1 / ln(2)
p = exp2(qk * qk_scale - m)
```

因为：

$$
2^{x/\ln 2}=e^x.
$$

因此：

$$
2^{s(QK^\top)/\ln2}=e^{s(QK^\top)}.
$$

使用 `exp2` 是 GPU kernel 中常见的性能实现方式。这里的 `m_i` 和最终的 `M` 都处于 base-2 指数坐标，所以反向也必须沿用同一套尺度。

### 5.3 Q tile 常驻

```python
q = desc_q.load([qo_offset_y, 0])
```

Q tile 在进入 inner loop 前只加载一次。随后每轮换入 K/V tile。这正是“固定输出行块、流式扫描列块”的数据复用方向。

### 5.4 causal 的两段式遍历

设当前 Q tile 覆盖行：

$$
[mB_M,(m+1)B_M).
$$

causal 时分两段：

1. off-band：K/V 列在 `[0, m*BLOCK_M)`，整个 tile 位于对角线左侧，不需要 mask；
2. on-band：列在 `[m*BLOCK_M, (m+1)*BLOCK_M)`，tile 穿过对角线，需要逐元素 mask。

对角线右侧的列完全不访问。这同时减少了无效计算和 HBM 读取。

源码的控制略绕：外层 `STAGE=3` 时，第一次传给 inner 的是 `4 - STAGE = 1`，第二次传 `2`。非 causal 外层 `STAGE=1` 时，inner 得到 `3`，表示扫描 `[0, N_CTX)`。

### 5.5 epilogue

inner loop 结束后：

```python
m_i += log2(l_i)
acc /= l_i[:, None]
store M
store O
```

所以保存的：

$$
M_i=m_i+\log_2 l_i
   =\log_2\sum_j 2^{x_{ij}},
$$

其中 $x_{ij}=s(QK^\top)_{ij}/\ln2$。它就是 softmax 分母的 base-2 log-sum-exp。

---

## 6. 前向核心：`_attn_fwd_inner`

### 6.1 一轮循环做了什么

每次迭代读一个 K/V tile：

```python
k  = load K tile, then transpose
qk = dot(q, k)
apply scale and optional causal mask
update online max
p = exp2(shifted qk)
rescale old accumulator
acc = dot(p, v, acc)
update l_i and m_i
```

矩阵形状流：

```text
[BM, D] @ [D, BN] -> [BM, BN]
[BM, BN] @ [BN, D] -> [BM, D]
```

局部 `qk` 和 `p` 只在这一轮存在，用完即丢弃，因此不会写成全局 $N^2$ 张量。

### 6.2 online softmax 推导

假设处理新 tile 之前，对某一行已经维护：

$$
m_i=\max_{j\in\text{old}}x_{ij},
$$

$$
l_i=\sum_{j\in\text{old}}2^{x_{ij}-m_i},
$$

$$
acc_i=\sum_{j\in\text{old}}2^{x_{ij}-m_i}V_j.
$$

新 tile 的行最大值记为 $m_{tile}$，合并后的最大值：

$$
m_{new}=\max(m_i,m_{tile}).
$$

旧状态原先以 $m_i$ 为基准，现在必须改到 $m_{new}$ 基准：

$$
\alpha=2^{m_i-m_{new}}.
$$

于是：

$$
p_{tile}=2^{x_{tile}-m_{new}},
$$

$$
l_{new}=\alpha l_i+\sum p_{tile},
$$

$$
acc_{new}=\alpha acc_i+p_{tile}V_{tile}.
$$

最终：

$$
O_i=\frac{acc_i}{l_i}.
$$

源码逐行对应：

```python
m_ij = maximum(m_i, max(qk, axis=1))
qk  -= m_ij[:, None]
p     = exp2(qk)
alpha = exp2(m_i - m_ij)
l_ij  = sum(p, axis=1)
acc   = acc * alpha[:, None]
acc   = dot(p, v, acc)
l_i   = l_i * alpha + l_ij
m_i   = m_ij
```

### 6.3 为什么必须缩放旧 `acc`

常见错误是更新了 `m_i` 和 `l_i`，却忘了 `acc *= alpha`。旧 `acc` 中的权重以旧最大值为基准；最大值变化后，如果不重标定，旧 tile 和新 tile 不在同一指数尺度上，结果必错。

可以用极小例子自检：旧 score 为 `[0]`，新 score 为 `[100]`。新最大值变为 100，旧贡献必须乘 $e^{-100}$，否则旧值仍会获得不合理权重。

### 6.4 mask 为什么用 `-1.0e6`

对角块中：

```python
mask = query_index >= key_index
qk = qk * qk_scale + where(mask, 0, -1.0e6)
```

`exp2(-1e6)` 数值上为 0，效果等价于 $-\infty$，同时避免某些低精度/编译路径对无穷值处理的复杂性。

### 6.5 `tl.multiple_of` 的性质

`tl.multiple_of` 是给编译器的对齐/整除提示，不是运行时做取整。错误地把它理解成 `round()` 会看不懂地址计算，也可能在自己写 kernel 时向编译器提供不真实的约束。

### 6.6 累积精度

- `qk`/`p` 会走适合 Tensor Core 的低精度输入；
- `m_i`、`l_i`、`acc` 使用 FP32；
- `p` 在送入第二个 `tl.dot` 前转成目标 dtype；
- 输出在 store 前转回 FP16 或 FP8。

这是典型的 mixed-precision 策略：乘法吞吐依赖低精度，归约和 softmax 稳定性依赖 FP32。

---

## 7. autotune 与硬件参数

候选参数覆盖：

```text
BLOCK_M   ∈ {64, 128}
BLOCK_N   ∈ {32, 64, 128}
num_stages ∈ {1} on HIP, otherwise {2, 3, 4}
num_warps ∈ {4, 8}
```

autotune key 包含：

```text
N_CTX, HEAD_DIM, FP8_OUTPUT, warp_specialize
```

含义是：这些输入特征变化时，需要重新选择最优 config；batch/head 没放入 key，是因为每个 program 的局部工作形状主要由序列长度、head dimension、dtype 和调度模式决定。

### 7.1 config 剪枝

源码在 benchmark 前先排除明显无效/不合适的组合，例如：

- `BLOCK_M > N_CTX`；
- causal 时通常要求 `BLOCK_M >= BLOCK_N`，便于对角区域切分；
- Hopper 上排除某些“小 tile + 8 warps”组合。

这说明 autotune 不是把所有组合无脑实测。工程上应先用算法约束、资源约束和架构经验缩小搜索空间。

### 7.2 参数的主要权衡

| 参数变大 | 可能的收益 | 可能的代价 |
|---|---|---|
| `BLOCK_M` | Q 重用更高，K/V 重读更少，GEMM 更大 | `acc` 更大，寄存器压力/occupancy 变差 |
| `BLOCK_N` | 每轮 GEMM 更大，循环次数更少 | `qk`/`p` tile 更大，片上空间增加 |
| `num_warps` | 单 program 并行度提高 | 每个 program 使用更多线程，驻留 program 数可能降低 |
| `num_stages` | 更好地重叠 load 与 compute | shared memory/寄存器占用增加 |

### 7.3 warp specialization

warp specialization 的目标是让不同 warp 承担不同流水职责，如加载和计算，从而提升 overlap。源码还按 Hopper/Blackwell 和具体 shape 调 `maxnreg`，说明它是强架构相关优化，不是一个在所有设备上都必然更快的布尔开关。

学习优先级：

1. 先用 `warp_specialize=False` 理清算法；
2. 再比较生成代码和 profiler 数据；
3. 最后解释为何特定架构/shape 获益或退化。

---

## 8. FP8 路径

当 `q.dtype == torch.float8_e5m2` 时，源码进入 FP8 forward 路径：

- Q/K 转为 FP8；
- V 的底层存储通过两次 permute 构造为转置友好的 layout；
- V descriptor 按 `[HEAD_DIM, y_dim]` 描述；
- inner loop 的 V load 路径不同；
- 输出也是 FP8，测试比较前再转 half；
- 当前教程明确不支持 FP8 backward。

这里“两次 permute”看似抵消，实际作用在于改变底层 storage/stride：第一次转置后 `contiguous()` 真正重排数据，第二次只恢复逻辑维度顺序，因此逻辑 shape 回来了，物理 layout 没回到原始方式。

这是读 GPU kernel 必须养成的习惯：不能只看 shape，还要看 stride 和 storage layout。

---

## 9. 反向数学

对单行 Attention：

$$
P=\operatorname{softmax}(S),\qquad O=PV.
$$

给定上游梯度 $dO$：

$$
dV=P^\top dO,
$$

$$
dP=dOV^\top.
$$

softmax Jacobian 可化成逐元素形式：

$$
dS=P\odot(dP-\Delta),
$$

其中每一行：

$$
\Delta_i=\sum_j P_{ij}dP_{ij}.
$$

利用 $O_i=\sum_jP_{ij}V_j$，有：

$$
\Delta_i
=\sum_jP_{ij}(dO_i\cdot V_j)
=dO_i\cdot\left(\sum_jP_{ij}V_j\right)
=dO_i\cdot O_i.
$$

因此无需保存或显式生成完整 dP，就可以先计算：

$$
\Delta_i=\langle O_i,dO_i\rangle.
$$

最后：

$$
dQ=s\,dSK,
$$

$$
dK=s\,dS^\top Q.
$$

这组公式对应全部 backward kernel。

---

## 10. 反向预处理：`_attn_bwd_preprocess`

预处理 kernel 每个 program 处理一个 `BLOCK_M × HEAD_DIM` 的 O/dO tile：

```python
delta = sum(o * do, axis=1)
```

得到：

```text
Delta.shape = [Z, H, N_CTX]
Delta_i = <O_i, dO_i>
```

这是把 softmax backward 中原本需要跨列归约的量提前压缩成每行一个标量。主 backward kernel 随后只需 load `Delta_i`。

---

## 11. 主反向：`_attn_bwd`

### 11.1 launch 与固定参数

wrapper 采用：

```text
BLOCK_M1 = 32,  BLOCK_N1 = 128    # dK/dV 路径
BLOCK_M2 = 128, BLOCK_N2 = 32     # dQ 路径
BLK_SLICE_FACTOR = 2
num_warps = 4
num_stages = 5
```

grid：

```python
(N_CTX // BLOCK_N1, 1, Z * H)
```

每个 program 的 `pid` 同时对应：

- 一个 128 行的 K/V block，用于累积 dK/dV；
- 一个 128 行的 Q block，用于累积 dQ。

这是工作分区上的重要观察：dK/dV 和 dQ 需要不同的扫描方向，因此使用两套互为转置感的 block 参数。

### 11.2 为什么 backward 重算 P

forward 只保存 O 和每行 `M`，没有保存 $N^2$ 的 P。backward 使用：

```python
p = exp2(qk - M)
```

重建归一化后的概率。因为：

$$
M_i=\log_2\sum_j2^{x_{ij}},
$$

所以：

$$
2^{x_{ij}-M_i}
=\frac{2^{x_{ij}}}{\sum_k2^{x_{ik}}}
=P_{ij}.
$$

这体现了 FlashAttention 的经典 trade-off：多做一部分 FLOPs，换取少得多的 HBM 流量与显存占用。在现代 GPU 上通常值得，因为 Attention 经常受 IO 限制。

### 11.3 base-2 缩放为何看起来不对称

wrapper 先做：

```python
arg_k = k * (sm_scale / ln(2))
```

于是 backward 中 `q @ arg_k.T` 直接处于 `exp2` 所需坐标，可与保存的 `M` 相减。

对 dK：

```python
dk += ds.T @ q
dk *= sm_scale
```

对 dQ，点乘使用的是已预缩放的 K，因此得到的量多了 `sm_scale / ln(2)`；末尾：

```python
dq *= ln(2)
```

最终也变成正确的 `sm_scale * ds @ K`。

不要孤立地看最后一个 `LN2`。必须连同 forward 保存的 base-2 `M`、预缩放 `arg_k`、`exp2` 一起检查量纲。

---

## 12. `dK/dV` 内层：`_attn_bwd_dkdv`

这个函数固定一块 K/V：

```text
k, v: [BLOCK_N1, HEAD_DIM]
```

然后沿 Q/dO 的 token 行方向扫描：

```text
q.T: [HEAD_DIM, BLOCK_M1]
dO : [BLOCK_M1, HEAD_DIM]
```

每轮：

1. 重算 `qkT = k @ qT`；
2. 用 `pT = exp2(qkT - m)` 恢复 $P^\top$ tile；
3. `dv += pT @ do`；
4. `dpT = v @ do.T`；
5. `dsT = pT * (dpT - Delta)`；
6. `dk += dsT @ q`。

形状验证：

```text
pT   [BN1, BM1]
do   [BM1, D]
dv   [BN1, D]

v    [BN1, D]
do.T [D, BM1]
dpT  [BN1, BM1]

dsT  [BN1, BM1]
q     [BM1, D]
dk    [BN1, D]
```

只要用 shape 把三次 dot 写出来，指针转置就不再神秘。

---

## 13. `dQ` 内层：`_attn_bwd_dq`

这个函数固定一块 Q/dO：

```text
q, dO: [BLOCK_M2, HEAD_DIM]
```

然后沿 K/V 列方向扫描：

```text
k.T, v.T: [HEAD_DIM, BLOCK_N2]
```

每轮：

1. `qk = q @ kT`；
2. `p = exp2(qk - m)`；
3. `dp = do @ vT`；
4. `ds = p * (dp - Delta)`；
5. `dq += ds @ k`。

形状：

```text
q     [BM2, D]
kT    [D, BN2]
p     [BM2, BN2]

do    [BM2, D]
vT    [D, BN2]
dp    [BM2, BN2]

ds    [BM2, BN2]
k     [BN2, D]
dq    [BM2, D]
```

dQ 和 dK/dV 分开设计扫描方向，是 FlashAttention-2 “更好的工作划分”在代码中的具体体现之一。

---

## 14. backward 中的 causal 切分

与 forward 相同，反向也把对角块和非对角块分开：

- 对角附近用更小的 `MASK_BLOCK_M1` 或 `MASK_BLOCK_N2` 并执行 mask；
- 完全位于合法三角区的 block 使用 `MASK=False`；
- 完全位于未来 token 区域的 block 不扫描。

`BLK_SLICE_FACTOR=2` 会把 mask 区的 tile 再切小一半。这样能减少对角边界附近的无效 lane 工作，但也增加循环/控制开销，是精细的边界优化。

理解 causal 指针范围的可靠方法不是死记 `start_m/start_n`，而是画一个 QKᵀ 方阵：

```text
key j →
query i ↓

valid valid invalid invalid
valid valid valid   invalid
valid valid valid   valid
...
```

对 dK/dV：固定列块，向下扫描所有允许关注该 key 的 query。  
对 dQ：固定行块，向左扫描该 query 能看到的 key。

---

## 15. 测试代码应该怎样读

`test_op` 不只是附属代码，它定义了实现的正确性契约。

覆盖维度包括：

- batch：1、4；
- heads：2、48；
- sequence：128、1024、较长序列；
- head dim：64、128；
- causal / non-causal；
- forward / backward；
- FP16，以及设备支持时的 FP8 forward；
- Blackwell 上的 warp specialization。

reference 路径显式构造：

```python
p = q @ k.transpose(-2, -1) * sm_scale
p = causal_mask(p)
p = softmax(p.float())
ref_out = p @ v
```

随后比较 output 以及 dQ/dK/dV。FP8 使用显著更宽的绝对误差，因为其表示精度远低于 FP16。

### 15.1 自己改 kernel 时应补的测试

官方参数并不覆盖所有边界。建议补：

- 第一行 causal attention，确保只关注 token 0；
- 全零 Q/K，输出应是 V 的均值或 causal 前缀均值；
- 极大 score，验证无 NaN/Inf；
- `N_CTX` 不等于单一 tile 的小 shape；
- 不同非 contiguous layout；
- head dim 16/32/256；
- 与 PyTorch SDPA 同时比较 forward 和 gradient；
- 固定随机种子，多组尺度分布。

---

## 16. benchmark 代码怎样解释

benchmark 以 TFLOPS 报告 Triton FP16、可用时的 FP8，以及安装了 `flash-attn` 时的 FlashAttention-2 provider。

单次前向有两次主 matmul：

1. $QK^\top$；
2. $PV$。

每次 matmul FLOPs 近似：

$$
2BHN^2d.
$$

所以 non-causal forward 总 FLOPs：

$$
4BHN^2d.
$$

causal 只计算下三角，近似乘 0.5。backward 还包含重算，源码按 forward 总量再乘 2.5 做吞吐估算。

注意 TFLOPS 是由理论 FLOPs 除以实测时间得到的有效吞吐，不等价于 kernel 实际执行的每一条指令计数。mask 边界、padding、类型转换和地址计算并未完全体现在这个公式里。

---

## 17. 与 SYS-601 四个核心概念的映射

### 17.1 SRAM 分块计算（Tiling）

- Q 按 `BLOCK_M × HEAD_DIM` 分块；
- K/V 按 `BLOCK_N × HEAD_DIM` 分块；
- score tile 是 `BLOCK_M × BLOCK_N`；
- 只保留一个 score/probability tile 和一个输出累加 tile；
- tile size 决定数据重用、寄存器压力和 occupancy。

### 17.2 在线 Softmax

- `m_i`：运行最大值；
- `l_i`：指数和；
- `alpha`：最大值变化时对旧状态的尺度修正；
- `acc`：与 `l_i` 同尺度的加权 V 和；
- `M = m_i + log2(l_i)`：反向重算概率所需的压缩状态。

### 17.3 算子融合

一个 forward kernel 内完成：

```text
matmul + scale + mask + max-reduction + exp + sum-reduction + matmul + normalize
```

融合减少 kernel launch 和 HBM 中间读写，但增大单 kernel 的活跃状态，可能带来寄存器 spill 或 occupancy 下降。因此“融合越多越好”不是普遍真理，需要 profiler 验证。

### 17.4 Thread Block 与 Warp 工作划分

- Triton program 近似对应 CUDA thread block/CTA 级工作单元；
- grid 先把 `(Q tile, batch-head)` 分给 programs；
- `num_warps` 决定一个 program 采用多少 warps；
- tile 内元素如何分发给 lanes 由 Triton layout/lowering 决定；
- warp specialization 在支持的架构上进一步拆分 load/compute 职责。

---

## 18. 与本地 Lab 的逐项映射

本地 `flash_attention_v1.py` 是理解官方 forward 的最佳跳板：它保留同一数学递推，但通过 PyTorch 循环表达，尚未真正融合为单个 Triton kernel。

| 本地 Lab | 官方 Triton |
|---|---|
| 外层 `for i in range(0, n, br)` | grid 的 `program_id(0)`，Q blocks 并行 |
| `qi = q[..., i:i_end, :]` | `desc_q.load([qo_offset_y, 0])` |
| 内层 `for j in range(0, n, bc)` | `tl.range(lo, hi, BLOCK_N)` |
| `s = qi @ kj.T` | `qk = tl.dot(q, k)` |
| `m_new = max(m_i, m_ij)` | 同名 online max 更新 |
| `exp(m_i-m_new)` | `alpha = exp2(m_i-m_ij)` |
| `o_i = alpha*o_i + p@vj` | `acc = alpha*acc; dot(p,v,acc)` |
| `o_i / l_i` | epilogue `acc / l_i[:,None]` |
| Python causal mask | 先切 off-band/on-band，只在对角带 mask |

本地 `flash_attention_v2.py` 只把 `(B,H)` reshape 为更大的 batch，并不等价于完整的官方 FA-2 kernel。它是“调度提示实验”，而官方源码还包含：

- Triton program 级并行；
- 真正的 kernel fusion；
- autotune；
- causal 对角切分；
- 自定义 backward 与概率重算；
- descriptor/FP8/硬件专用路径。

建议学习顺序：

1. 在纸上推导本地 v1 的 `m/l/o` 更新；
2. 用小矩阵逐步打印每个 tile 的状态；
3. 对照 `_attn_fwd_inner`，逐句建立映射；
4. 再看 program grid 和 pointer/descriptor；
5. 最后进入 backward 和硬件调优。

---

## 19. 容易误读或踩坑的地方

### 19.1 `M` 不是单纯 max

它先作为运行最大值 `m_i`，epilogue 后保存的是：

```text
max + log2(exp-sum) = log2sumexp
```

反向 `exp2(qk - M)` 得到的是归一化 P，而非未归一化指数。

### 19.2 两种 “stage” 不同

`STAGE` 是 causal 算法位；`num_stages` 是 load/compute 软件流水深度。

### 19.3 `start_m` 的单位会变化

前向 outer 中 `start_m` 是 Q block 编号；进入 inner 后用 `start_m * BLOCK_M` 变成 token 起点。反向 helper 的 `start_m/start_n` 通常已是绝对 token offset。阅读时必须看调用点。

### 19.4 shape 相同不代表 layout 相同

FP8 V 在两次 permute 后恢复逻辑 shape，但 stride/storage 已改变。descriptor 也随之变化。

### 19.5 backward 有更强的 shape 前提

当前 wrapper 明确断言：

```python
N_CTX % 128 == 0
```

且要求 Q/K/V/O/dO strides 相同、dO contiguous。移植到通用生产算子时必须处理 tail、不同 strides、GQA/MQA、variable length 等情况。

### 19.6 `main` 分支不是稳定教材快照

当前查阅的源码中，`forward` 有 6 个显式输入：

```text
q, k, v, causal, sm_scale, warp_specialize
```

但 `backward` 末尾在当前页面显示返回 `dq, dk, dv` 加 4 个 `None`，合计 7 项。按 PyTorch `autograd.Function` 的常规契约，返回项应与 forward 输入项一一对应；Triton 仓库也有针对这一点的 [Issue #7666](https://github.com/triton-lang/triton/issues/7666)。复现实验前应检查所安装版本是否已经修正，并以实际 commit 为准。

### 19.7 教程 kernel 不等于完整生产 API

该源码重点展示算法与 Triton 技术，不应默认覆盖生产库的全部能力。缺失或受限项包括部分 shape、FP8 backward、任意 strides、dropout、attention bias、GQA/MQA、paged KV cache、可变长度序列等。

---

## 20. 建议的 profiler 实验

### 实验 A：朴素 Attention vs 本地 block 版 vs 融合版

固定：

```text
B=4, H=32, D=64
N ∈ {1024, 2048, 4096, 8192}
```

记录：

- forward latency；
- backward latency；
- peak allocated memory；
- achieved memory bandwidth；
- Tensor Core utilization；
- occupancy；
- registers per thread；
- shared memory per block；
- kernel launch 数量。

预期：朴素路径中会看到 score/softmax/PV 的多个 kernel 及大中间量；融合版 kernel 数更少，HBM 流量显著下降。

### 实验 B：tile size 与 occupancy

固定输入，比较：

```text
(BM, BN) = (64,32), (64,64), (128,64), (128,128)
num_warps = 4 or 8
```

同时记录吞吐和寄存器 spill。目标不是只找最快配置，而是解释：

```text
tile 变大
→ 数据重用/GEMM 效率可能提高
→ acc/qk/p 活跃状态变大
→ 寄存器或 shared memory 压力上升
→ occupancy 可能下降
```

### 实验 C：causal 与 non-causal

比较两种模式的：

- 理论 FLOPs；
- 实际 latency；
- TFLOPS；
- 分支/mask 开销。

causal 理论工作约减半，但时间通常不会精确减半，因为对角 mask、launch 固定开销、流水利用率和小循环都会影响结果。

### 实验 D：验证 online softmax 状态

在本地 PyTorch v1 中为一个极小输入打印每轮：

```text
m_i, alpha, l_i, o_i
```

并与一次性 `torch.softmax` 对比。建议使用能让新 tile 最大值剧烈变化的数据，以验证 `alpha` 的必要性。

---

## 21. Nsight Compute 阅读清单

查看融合 kernel 时优先回答：

1. DRAM 吞吐是否接近峰值？若接近且 Tensor Core 不忙，仍可能 memory-bound；
2. Tensor Core pipe 是否充分利用？
3. occupancy 低是寄存器限制、shared memory 限制还是 block 数不足？
4. 是否出现 local memory load/store，暗示 register spill？
5. `BLOCK_M/BLOCK_N` 变化怎样影响 wave 数与 tail effect？
6. causal 下有效 Tensor Core 工作比例是多少？
7. load 与 dot 是否有良好 overlap，`num_stages` 增加是否真有收益？
8. kernel 前后是否仍存在不必要的 layout conversion/copy？

不要只看 occupancy。Attention kernel 可以在较低 occupancy 下依靠高数据重用和指令级并行获得高吞吐；最终应以延迟、吞吐和瓶颈指标共同判断。

---

## 22. 可以独立复述的面试答案

### Q1：FlashAttention 是否降低了计算复杂度？

没有。精确 Attention 仍是 $O(N^2d)$ FLOPs。它主要降低 HBM IO 和中间显存，通过 tile、融合和 online softmax 避免物化 $N^2$ 的 S/P。

### Q2：为什么 online softmax 数值稳定？

每处理一个 tile 都维护当前行最大值，并让指数在减去合并最大值后计算；最大值变化时用 `alpha = exp(old_max-new_max)` 重标定旧的分母与输出累加器。

### Q3：为什么 backward 不保存 P？

P 是 $N^2$ 中间量，保存会破坏 IO 优势。forward 只保存每行 log-sum-exp；backward 重算 QKᵀ tile，并用 `exp(score-LSE)` 恢复 P tile。

### Q4：为什么 dK/dV 与 dQ 使用不同 tile 方向？

dK/dV 适合固定 K/V 行块后扫描 Q/dO；dQ 适合固定 Q/dO 行块后扫描 K/V。分别选择更适合输出累加方向的 tile 能改善并行和数据复用。

### Q5：为什么使用 `exp2`？

GPU 上 base-2 指数通常有高效实现。将自然指数输入乘 `1/ln(2)` 后，`exp2(x/ln2)` 与 `exp(x)` 等价；前后向必须统一保存量和梯度中的尺度。

### Q6：大 tile 为什么不一定快？

它提高数据重用并减少循环，但也扩大 qk/p/acc 的片上状态，可能造成寄存器压力、spill、shared memory 占用增加和 occupancy 下降。

### Q7：Triton program 与 CUDA thread 有何区别？

Triton 代码通常描述一个 program 对整个 tensor tile 的操作；编译器把 tile 运算映射到该 program 的多个 threads/warps。`program_id` 更接近 CUDA block/CTA 级索引，而不是单 thread 索引。

---

## 23. 手写推导检查表

完成本次 source code study 后，应能不看源码写出：

- [ ] 标准 Attention 的 S、P、O 三式；
- [ ] causal mask 的合法区域；
- [ ] `m/l/acc` 的 online softmax 更新式；
- [ ] `alpha` 的来源；
- [ ] `M = m + log2(l)` 为什么足够重算 P；
- [ ] `Delta = rowsum(O * dO)` 的推导；
- [ ] `dV = PᵀdO`；
- [ ] `dP = dOVᵀ`；
- [ ] `dS = P * (dP - Delta)`；
- [ ] `dQ = s dS K` 与 `dK = s dSᵀQ`；
- [ ] forward grid 中两个 program id 的含义；
- [ ] `BLOCK_M/BLOCK_N/num_warps/num_stages` 的权衡；
- [ ] 为什么 causal 要区分 off-band 和 on-band；
- [ ] 为什么不物化 P 可以降低 HBM 流量；
- [ ] shape、stride、layout 三者的区别。

---

## 24. 一页复习版

```text
目标:
    O = softmax(s QKᵀ) V

program mapping:
    pid0 -> Q 的 BLOCK_M 行
    pid1 -> 一个 (batch, head)

forward:
    Q tile 只 load 一次
    循环 load K/V tile
    qk = Q @ Kᵀ
    m_new = max(m_old, rowmax(qk))
    alpha = exp2(m_old - m_new)
    p = exp2(qk - m_new)
    acc = alpha * acc + p @ V
    l = alpha * l + rowsum(p)
    O = acc / l
    M = m + log2(l)                # 保存 LSE

causal:
    严格下三角 tile -> 无 mask 快路径
    对角 tile       -> mask 路径
    上三角 tile     -> 不访问

backward:
    Delta = rowsum(O * dO)
    P = exp2(recomputed_qk - M)
    dP = dO @ Vᵀ
    dS = P * (dP - Delta)
    dV = Pᵀ @ dO
    dQ = s dS @ K
    dK = s dSᵀ @ Q

性能本质:
    不写 N×N 的 S/P 到 HBM
    用更多重算换更少 IO
    tile 越大不一定越快：重用 vs 片上资源/occupancy
```

---

## 25. 后续实践任务

1. 给本地 `flash_attention_v1.py` 加一个 debug 模式，逐 tile 输出 `m/l/acc`；
2. 实现一个只支持 FP16、non-causal、forward 的最小 Triton kernel；
3. 加 causal 两段式遍历，不在所有 tile 上统一 mask；
4. 对 `BLOCK_M/BLOCK_N` 做小规模 autotune；
5. 保存 LSE，并实现 `Delta` 预处理 kernel；
6. 先实现 dQ，再实现 dK/dV；
7. 用 PyTorch SDPA 做 forward/gradient reference；
8. 用 Nsight Compute 解释最快与最慢 config 的差异；
9. 记录 GPU 型号、Triton/PyTorch/CUDA 版本和源码 commit，保证结果可复现。

## 参考资料

- [Triton 官方 fused attention 源码](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)
- [Triton 官方 Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Triton GitHub Issue #7666：backward 返回项数量](https://github.com/triton-lang/triton/issues/7666)

