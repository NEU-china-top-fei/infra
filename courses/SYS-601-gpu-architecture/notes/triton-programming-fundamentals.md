# Triton GPU 编程基础：从 Vector Add 到 Tiled Matmul

> 课程：SYS-601 GPU Architecture and Operator Optimization  
> 主参考：[Introduction to GPU Programming with Triton](https://medium.com/@katherineolowookere/introduction-to-gpu-programming-with-triton-d7412289bd51)，Katherine Oluwadarasimi Olowookere，2025-12-10  
> 官方核对：[Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)  
> 本地练习：[`../labs/lab02-triton-fundamentals/`](../labs/lab02-triton-fundamentals/)  
> 整理日期：2026-09-07  
> 公式格式：行内 `$...$`、块级 `$$...$$`，兼容 VS Code Markdown 数学渲染。

## 1. 这篇文章要解决什么问题

文章的核心问题不是“怎样用 Python 做 GPU 计算”，而是：

> 怎样用一个比 CUDA 更高层的 block/tile 编程模型，仍然显式控制数据移动，从而写出高性能 GPU kernel？

全文通过三个逐渐复杂的例子回答：

```text
Vector Add
  └─ program id、grid、offset、pointer、mask、load/store

Fused Softmax
  └─ 行归约、数值稳定、算子融合、减少 HBM 往返

Blocked Matmul
  └─ 二维 tile、K 维循环、FP32 accumulator、L2 分组、autotune
```

读完后应建立四个核心认识：

1. Triton kernel 是从一个 program instance 的视角描述一个数据 tile；
2. GPU 优化通常首先是数据移动问题，其次才是算术表达式问题；
3. 正确的 offset、stride 和 mask 是 kernel 正确性的基础；
4. tiling、fusion、program ordering 和 autotune 是性能优化的主线。

---

## 2. 为什么需要 Triton

### 2.1 PyTorch eager 的优点与成本

PyTorch eager mode 让模型代码容易编写、调试和修改，但一个由多个 tensor op 组成的表达式，可能对应多个 GPU kernel：

```python
y = torch.exp(x - x.max(dim=-1, keepdim=True).values)
out = y / y.sum(dim=-1, keepdim=True)
```

逻辑上只有 softmax，运行时却可能经历：

```text
kernel 1: reduce max
kernel 2: subtract
kernel 3: exp
kernel 4: reduce sum
kernel 5: divide
```

每个 kernel 之间都可能需要把中间结果写回 HBM，再由下一 kernel 读回。代价包括：

- kernel launch overhead；
- 额外的全局内存读写；
- 中间 tensor 分配；
- cache locality 变差；
- 对 memory-bound 算子尤其不利。

现代 PyTorch 编译器可能自动融合部分图，因此“eager 表达式一定产生五个 kernel”不是永恒规则；但这仍是理解 fusion 动机的正确模型。

### 2.2 CUDA 与高层框架之间的空档

CUDA 提供 thread、warp、shared memory、同步和指令级细节的精细控制，但学习和开发成本高。纯框架算子使用方便，却不一定能表达新的融合算法或特殊数据布局。

Triton 位于两者之间：

| 层次 | 开发者主要控制 | 系统主要处理 |
|---|---|---|
| PyTorch tensor op | 算子组合与张量语义 | kernel 实现和调度 |
| Triton | tile、grid、offset、load/store、归约、meta-parameters | tile 内 thread/warp 映射及大量低层 lowering |
| CUDA | thread/block、shared memory、同步、指令与内存细节 | 编译与硬件执行 |

Triton 的价值不是“Python 比 C++ 快”，而是其 DSL 和编译器允许开发者以 tile 为单位表达算法，同时保留对关键数据移动的控制。

---

## 3. GPU 执行层次：先知道谁在做工作

### 3.1 CUDA 常见层次

```text
Grid
  └─ Thread Block / CTA
      └─ Warp
          └─ Thread
```

| 概念 | 作用 |
|---|---|
| Thread | 最小的标量执行上下文，拥有私有寄存器状态 |
| Warp | NVIDIA GPU 上通常为 32 threads 的调度单位 |
| Thread Block / CTA | 在同一 SM 上协作的一组 threads，可使用 shared memory 与同步 |
| Grid | 一次 kernel launch 的全部 blocks |
| SM | 调度 warps、分配寄存器/shared memory、执行计算的硬件单元 |

### 3.2 Triton program instance

Triton 不要求开发者为每个 CUDA thread 写标量程序。开发者写的是：

```text
一个 program instance 应处理哪个 tile，以及怎样处理这个 tile？
```

常见近似关系：

```text
Triton program instance ≈ CUDA thread block / CTA 级工作单元
```

但这只是理解上的类比，不是所有后端和所有编译路径都保证严格一一对应。更可靠的表述是：program instance 是 Triton launch grid 中可独立索引的程序实例，编译器再把其 block operations 映射到目标硬件的 threads/warps。

### 3.3 SPMD 心智模型

所有 program instances 执行同一份 kernel 代码，但 `program_id` 不同：

```python
pid = tl.program_id(axis=0)
```

因此它们通过不同 `pid` 计算不同地址，覆盖输出的不同区域。

以长度 `N=1000`、`BLOCK_SIZE=128` 为例：

```text
grid = ceil_div(1000, 128) = 8

pid 0 -> logical offsets   0..127
pid 1 -> logical offsets 128..255
...
pid 7 -> logical offsets 896..1023
```

最后一个实例中的 1000..1023 越界，因此必须用 mask 禁止对应 load/store。

---

## 4. GPU 内存层次与性能主线

从靠近计算到远离计算，可以用以下简化层次理解：

```text
Registers
    ↓ 容量增大、延迟通常增大
Shared Memory / L1
    ↓
L2 Cache
    ↓
HBM / Global Memory
```

### 4.1 各层作用

| 层级 | 特点 | kernel 中的典型用途 |
|---|---|---|
| Registers | 每个 thread 私有，最快、最小 | 标量、tile 的局部值、accumulator |
| Shared Memory | SM 片上、block/CTA 内共享 | 跨线程数据复用、显式 staging |
| L1/L2 Cache | 硬件管理 | 自动缓存近期 global memory 访问 |
| HBM | 容量大、带宽高但相对片上层级慢 | 输入、输出、模型权重、大 tensor |

### 4.2 必须修正的简化说法

文章为了教学，常把 `tl.load` 描述为“从 DRAM 加载到 SRAM”。更严谨的说法是：

- `tl.load` 表达从给定 pointer 读取 block；
- 读取会经过 GPU cache/memory hierarchy；
- 读取结果成为 Triton IR 中的 SSA tensor value；
- 编译器可能把值放在 registers，或为某些操作使用 shared memory/staging；
- 不能仅凭一条 `tl.load` 就断言它一定进入显式 CUDA shared memory。

学习阶段真正需要把握的是：tile 在片上被消费和复用，避免反复显式写回 HBM。

### 4.3 黄金原则

GPU kernel 优化可以先用一句话概括：

> 从 HBM 少读、少写；数据加载到片上后尽量多复用；让数据移动和计算重叠。

这会导出本文后面的全部技术：

- fusion：减少中间 tensor 的 HBM 往返；
- tiling：让一份输入 tile 服务更多 FLOPs；
- grouping：提高 L2 reuse；
- software pipelining：重叠下一 tile 的加载与当前 tile 的计算；
- autotune：在复用和片上资源压力之间找平衡。

---

## 5. Triton kernel 的基本构成

一个 Triton 算子通常有两部分。

### 5.1 device kernel

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    ...
```

它描述 GPU 上每个 program instance 的工作。

### 5.2 host wrapper

```python
def op(x):
    out = torch.empty_like(x)
    grid = (...,)
    kernel[grid](x, out, ..., BLOCK_SIZE=...)
    return out
```

它运行在 CPU/Python 侧，负责：

- 检查 shape、dtype、device、layout；
- 分配输出；
- 计算 launch grid；
- 传入 runtime arguments 和 compile-time meta-parameters；
- launch kernel。

### 5.3 JIT 发生在何时

`@triton.jit` 标记函数可以被 Triton 编译。真正 launch 时，编译器已经知道：

- 目标 GPU/backend；
- runtime 参数的类型；
- `tl.constexpr` meta-parameters；
- 部分可用于 specialization 的信息。

第一次遇到新的 specialization 通常包含编译成本；后续相同变体可复用 cache。benchmark 时必须 warm up，不能把首次编译时间当 kernel latency。

### 5.4 runtime 参数与 `tl.constexpr`

```python
def kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

| 参数 | 性质 | 例子 |
|---|---|---|
| runtime argument | launch 时传值，生成的 kernel 从参数读取 | pointer、`n_elements`、stride |
| `tl.constexpr` | 编译期已知，可决定 shape/循环展开/分支特化 | `BLOCK_SIZE`、`BLOCK_M/N/K` |

tile shape 通常必须在编译期已知，所以 block size 常标为 `tl.constexpr`。

---

## 6. 第一个 kernel：Vector Add

本地对应代码：[`vector_add.py`](../labs/lab02-triton-fundamentals/vector_add.py)。

### 6.1 算法

$$
z_i=x_i+y_i,\qquad 0\le i<N.
$$

这是 elementwise、无归约、无跨元素依赖的操作，适合学习 Triton 最小闭环。

### 6.2 kernel 骨架

下面是与本地 Lab 一致的精简版本：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

host wrapper：

```python
def triton_add(x, y):
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

### 6.3 逐句理解

#### `pid = tl.program_id(axis=0)`

读取当前 program 在 launch grid 第 0 维的编号。

#### `tl.arange(0, BLOCK_SIZE)`

创建一个 block value：

```text
[0, 1, 2, ..., BLOCK_SIZE - 1]
```

这不是在 Python 中创建普通 list，也不是说每个 program 只有一个硬件 thread。它表达 program 内部的一组逻辑 lanes/elements，后续由编译器映射。

#### `offsets = pid * BLOCK_SIZE + ...`

把 tile 内相对 offset 转换成全局逻辑索引。

#### `x_ptr + offsets`

pointer arithmetic 是按 pointer element type 计步，不是手动按 byte 计算。若 `x_ptr` 指向 FP32，`+1` 表示下一个 FP32 元素。

#### `mask = offsets < n_elements`

保护最后一个不完整 tile。mask 必须同时用于 load 和 store；只保护 load 但不保护 store 仍会越界写。

#### `other=0.0`

masked load 的填充值。Vector Add 的越界 lane 最终不会 store，因此取 0 主要让中间表达式安全清晰。

### 6.4 grid 为什么向上取整

$$
\text{grid size}=\left\lceil\frac{N}{\text{BLOCK\_SIZE}}\right\rceil.
$$

若向下取整，尾部元素永远没有 program 处理。向上取整后 over-launch，再由 mask 处理 tail，是 GPU kernel 的常见模式。

### 6.5 性能模型

Vector Add 对每个元素执行：

- 读 x；
- 读 y；
- 写 z；
- 只做一次加法。

算术强度极低，通常是 memory-bound。若元素大小为 $b$ bytes，理想数据流量约为：

$$
3Nb\ \text{bytes}.
$$

因此 benchmark 更适合报告有效带宽：

$$
\text{GB/s}=\frac{3Nb}{t}\times 10^{-9}.
$$

短向量主要受 launch overhead 影响；足够长后才能接近稳定带宽区间。

---

## 7. Offset、Pointer、Stride：最容易出错的基础

### 7.1 二维 row-major 地址

对二维 tensor `X`：

$$
\operatorname{addr}(X[i,j])
=X_{ptr}+i\cdot stride_0+j\cdot stride_1.
$$

PyTorch/Triton 传入的 stride 通常以“元素数”为单位。例如 contiguous 的 `[M,N]` tensor：

```text
stride(0) = N
stride(1) = 1
```

### 7.2 为什么不要写死 stride

若默认 `stride_row = n_cols`，只适用于特定 contiguous layout。转置、切片或 view 可能拥有不同 stride。通用 wrapper 应传入实际：

```python
x.stride(0), x.stride(1)
```

### 7.3 广播如何生成二维 pointer block

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BM]
offs_k = tl.arange(0, BLOCK_K)                     # [BK]

a_ptrs = (
    a_ptr
    + offs_m[:, None] * stride_am                  # [BM, 1]
    + offs_k[None, :] * stride_ak                  # [1, BK]
)
```

广播后 `a_ptrs` 形状为 `[BM, BK]`，每项对应一个 A 元素的地址。

### 7.4 调试 pointer 的 shape 表

写 kernel 时建议先写表，不要只在脑中推：

| 值 | 逻辑 shape |
|---|---|
| `offs_m` | `[BM]` |
| `offs_m[:, None]` | `[BM, 1]` |
| `offs_k[None, :]` | `[1, BK]` |
| `a_ptrs` | `[BM, BK]` |
| `offs_k[:, None]` | `[BK, 1]` |
| `offs_n[None, :]` | `[1, BN]` |
| `b_ptrs` | `[BK, BN]` |

若 `tl.dot(a, b)` 需要输出 `[BM, BN]`，那么输入必须是 `[BM, BK] @ [BK, BN]`。shape 表能快速发现转置错误。

---

## 8. 第二个 kernel：Fused Softmax

本地对应代码：[`softmax.py`](../labs/lab02-triton-fundamentals/softmax.py)。

### 8.1 数学定义

对矩阵的一行 $x$：

$$
\operatorname{softmax}(x_i)
=\frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

直接计算可能在大正数处 overflow，因此使用稳定形式：

$$
m=\max_jx_j,
$$

$$
\operatorname{softmax}(x_i)
=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}.
$$

减去常数不改变 softmax，因为分子和分母同时乘了 $e^{-m}$。

### 8.2 简单的一行一 program 版本

```python
@triton.jit
def softmax_kernel(inp_ptr, out_ptr,
                   input_stride_row, output_stride_row, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    in_ptrs = inp_ptr + row * input_stride_row + offsets
    x = tl.load(in_ptrs, mask=mask, other=-float("inf"))

    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    out_ptrs = out_ptr + row * output_stride_row + offsets
    tl.store(out_ptrs, y, mask=mask)
```

launch grid：

```python
grid = (n_rows,)
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

### 8.3 为什么 block size 取 2 的幂

Triton block operations 对 shape 有编译期约束，基础教程用 `next_power_of_2(n_cols)` 把整行放进一个合法 block。例如：

```text
n_cols = 768
BLOCK_SIZE = 1024
```

多出来的 256 lanes 通过 mask 屏蔽。

### 8.4 masked value 为什么是 `-inf`

归约最大值时，越界 lane 不应改变合法最大值：

$$
\max(x,-\infty)=x.
$$

随后：

$$
e^{-\infty}=0,
$$

所以越界 lane 也不会进入分母。这说明 `other` 必须根据后续归约的单位元/吸收元选择：

| 后续操作 | 常见 masked fill |
|---|---|
| sum | 0 |
| max | `-inf` |
| min | `+inf` |
| product | 1 |

### 8.5 fusion 的价值

未融合实现会产生 max、减法、exp、sum、除法等中间结果。融合 kernel 的理想数据路径是：

```text
HBM read row once
  -> max/sub/exp/sum/div on chip
HBM write result once
```

最低显式 global traffic 约为一个输入读和一个输出写，即：

$$
2MN b\ \text{bytes}.
$$

文章给出的 eager traffic 计数适合说明“中间 tensor 很贵”，但实际读写次数会受 PyTorch 版本、编译/融合、cache 和 kernel 实现影响，不应当作所有环境下固定不变的硬件计数。

### 8.6 简单 softmax 的限制

一行全部放入一个 program 意味着 row 很宽时：

- block value 很大；
- register/shared-memory 压力上升；
- 可能 spill；
- 一行无法适配可接受的片上资源；
- occupancy 可能严重下降。

当前 Triton 官方教程已展示 persistent scheduling：一个 program 以 `tl.range(row_start, n_rows, row_step)` 处理多行，并基于设备资源决定 program 数量。文章和本地 Lab 的“一行一 program”仍适合入门，但不是所有 shape 的最终优化方案。

---

## 9. Fusion 不是无条件越多越好

融合的主要收益：

- 减少 launch；
- 减少中间 tensor；
- 减少 HBM traffic；
- 保留 producer-consumer locality；
- 可能避免重复计算地址和同步边界。

融合的主要代价：

- 活跃变量更多；
- register pressure 增大；
- shared memory 占用可能增加；
- instruction footprint 增大；
- occupancy 下降；
- 某些算子之间的最佳 tile 不一致；
- 过度融合可能导致 spill，反而增加 global/local memory traffic。

正确做法是使用 profiler 判断。不能仅凭 kernel 数更少就断言一定更快。

---

## 10. 第三个 kernel：Blocked/Tiled Matmul

本地对应代码：[`matmul.py`](../labs/lab02-triton-fundamentals/matmul.py)。

### 10.1 数学定义

$$
A\in\mathbb{R}^{M\times K},\qquad
B\in\mathbb{R}^{K\times N},
$$

$$
C=AB,\qquad
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}.
$$

总计算量约为：

$$
2MNK\ \text{FLOPs}.
$$

### 10.2 为什么逐元素算法低效

如果每个输出元素独立从 HBM 读取完整 A 行和 B 列：

- 同一 A 元素被多个输出列重复读取；
- 同一 B 元素被多个输出行重复读取；
- 缺少片上复用；
- B 的列访问还可能具有不友好的连续性；
- 很难充分利用 Tensor Cores。

### 10.3 tiled 算法

把输出 C 分为 `[BLOCK_M, BLOCK_N]` tiles。一个 program 固定一个 C tile，并沿 K 维扫描：

```text
acc[BM, BN] = 0

for k0 in 0, BK, 2BK, ...:
    a[BM, BK] = A[m0:m0+BM, k0:k0+BK]
    b[BK, BN] = B[k0:k0+BK, n0:n0+BN]
    acc += a @ b

store C tile
```

同一个 A tile 的每个元素参与 `BLOCK_N` 个输出，同一个 B tile 的每个元素参与 `BLOCK_M` 个输出。这就是数据复用。

### 10.4 单个 K tile 的算术强度

一轮加载元素数：

$$
B_MB_K+B_KB_N.
$$

执行 FLOPs：

$$
2B_MB_NB_K.
$$

忽略 cache 和输出写，若元素大小为 $b$ bytes，tile 层面的近似 arithmetic intensity：

$$
AI\approx
\frac{2B_MB_NB_K}
{b(B_MB_K+B_KB_N)}.
$$

增大 `BLOCK_M`/`BLOCK_N` 可提高复用，但 accumulator 大小为 `BLOCK_M × BLOCK_N`，会增加寄存器压力。这是 matmul 调参的核心矛盾。

---

## 11. Matmul program mapping

### 11.1 最容易理解的二维 grid

本地 Lab 使用：

```python
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)

grid = (
    triton.cdiv(M, BLOCK_M),
    triton.cdiv(N, BLOCK_N),
)
```

这让每个 program 直接对应一个 `(pid_m, pid_n)` 输出 tile，最适合入门。

### 11.2 官方教程常用一维 grid

官方高性能版本通常使用一维 `pid`，再映射到二维 tile 坐标。这样可自定义 program launch ordering，为 L2 locality 做 grouping：

```python
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
```

普通 row-major 映射可写成：

```python
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

高性能版本会使用 grouped ordering，后文详解。

### 11.3 输出 tile 与输入 tile

给定 `(pid_m, pid_n)`：

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)
```

首个 K chunk 的 pointers：

```python
a_ptrs = a_ptr + (
    offs_m[:, None] * stride_am
    + offs_k[None, :] * stride_ak
)

b_ptrs = b_ptr + (
    offs_k[:, None] * stride_bk
    + offs_n[None, :] * stride_bn
)
```

形状：

```text
a_ptrs: [BLOCK_M, BLOCK_K]
b_ptrs: [BLOCK_K, BLOCK_N]
acc:    [BLOCK_M, BLOCK_N]
```

---

## 12. Matmul K-loop 逐步解读

### 12.1 accumulator

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

即使输入/输出是 FP16，通常也使用 FP32 accumulator，以降低长 K 维归约的误差。

### 12.2 tail mask

```python
for k in range(0, tl.cdiv(K, BLOCK_K)):
    k_offsets = k * BLOCK_K + offs_k

    a = tl.load(
        a_ptrs,
        mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
        other=0.0,
    )

    b = tl.load(
        b_ptrs,
        mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
        other=0.0,
    )
```

越界输入用 0，因为 0 是乘加归约的安全填充值。

### 12.3 tile dot

```python
acc = tl.dot(a, b, acc)
```

逻辑上执行：

$$
acc\leftarrow acc+a\times b.
$$

在满足 dtype、shape、layout 和硬件条件时，编译器可映射到 Tensor Core 路径。不要在性能版中随意先把低精度 `a/b` 转成 FP32；这可能改变 `tl.dot` 所走的硬件路径。常见策略是低精度 operands + FP32 accumulator。

### 12.4 pointer advance

```python
a_ptrs += BLOCK_K * stride_ak
b_ptrs += BLOCK_K * stride_bk
```

它们都沿逻辑 K 维前进一个 chunk。通过增量更新 pointer block，可避免每轮重建全部地址表达式。

### 12.5 store output

K-loop 全部结束后再转换并写回：

```python
c = acc.to(tl.float16)
c_ptrs = c_ptr + (
    offs_m[:, None] * stride_cm
    + offs_n[None, :] * stride_cn
)
mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, c, mask=mask_c)
```

必须确认 dtype cast 和 store 在 K-loop 外；若每轮都覆盖 C，只会得到不完整的 partial sum，若每轮把 accumulator 降精度也会增大误差。

### 12.6 本地 Lab 的一个性能观察

本地 `matmul.py` 为清晰起见，把 load 后的 A/B tile 转成 FP32：

```python
a = tl.load(...).to(tl.float32)
b = tl.load(...).to(tl.float32)
```

它适合作为 correctness starter，但若目标是 Tensor Core 高吞吐，应实验保留 FP16/BF16 operands，仅让 accumulator 为 FP32，并检查生成代码与 Nsight Compute 指标。不要只根据数值误差猜测是否用了 Tensor Core。

---

## 13. Super Grouping：通过 program 顺序改善 L2 locality

### 13.1 为什么 program 顺序有影响

相邻 C tiles 会复用输入：

- 同一 tile row 的多个 C tiles 复用 A 的相同行块；
- 同一 tile column 的多个 C tiles 复用 B 的相同列块。

若最近执行的 programs 落在相邻输出区域，A/B tiles 更可能仍留在 L2 中。虽然程序语义不能依赖具体执行顺序，但 launch id 的排列可影响调度的局部性倾向。

### 13.2 grouped mapping

官方教程的核心映射：

```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

它把 M 方向的若干 tile rows 组成一个 group，并在 group 内以更利于复用的次序遍历 N tiles。

### 13.3 小例子

设：

```text
num_pid_m = 5
num_pid_n = 4
GROUP_SIZE_M = 2
```

前 8 个 program 属于 group 0，覆盖 M tile rows 0、1 和全部 4 个 N tile columns。它们会在有限区域内集中访问 A/B，从而提高某些输入 tile 的 L2 命中概率。

### 13.4 注意边界 group

最后一个 group 可能不足 `GROUP_SIZE_M`，所以必须使用：

```python
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
```

后续取模也必须用 `group_size_m`，而不是固定的 `GROUP_SIZE_M`，否则最后一组会映射到不存在的 tile rows。

### 13.5 不要把 grouping 误解成同步

grouping 只改变 program id 到输出 tile 的映射顺序：

- 不让 programs 直接共享 registers/shared memory；
- 不提供跨 program barrier；
- 不保证某个 program 必然先于另一个执行；
- 收益主要来自 cache locality 和调度倾向。

---

## 14. Autotuning

### 14.1 为什么需要 autotune

最佳 tile 与以下因素相关：

- M/N/K shape；
- dtype；
- GPU 架构；
- register file 和 shared memory 容量；
- Tensor Core tile 支持；
- 输入 layout；
- fusion 后的额外状态；
- `num_warps` 与 `num_stages`。

不存在对所有问题都最优的固定 `(BLOCK_M, BLOCK_N, BLOCK_K)`。

### 14.2 基本结构

```python
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32,
             "GROUP_SIZE_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32,
             "GROUP_SIZE_M": 8},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(...):
    ...
```

首次遇到新的 key 时，Triton 会为候选配置生成/测试变体并缓存选择。生产 benchmark 应区分：

- compile/tune time；
- warm kernel latency；
- cache hit 后的 steady-state latency。

### 14.3 参数权衡

| 参数 | 变大可能带来的收益 | 变大可能带来的代价 |
|---|---|---|
| `BLOCK_M/N` | 更高输入复用、更大 GEMM | accumulator 增大、register pressure 上升 |
| `BLOCK_K` | K-loop 次数减少、更大 dot | operands/staging 增大、tail 浪费 |
| `num_warps` | program 内并行度提高 | 每个 program 线程资源增加，驻留数下降 |
| `num_stages` | 更好地 overlap load/compute | shared memory/register 使用增加 |
| `GROUP_SIZE_M` | L2 locality 可能改善 | 不合适时破坏另一输入的 locality |

### 14.4 先剪枝再调优

不要把所有组合无脑加入搜索。应先排除：

- 不满足 `tl.dot` shape/dtype 要求的配置；
- tile 大于问题规模太多的配置；
- 明显超过片上资源的配置；
- 对特定架构无效的 `num_stages/num_warps`；
- 边界浪费严重的配置。

autotune 是实验选择器，不会替开发者修复错误算法或错误内存布局。

---

## 15. `num_warps`、`num_stages` 与 occupancy

### 15.1 `num_warps`

它指定/提示一个 Triton program 使用的 warps 数。更多 warps 可能让大 tile 更快完成，但也会增加一个 program 的线程资源占用。

### 15.2 `num_stages`

主要控制软件流水深度，目标是：

```text
计算当前 K tile
同时预取后续 K tile
```

更深 pipeline 可以隐藏内存延迟，却需要更多 staging 资源。

### 15.3 occupancy 不是唯一目标

occupancy 高只表示 SM 上有较多活跃 warps，不自动等于高性能。一个 matmul kernel 即使 occupancy 较低，也可能凭借：

- 高 Tensor Core utilization；
- 高数据复用；
- 良好的软件流水；
- 足够的 instruction-level parallelism；

获得更高吞吐。最终要结合 latency、Tensor Core 利用率、memory throughput 和 stall reasons 判断。

---

## 16. Benchmark 应该怎样做

### 16.1 正确性先于性能

每个 kernel 先与可信 reference 比较：

```python
torch.testing.assert_close(actual, expected, atol=..., rtol=...)
```

覆盖：

- 非整 tile shape；
- 小 shape；
- 大 shape；
- 不同 dtype；
- 不同 stride/layout；
- 极端数值；
- 若有 backward，使用 gradient reference。

### 16.2 warm up

排除：

- JIT compile；
- autotune 搜索；
- allocator 首次初始化；
- GPU frequency 从 idle 提升的影响。

### 16.3 GPU 是异步的

普通 Python wall-clock 若不同步，测到的可能只是 enqueue 时间。优先使用 `triton.testing.do_bench` 或正确的 CUDA event/synchronize 流程。

### 16.4 指标选择

| 算子 | 更直观指标 |
|---|---|
| Vector Add | latency、有效 GB/s |
| Softmax | latency、有效 GB/s、row width scaling |
| Matmul | latency、TFLOPS、Tensor Core utilization |

Matmul 吞吐：

$$
\text{TFLOPS}=\frac{2MNK}{t}\times10^{-12}.
$$

若 $t$ 以秒为单位使用上式；若是毫秒，要先乘 $10^{-3}$ 转秒。

### 16.5 baseline 必须公平

- 相同 device/dtype/shape；
- 相同输入 layout；
- 相同精度语义，例如 TF32 是否开启；
- 不把数据转换成本只算在某一方；
- 不比较一个专用 kernel 与一个功能范围更广的通用 API 后直接下绝对结论；
- 报告 GPU、驱动、CUDA/ROCm、PyTorch、Triton 版本。

文章中的性能图反映作者环境，不应直接外推到另一型号 GPU 或另一 Triton 版本。

---

## 17. 调试 Triton kernel

文章推荐 `TRITON_INTERPRET=1`，官方文档还提供更多层次的调试手段。

### 17.1 Interpreter

```bash
TRITON_INTERPRET=1 python vector_add.py
```

作用：

- 不走 GPU 编译；
- 用 NumPy 等价操作解释 kernel；
- program instances 顺序执行；
- 可用 Python `print`、`pdb` 查看中间值。

已知限制包括部分 BF16 操作和某些间接内存访问模式，因此 interpreter 通过不代表 GPU backend 一定通过，interpreter 失败也不一定代表 GPU kernel 逻辑完全错误。

### 17.2 编译期检查

```python
tl.static_assert(BLOCK_K % 16 == 0)
tl.static_print(BLOCK_M, BLOCK_N, BLOCK_K)
```

用于验证 `tl.constexpr` 条件和 specialization 信息。

### 17.3 设备端检查

```python
tl.device_print("pid", pid)
tl.device_assert(condition, "message")
```

`device_assert` 需要相应 debug 配置/环境。大量 device print 会显著扰动执行，只适合小 shape。

### 17.4 内存错误

NVIDIA 环境可使用：

```bash
compute-sanitizer python matmul.py
```

用于发现越界和部分数据竞争。它很慢，应先缩小 shape。

### 17.5 推荐排错顺序

```text
1. 最小 shape + PyTorch reference
2. 手写每个中间 tensor 的 shape
3. 验证 pid -> output tile 映射
4. 验证 pointer = base + Σ(offset * stride)
5. 验证 load/store 的 mask 和 other
6. Interpreter 打印单个 pid
7. device_print / static_assert
8. compute-sanitizer
9. 检查 TTIR/TTGIR/LLVM IR/PTX
10. Nsight 分析性能问题
```

正确性工具和性能工具不要混用：Nsight 指标不能证明数值正确，reference test 也不能证明没有越界未触发。

---

## 18. 最常见的 bug

### 18.1 忘记 tail mask

症状：只在非整 block shape 崩溃或随机错误。

### 18.2 load 有 mask，store 没 mask

症状：输入读取安全，但输出尾部越界写。

### 18.3 `other` 与归约不匹配

例如 max reduction 用 0 padding，会在所有合法值为负数时错误地把最大值变成 0。

### 18.4 stride 写死

contiguous 测试通过，转置或切片输入错误。

### 18.5 广播轴写反

`offs_m[None, :]` 与 `offs_m[:, None]` 混淆，造成 pointer block shape 不符合 dot。

### 18.6 K-loop pointer 没前进或前进维度错误

症状：反复乘第一个 K tile，或访问越界。

### 18.7 K-loop 内提前 cast/store

症状：只保留 partial sum，或每轮发生低精度舍入。

### 18.8 `tl.constexpr` 缺失

当参数参与 `tl.arange`、block shape 或 compile-time 分支时，未标为 compile-time constant 会导致编译错误或无法按预期特化。

### 18.9 误把 program id 当元素 id

`pid` 通常标识 tile；元素 offset 还要加 `tl.arange`。

### 18.10 只测整齐的 1024×1024

整 tile shape 会隐藏大量边界错误。至少加入 1000、1003、513×769 等 irregular shapes。

---

## 19. 文章中的简化表述：怎样准确理解

### 19.1 “Triton 自动保证 coalescing”

应理解为 Triton 编译器负责许多 thread-level 映射，但开发者定义的 pointer block 仍决定访问模式。若 offsets 跨很大 stride 或随机跳转，编译器无法把任意逻辑访问神奇地变成理想连续访问。

### 19.2 “program 就是 CUDA block”

适合作为入门类比，但最好使用“CTA 级工作单元的近似心智模型”。后端 lowering 和新架构特性可能更复杂。

### 19.3 “load 到 SRAM”

应理解为读取后的 tile 在片上参与计算。具体使用 registers、shared memory 或何种 staging，由编译器与 backend 决定。

### 19.4 “L2 不可控制”

开发者通常不能像 shared memory 那样直接寻址和管理 L2，但可通过访问顺序、program grouping、数据 layout 和复用模式间接影响命中率。

### 19.5 “自定义 Triton 一定比 PyTorch/cuBLAS 快”

不成立。Triton 的优势是容易编写专用、可融合、可定制 kernel；vendor libraries 在标准 GEMM 上非常成熟。胜负取决于 shape、架构、融合机会、版本、精度和调参质量。

---

## 20. 从文章到本地 Lab

### 20.1 文件映射

| 学习主题 | 本地文件 | 重点观察 |
|---|---|---|
| Vector Add | [`vector_add.py`](../labs/lab02-triton-fundamentals/vector_add.py) | 1D grid、tail mask |
| Fused Softmax | [`softmax.py`](../labs/lab02-triton-fundamentals/softmax.py) | 一行一 program、归约、`-inf` padding |
| Tiled Matmul | [`matmul.py`](../labs/lab02-triton-fundamentals/matmul.py) | 2D grid、pointer broadcasting、K-loop |

### 20.2 本地 Lab 与文章/官方版的差异

| 项目 | 本地 Lab | 文章/官方高性能方向 |
|---|---|---|
| Vector block | 固定 1024 | 可 benchmark/autotune 多种大小 |
| Softmax scheduling | 一行一 program | 当前官方教程支持 persistent rows |
| Matmul grid | 直观 2D grid | 1D grid + grouped ordering 改善 L2 |
| Matmul operands | load 后转 FP32 | 常见为低精度 operands + FP32 acc |
| Autotune | 无 | 多组 BM/BN/BK/warps/stages |
| Benchmark | 只做 correctness | 应加 latency、GB/s、TFLOPS |

这不表示本地 Lab 写错了：它的目标是分阶段暴露概念。正确学习方式是先让简单版本完全可解释，再逐项增加高性能机制，每次都重新做 correctness 和 benchmark。

---

## 21. 推荐实践顺序

### 阶段 1：Vector Add

- [ ] 画出 `N=10, BLOCK=4` 的三个 programs 和 offsets；
- [ ] 去掉 mask，观察 irregular shape 的失败；
- [ ] 比较 block size 128/256/512/1024；
- [ ] 报告短向量和长向量的 latency/GB/s；
- [ ] 支持不同 dtype。

### 阶段 2：Softmax

- [ ] 推导减 max 不改变 softmax；
- [ ] 用全负输入验证 masked fill 必须为 `-inf`；
- [ ] 比较 `n_cols=768` 与 `BLOCK_SIZE=1024`；
- [ ] 测试极大/极小输入是否有 NaN；
- [ ] 比较 eager、`torch.softmax`、Triton；
- [ ] 观察宽行导致的资源压力；
- [ ] 阅读官方 persistent softmax。

### 阶段 3：Matmul

- [ ] 用纸写出 `[BM,BK] @ [BK,BN]`；
- [ ] 测 irregular M/N/K；
- [ ] 验证每个 stride 的单位和方向；
- [ ] 去掉 FP32 operand cast，比较精度与 Tensor Core 指标；
- [ ] 添加 grouped 1D program mapping；
- [ ] 添加 autotune；
- [ ] 与 `torch.matmul` 比较 TFLOPS；
- [ ] 用 Nsight 检查 register spill、occupancy 和 Tensor Core utilization。

### 阶段 4：连接 FlashAttention

完成前三个例子后，FlashAttention 可理解为：

```text
Tiled Matmul (QKᵀ)
  + Fused/Online Softmax
  + Tiled Matmul (PV)
  + 不物化 N×N 中间矩阵
```

然后继续阅读：[`source-code-study-06-fused-attention.md`](source-code-study-06-fused-attention.md)。

---

## 22. 写一个新 Triton kernel 的模板思路

### 22.1 先写算法契约

```text
Inputs:
Outputs:
Shapes:
Dtypes:
Layouts/strides:
Boundary conditions:
Numerical tolerance:
Reference implementation:
```

### 22.2 决定 program 的输出所有权

问：

> 一个 program 独占输出的哪个 tile？

最好让不同 programs 写不重叠输出，避免原子操作和跨 program 协调。

### 22.3 从输出 tile 反推输入 tile

```text
output tile
  -> 需要哪些输入坐标
  -> 如何生成 offsets
  -> 如何乘 strides 得到 pointers
  -> 哪些维度需要循环
```

### 22.4 为每个 load/store 写 mask

逐维检查：

```text
M boundary?
N boundary?
K boundary?
batch/head boundary?
```

### 22.5 决定累积精度

归约、softmax、norm 和长 dot 通常需要比输入更高的累积精度。

### 22.6 最后才调性能

推荐顺序：

```text
correct scalar/reference math
  -> correct tile mapping
  -> correct boundary handling
  -> benchmark
  -> profile
  -> tune tile/warps/stages/layout/order
```

---

## 23. 面试高频问答

### Q1：Triton 与 CUDA 的核心编程模型差异是什么？

CUDA 通常从 thread 视角写标量程序；Triton 从 program instance 视角对 tensor block/tile 编程，编译器负责大量 tile 内 thread/warp 映射。开发者仍控制 grid、tile、offset、pointer、mask 和数据移动。

### Q2：为什么 Vector Add 通常 memory-bound？

每个元素只有一次加法，却需要两次读取和一次写入，算术强度极低。性能更多由 HBM 带宽和 launch overhead 决定。

### Q3：为什么 softmax 要减去最大值？

softmax 对整体平移不变。减最大值后最大的指数输入为 0，可避免大正数 exponent overflow，并改善数值稳定性。

### Q4：为什么 softmax 的 masked lane 填 `-inf`？

它不会影响 max reduction，指数后又变成 0，不会进入分母。

### Q5：Matmul tiling 为什么减少 HBM traffic？

一个 A tile 在输出 tile 的多个列间复用，一个 B tile在多个行间复用；加载一次后执行许多乘加，提高 arithmetic intensity。

### Q6：为什么 accumulator 用 FP32？

K 维包含大量累加，低精度误差会不断积累。低精度 operands 提供 Tensor Core 吞吐，FP32 accumulator 提升数值稳定性。

### Q7：`num_warps` 越大越好吗？

不是。它可能提高单 program 并行度，也会增加资源占用并减少并发驻留 programs。必须结合 tile 和硬件实测。

### Q8：Super Grouping 做了什么？

它重排 program id 到 C tiles 的映射，让近期 programs 更集中地访问相关 A/B tiles，从而提高 L2 locality；它不提供跨 program 共享内存或同步。

### Q9：autotune 的 key 有什么作用？

key 指定哪些 runtime problem features 变化时需要重新选择最佳配置。若遗漏会导致错误复用不合适的配置，若加入过多则增加编译和调优缓存数量。

### Q10：怎样判断一个 Triton kernel 是否真的更快？

在相同 shape、dtype、layout、精度语义和设备上，先验证正确性，完成 warmup，再用 GPU-aware benchmark 测 latency/GB/s/TFLOPS，并用 profiler 确认瓶颈，而不是只看单次 Python 计时。

---

## 24. 一页复习版

```text
Triton 心智模型:
    一个 program instance 处理一个输出 tile
    pid 选择 tile
    arange 生成 tile 内 offsets
    pointer = base + Σ(offset * stride)
    mask 保护 boundary
    load -> compute/reduce/dot -> store

Vector Add:
    grid = ceil(N / BLOCK)
    offsets = pid * BLOCK + arange(BLOCK)
    2 reads + 1 write + 1 add
    memory-bound

Softmax:
    one row/tile per program（基础版）
    x <- load(mask, other=-inf)
    x <- x - max(x)
    y <- exp(x) / sum(exp(x))
    one explicit HBM read + one write
    wide rows may exceed efficient on-chip resources

Matmul:
    one program owns C[BM, BN]
    for K chunks:
        A tile [BM, BK]
        B tile [BK, BN]
        acc[BM, BN] += dot(A, B)
    low-precision operands + FP32 acc
    grouped pid ordering improves L2 locality

Performance:
    fusion -> fewer HBM round trips
    tiling -> more reuse / higher arithmetic intensity
    larger tile -> more reuse but more register/shared-memory pressure
    autotune -> choose BM/BN/BK/warps/stages per shape/hardware

Debug:
    shape table -> pointer math -> masks -> interpreter
    -> static/device checks -> compute-sanitizer -> IR/PTX -> profiler
```

---

## 25. 掌握度检查表

- [ ] 能解释 Triton program instance 与 CUDA thread/block 的关系；
- [ ] 能写出 1D Vector Add 的 grid、offset 和 tail mask；
- [ ] 能解释 stride 是元素跨度而非固定 byte 数；
- [ ] 能从一维 offsets 广播出二维 pointer block；
- [ ] 能根据归约选择正确的 masked `other`；
- [ ] 能推导稳定 softmax；
- [ ] 能解释 fusion 怎样减少 HBM traffic；
- [ ] 能画出 tiled matmul 的 `[BM,BK] @ [BK,BN]`；
- [ ] 能解释为什么使用 FP32 accumulator；
- [ ] 能手算 `pid -> (pid_m,pid_n)`；
- [ ] 能解释 grouped ordering 只影响 locality、不提供同步；
- [ ] 能解释 `num_warps` 和 `num_stages` 的资源权衡；
- [ ] 能设计 correctness + benchmark + profiler 三层验证；
- [ ] 能用 `TRITON_INTERPRET=1` 缩小 pointer/shape 问题；
- [ ] 能说明 Triton 并不保证任意访问模式自动 coalesced；
- [ ] 能把 Vector Add、Softmax、Matmul 知识迁移到 FlashAttention。

## 参考资料

- [Introduction to GPU Programming with Triton](https://medium.com/@katherineolowookere/introduction-to-gpu-programming-with-triton-d7412289bd51)
- [Triton 官方：Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton 官方：Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton 官方：Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton 官方：Debugging Triton](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
- [Triton 官方教程索引](https://triton-lang.org/main/getting-started/tutorials/index.html)
