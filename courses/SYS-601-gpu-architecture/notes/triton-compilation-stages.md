# Triton Kernel 编译阶段：从 Python AST 到 GPU 二进制

> 课程：SYS-601 GPU Architecture and Operator Optimization  
> 主参考：[Triton Kernel Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)，Sara Kokkila-Schumacher、Brian Vaughan、Raghu Ganti、Less Wright  
> 官方源码核对：[Triton NVIDIA backend compiler](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py)、[Triton generic compiler](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py)  
> 本地实验：Triton 3.8.0、PyTorch 2.14.0+cu130、RTX 4060 Laptop GPU（compute capability 8.9）  
> 整理日期：2026-09-09  
> 公式格式：行内 `$...$`、块级 `$$...$$`，兼容 VS Code Markdown 数学渲染。

## 1. 这篇文章解决什么问题

写 Triton 时，我们看到的是 Python 风格的 tile-level 程序：

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets, mask=offsets < n)
```

GPU 最终执行的却是由 threads、registers、load/store、branch 和机器指令组成的程序。两者之间需要编译器逐步降低抽象层级：

```text
Python @triton.jit kernel
        │
        ▼ 解析 Python AST、类型与 constexpr 特化
      TTIR                 Triton IR，机器无关的 tile 语义
        │
        ▼ 决定 layout、warp/CTA 映射和目标 GPU 信息
      TTGIR                Triton GPU IR，GPU-aware
        │
        ▼ 降低 Triton/MLIR 操作
      LLVM IR              thread-level、pointer、intrinsic
        │
        ├─ NVIDIA ───────► PTX ─────► cubin ─────► GPU
        │                                └─ 可反汇编为 SASS
        │
        └─ AMD ──────────► AMDGCN ──► HSACO ─────► GPU
```

这条管线的核心思想是：

> 高层 IR 保留算法语义，便于做 tile 级优化；低层 IR 逐渐加入硬件约束，直到生成目标设备能够装载的二进制。

学完后应能回答：

1. TTIR、TTGIR、LLVM IR 分别保留什么信息？
2. Triton 的 block tensor 如何映射到 warps 和 threads？
3. PTX 是不是 GPU 真正执行的机器码？
4. 为什么修改 dtype、`BLOCK_SIZE` 或 `num_warps` 可能触发重新编译？
5. 怎样从一个 Triton kernel 取出各阶段 IR 并追踪同一个操作？

---

## 2. 贯穿全文的 Vector Add

文章使用 vector add，因为它足够简单，可以把注意力集中在编译管线：

```python
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

Host wrapper 完成输出分配、grid 计算和 kernel launch：

```python
def add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    grid = (triton.cdiv(n, 1024),)
    compiled = add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out, compiled
```

这里有两类参数：

| 参数 | 例子 | 编译器视角 |
|---|---|---|
| Runtime argument | pointers、`n_elements` | kernel launch 时传入；IR 中保留函数参数 |
| Compile-time meta-parameter | `BLOCK_SIZE: tl.constexpr` | 编译时已知；可决定 tensor shape、展开和特化 |

在 `BLOCK_SIZE=1024` 的变体中，`tl.arange(0, BLOCK_SIZE)` 会成为固定长度 `tensor<1024xi32>`，而 `BLOCK_SIZE` 通常不会作为普通运行时参数继续传到最终 kernel。

---

## 3. JIT 前端：何时开始编译

### 3.1 `@triton.jit` 不等于立刻编译

装饰器把 Python 函数注册为 JIT function。真正 launch 时，Triton 才获得本次变体所需的信息，例如：

- pointer 指向的元素类型；
- scalar argument 的类型；
- `tl.constexpr` 的具体值；
- target backend 与 GPU architecture；
- `num_warps`、`num_stages` 等编译选项；
- 编译器版本及影响代码生成的配置。

第一次遇到一个新 specialization 时需要编译；后续相同变体通常可以命中 cache。因此 benchmark 必须先 warm up，不能把首次 JIT 时间当作 kernel latency。

### 3.2 Specialization 的直观例子

下面几次调用不一定复用同一份二进制：

```python
add_kernel[grid](x_fp32, y_fp32, out_fp32, n, BLOCK_SIZE=1024)
add_kernel[grid](x_fp16, y_fp16, out_fp16, n, BLOCK_SIZE=1024)
add_kernel[grid](x_fp16, y_fp16, out_fp16, n, BLOCK_SIZE=256)
```

原因是：

- FP32 与 FP16 pointer types 不同；
- `BLOCK_SIZE` 是 constexpr，256 与 1024 产生不同 tile shape；
- tile shape 变化会继续影响 layout、线程工作分配和寄存器需求。

不要把“Python 函数相同”误解为“只有一个 GPU kernel”。更准确的模型是：

```text
一个 @triton.jit 函数
    └─ 多个 specialized compiled kernels
          ├─ dtype/constexpr/target/options 变体 A
          ├─ dtype/constexpr/target/options 变体 B
          └─ ...
```

---

## 4. 阶段一：Python AST → TTIR

### 4.1 AST 做了什么

编译器遍历被 `@triton.jit` 标记函数的 Python AST，把支持的 Python/Triton DSL 结构转换为 Triton IR。此时已经不是在执行普通 Python tensor 运算。

典型对应关系：

| Triton source | TTIR 中的核心操作 |
|---|---|
| `tl.program_id(0)` | `tt.get_program_id` |
| `tl.arange(0, BLOCK_SIZE)` | `tt.make_range` |
| scalar 广播到 block tensor | `tt.splat` |
| pointer + offsets | `tt.addptr` |
| `tl.load(...)` | `tt.load` |
| `x + y` | `arith.addf` 或相应类型的 add |
| `tl.store(...)` | `tt.store` |

### 4.2 TTIR 的定位

TTIR（Triton IR）的主要特征：

- 基于 MLIR 表达；
- 保留 Triton 的 block/tile 编程语义；
- 基本不绑定某个具体 GPU 的 thread/warp layout；
- 已经包含 specialization 后的 dtype 和 constexpr shape；
- 使用 SSA（Static Single Assignment）形式表示数据依赖。

一个高度简化的 TTIR 片段可以读作：

```mlir
%pid = tt.get_program_id x : i32
%range = tt.make_range {start = 0, end = 1024} : tensor<1024xi32>
%base = tt.splat %pid : i32 -> tensor<1024xi32>
%offsets = arith.addi %base, %range : tensor<1024xi32>
%ptrs = tt.addptr %x_base, %offsets
%x = tt.load %ptrs, %mask
```

阅读方法：

1. `%name` 是 SSA value，每个 value 只定义一次；
2. `tensor<1024xi32>` 表示长度 1024 的 i32 block tensor；
3. `tt.*` 属于 Triton dialect；
4. `arith.*` 是 MLIR 的通用算术 dialect；
5. `loc(...)` 可把 IR 操作追溯到原始 kernel 的源码位置。

### 4.3 为什么 TTIR 已经有 dtype

文章展示的参数类型类似：

```text
!tt.ptr<f32>
```

这说明 pointer element type 已在 JIT specialization 时确定。TTIR 虽然仍然是相对机器无关的 IR，但不是完全不带类型的通用模板。

“机器无关”与“类型无关”不是一回事：

- TTIR 可以不知道目标是 `sm_70` 还是 `sm_89`；
- 但它必须知道当前操作是 FP16、FP32 还是整数运算；
- 否则无法检查 `tl.dot`、load/store 和算术操作是否合法。

---

## 5. 阶段二：TTIR → TTGIR

### 5.1 TTGIR 增加了什么

TTGIR（Triton GPU IR，也常写 TritonGPU IR）开始加入 GPU 执行相关信息：

- target backend 和 compute capability；
- threads per warp / wavefront；
- warps per CTA；
- CTAs 数量或 cluster 信息；
- block tensor 在 threads/warps 之间的分布 layout；
- shared-memory、dot operand、matrix-core 等编码；
- 后续 lowering 所需的硬件约束。

文章在 V100 上展示的 module target 是 `cuda:70`。本地 RTX 4060 实测为：

```mlir
module attributes {
  "ttg.num-ctas" = 1,
  "ttg.num-warps" = 4,
  ttg.target = "cuda:89",
  "ttg.threads-per-warp" = 32
}
```

这说明同一个 Triton source 会根据目标设备生成不同 TTGIR。

### 5.2 Layout encoding 是这一阶段的核心

本地 vector add 的 block tensor 带有类似布局：

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [4],
  threadsPerWarp = [32],
  warpsPerCTA = [4],
  order = [0]
}>
```

可以按下面的思路理解：

- 一个 CTA 使用 4 个 warps；
- 每个 warp 32 个 threads；
- 每个 thread 负责若干连续元素；
- `order` 描述维度访问/分布顺序；
- 合起来覆盖逻辑上的 1024-element block tensor。

不要把 `tensor<1024xf32>` 理解为“某个线程持有 1024 个标量”。在 TTGIR 中，layout encoding 定义了这个逻辑 tensor 如何分散给整个 program instance 的 threads/warps。

### 5.3 常见 layout/encoding

文章列举的 encoding 包括：

| Encoding | 主要用途 |
|---|---|
| `blocked` | 通用 block tensor 在线程/warp 间的分布 |
| `slice` | 从已有布局沿某个维度切片/派生 |
| `dot_op` | matrix multiplication operand 的专用布局 |
| `shared` | shared-memory 中的数据布局 |
| `nvidia_mma` | NVIDIA Tensor Core/MMA 相关布局 |
| `amd_mfma` | AMD Matrix Fused Multiply-Add 布局 |
| `amd_wmma` | AMD WMMA 相关布局 |

这些名字和具体 IR 形式会随 Triton 版本演进。文章还提到 linear layout 的统一方向，因此读新版本 dump 时，应关注语义和数据分布，不要死记某个版本的打印文本。

### 5.4 `num_warps` 与 `num_stages` 不要混淆

```python
kernel[grid](..., BLOCK_SIZE=1024, num_warps=4, num_stages=3)
```

- `num_warps`：一个 program/CTA 使用多少 warps，直接影响 TTGIR 的执行布局；
- `num_stages`：software pipelining 的 stage 数，常影响循环中 load/compute overlap；
- 本文的“编译阶段”是 TTIR、TTGIR、LLVM IR 等 lowering stages。

因此，`num_stages=3` 绝不是说“编译器只有三个 IR 阶段”。这是两个完全不同的 stage 概念。

---

## 6. 阶段三：TTGIR → LLVM IR

### 6.1 抽象层级再次下降

到了 LLVM IR，很多 tile-level 操作已经被展开为更接近 thread-level 的表示：

- 读取 CTA/thread id 的 target intrinsics；
- 计算每个 thread 负责的元素 offset；
- address-space-aware pointer arithmetic；
- scalar/vector load 与 store；
- predicate、branch 和 select；
- 算术操作与 target intrinsics；
- 为 PTX/AMDGCN backend 准备的低层信息。

本地 NVIDIA dump 中可以看到：

```llvm
call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
```

它们分别读取 CTA id 和 thread id。高层的 `tl.program_id(0)` 与 block tensor 分布，到这里已经被落实为硬件执行索引和每线程工作。

### 6.2 LLVM IR 的作用

LLVM IR 是一个成熟的低层优化与代码生成接口。Triton 借助它完成：

- 通用低层优化；
- target intrinsic 表达；
- control flow 和 pointer lowering；
- 后端指令选择前的准备；
- 向 NVPTX 或 AMDGPU backend 移交代码生成。

TTGIR 和 LLVM IR 之间并不是简单的逐行翻译。一个高层 `tt.load` 可能变成多个线程的地址计算、predicate 和若干 load；布局转换、向量化和硬件 intrinsic 也可能改变 IR 结构。

---

## 7. 阶段四：LLVM IR → PTX

### 7.1 PTX 是什么

PTX（Parallel Thread Execution）是 NVIDIA 的虚拟 ISA/汇编层。它已经明确表达：

- kernel entry 与参数；
- virtual registers；
- address spaces；
- thread/block special registers；
- load/store；
- arithmetic、predicate 与 branch；
- 目标 SM 和 PTX ISA 版本。

本地 RTX 4060 生成的 PTX header 包含：

```ptx
.version 8.8
.target sm_89
.address_size 64
```

Vector add 的浮点加法最终可表现为类似：

```ptx
add.f32 destination, source_x, source_y;
```

### 7.2 PTX 不是最终机器码

一个常见误解是“看到了 PTX，就看到了 GPU 真正执行的指令”。更准确的层次是：

```text
PTX                    NVIDIA virtual ISA
  │ ptxas / driver JIT
  ▼
cubin 中的 machine code
  │ disassembler
  ▼
SASS                    面向具体 SM 的机器指令表示
```

PTX 与 SASS 的差异类似于稳定的虚拟指令接口与具体微架构机器指令之间的差异。研究寄存器、真实 memory instructions 或 Tensor Core 指令时，最终还要看 SASS；研究 Triton 生成的 NVIDIA 低层逻辑时，PTX 通常已经非常有帮助。

---

## 8. 阶段五：PTX → cubin

NVIDIA backend 调用 assembler 将 PTX 编译为 cubin。cubin 是可由 CUDA runtime/driver 装载的二进制容器，包含目标架构对应的机器代码和相关元数据。

重要区别：

| 产物 | Python 中的典型类型 | 是否适合 `print`/文本保存 |
|---|---|---|
| TTIR | `str` | 是 |
| TTGIR | `str` | 是 |
| LLVM IR | `str` | 是 |
| PTX | `str` | 是 |
| cubin | `bytes` | 否，应二进制写入 |

文章为了展示阶段，把多种产物写入文件。实际使用时不要用文本方式保存 cubin：

```python
from pathlib import Path

Path("kernel.cubin").write_bytes(compiled.asm["cubin"])
```

若工具链支持，`compiled.asm["sass"]` 可以触发反汇编；它可能依赖 `nvdisasm` 等 NVIDIA 工具，并不一定预先出现在 `asm.keys()` 中。

---

## 9. 同一条语句如何跨阶段演化

以 `x + y` 为例：

| 层级 | 表达重点 | 近似形式 |
|---|---|---|
| Triton source | block tensor 语义 | `output = x + y` |
| TTIR | typed tile operation | `arith.addf` on `tensor<...xf32>` |
| TTGIR | typed tile + layout | `arith.addf` on `tensor<...xf32, #blocked>` |
| LLVM IR | 每线程低层值 | scalar/vector `fadd` |
| PTX | virtual ISA | `add.f32` |
| SASS | 具体 GPU machine instruction | 由目标架构决定 |

再以 `tl.load` 为例：

```text
Triton:  pointer block + mask
TTIR:    tt.addptr + tt.load，仍是逻辑 tensor
TTGIR:   带 layout encoding 的 pointer/value tensor
LLVM IR: thread id + per-thread offset + predicate + address-space load
PTX:     global load/predicate 等虚拟指令
SASS:    具体 SM 的 load 指令
```

关键结论：

> Triton source 中的一条语句不保证对应一条 PTX/SASS 指令；一对多、多对一和跨语句优化都很常见。

---

## 10. 在当前环境中导出全部阶段

下面的代码可以放在 Lab 02 vector add 的 host 侧。必须保存 kernel launch 的返回值，而不是只保存 output tensor：

```python
from pathlib import Path

import torch
import triton

x = torch.randn(1024, device="cuda")
y = torch.randn_like(x)
out = torch.empty_like(x)

compiled = _add_kernel[(1,)](
    x,
    y,
    out,
    x.numel(),
    BLOCK_SIZE=1024,
)
torch.cuda.synchronize()

print(compiled.asm.keys())

dump_dir = Path("triton-dump")
dump_dir.mkdir(exist_ok=True)
for stage in ("source", "ttir", "ttgir", "llir", "ptx"):
    if stage in compiled.asm:
        (dump_dir / f"add_kernel.{stage}").write_text(
            compiled.asm[stage],
            encoding="utf-8",
        )

(dump_dir / "add_kernel.cubin").write_bytes(compiled.asm["cubin"])
```

使用本课程的 uv 环境运行：

```bash
uv run python inspect_compilation.py
```

本地 Triton 3.8.0 实测：

```text
asm keys: source, ttir, ttgir, llir, ptx, cubin

source  str
ttir    str
ttgir   str
llir    str
ptx     str
cubin   bytes
```

博客示例列出 `ttir / ttgir / llir / ptx / cubin`。当前版本额外暴露 `source`：本地观察它是 compiler 初始化后、TTIR pass pipeline 之前的早期 MLIR-like 表示，不是原始 Python 源码字符串。接口和打印格式属于版本相关细节，使用前应先检查 `compiled.asm.keys()`。

### 10.1 推荐的阅读顺序

第一次不要从头到尾逐行读数千行 IR。按问题驱动阅读：

1. 在 TTIR 搜索 `tt.get_program_id`、`tt.make_range`、`tt.load`、`tt.store`；
2. 查看 function signature 中的 pointer dtype 和 divisibility attributes；
3. 在 TTGIR 查看 module target、`num-warps` 和 layout definitions；
4. 对照相同的 `loc(...)` 追踪源码位置；
5. 在 LLVM IR 搜索 `ctaid`、`tid`、`fadd` 和 NVVM intrinsics；
6. 在 PTX 搜索 `.target`、`.entry`、`ld.global`、`st.global`、`add.f32`；
7. 最后比较不同 meta-parameters 生成的 diff。

---

## 11. 多级 IR 为什么必要

如果直接把 Triton Python 翻译成 PTX，会同时面对：

- 高层 tile 算法语义；
- layout 选择；
- thread/warp 分工；
- shared-memory staging；
- target-specific matrix instructions；
- pointer/control-flow lowering；
- 指令选择和最终装配。

把它们拆到多级 IR 后，每一层负责一组相对清晰的问题：

| 层级 | 最适合回答的问题 |
|---|---|
| TTIR | 程序计算了什么 tile-level operation？类型和 shape 是什么？ |
| TTGIR | tile 如何分布到 GPU execution hierarchy？使用什么 layout？ |
| LLVM IR | 每线程如何计算地址和执行低层操作？ |
| PTX/AMDGCN | backend 生成了哪些虚拟 ISA 指令？ |
| cubin/HSACO | 能否被目标 runtime 装载执行？ |

这种 progressive lowering 还有两个工程优势：

1. 高层优化可以复用于不同 GPU backend；
2. target-specific 优化可以集中在 TTGIR 之后的 backend 中。

---

## 12. NVIDIA 与 AMD backend 的分叉

博客主要展示 NVIDIA 路径，但 Triton 的目标是支持多种现代 GPU。当前官方 backend 源码中的主路径可以概括为：

```text
NVIDIA:
AST → TTIR → TTGIR → LLVM IR → PTX → cubin

AMD:
AST → TTIR → TTGIR → LLVM IR → AMDGCN → HSACO
```

共同部分保留 Triton 的 tile 与 GPU layout 语义；后端阶段根据目标选择不同 virtual ISA、assembler 和 binary format。

这解释了为什么 TTIR 不能过早绑定 NVIDIA-only 概念：越晚分叉，越多高层分析与优化能够跨 backend 复用。但 TTGIR 中仍会出现 backend-specific encoding，因为 matrix core、warp/wavefront 和 memory mechanisms 并不完全相同。

---

## 13. 与 FlashAttention kernel 的联系

Vector add 只展示 `blocked` layout；FlashAttention 会让每一层暴露更多值得检查的信息。

### TTIR 重点

- Q/K/V pointer arithmetic 与 mask 是否正确；
- `tl.dot`、`tl.max`、`tl.exp`、`tl.sum` 的 tile shapes；
- causal 分支是否因 `tl.constexpr` 被特化；
- KV loop 是保留、展开还是经过 canonicalization。

### TTGIR 重点

- Q/K/V 和 accumulator 的 layout encoding；
- dot operands 是否转换为专用布局；
- shared-memory allocation/layout conversion；
- `num_warps`、`num_stages` 对数据分布和 pipeline 的影响；
- 是否出现 matrix-core 相关 encoding。

### LLVM IR / PTX / SASS 重点

- global-memory load/store 是否向量化；
- address calculation 是否过多；
- 是否生成预期的 matrix instructions；
- barrier、shared-memory 和 async-copy 相关操作是否出现；
- register usage 与 spill 情况。

注意：源码里写了 `tl.dot`，不代表所有 dtype/shape/target 都必然生成同一种 Tensor Core 指令。最终结果受输入类型、layout、精度选项、尺寸和 GPU 架构共同影响，必须查看低层产物或 profiler。

---

## 14. 常见误解与纠正

### 误解 1：TTIR 完全不知道类型

错误。TTIR 相对机器无关，但 JIT specialization 已把 pointer element type 和 constexpr tile shape 带入 IR。

### 误解 2：TTGIR 中的 tensor 属于一个 thread

错误。带 layout encoding 的逻辑 tensor 被分布到 program instance 内的 threads/warps。

### 误解 3：`num_stages` 表示 IR stage 数

错误。它通常控制 software pipeline stages，与 TTIR/TTGIR/LLVM IR 的编译阶段无关。

### 误解 4：PTX 就是 GPU 最终机器码

错误。PTX 还会被装配/JIT 为具体架构机器码；SASS 更接近 GPU 实际执行指令。

### 误解 5：每条 Triton 语句对应一条 PTX

错误。lowering、layout conversion、向量化、CSE 和 instruction selection 会产生一对多或多对一映射。

### 误解 6：`tl.load` 就是“DRAM 搬到 shared memory”

不严谨。`tl.load` 表达从 pointer block 读取；数据会经过 GPU memory hierarchy，结果可能放在 registers，也可能因具体 lowering/staging 使用 shared memory。是否使用 shared memory 应查看 TTGIR/低层 IR，而不是仅凭 source 判断。

### 误解 7：`compiled.asm["cubin"]` 可以当文本打印

错误。当前环境中 cubin 是 `bytes`，应使用二进制方式保存。

---

## 15. 调试定位：在哪一层找问题

| 症状 | 优先查看 | 理由 |
|---|---|---|
| dtype/shape/DSL 编译错误 | Triton source、TTIR diagnostic | 前端语义或类型阶段就失败 |
| mask、offset、算法结果错误 | Source + TTIR + correctness test | 高层地址/语义错误通常已体现在 TTIR |
| warp/layout conversion 异常 | TTGIR | GPU tensor distribution 在这里显式化 |
| shared memory 超限 | metadata + TTGIR | tile/layout/pipeline 决定片上资源需求 |
| 没有使用 Tensor Core | TTGIR + PTX/SASS | 先查 dot layout，再查最终指令 |
| global load 不合并 | TTGIR + LLVM IR/PTX + profiler | layout 与每线程地址共同决定访问模式 |
| 寄存器过多或 spill | PTX/SASS + ptxas info + Nsight Compute | 最终资源分配接近 backend/assembler 阶段 |
| 首次运行特别慢 | JIT/cache 行为 | 可能测到了编译而不是执行 |

IR 是解释“编译器生成了什么”的证据；Nsight Systems/Compute 是解释“它在硬件上实际表现如何”的证据。两者不能互相替代。

---

## 16. 建议实验

### 实验 A：追踪 Vector Add 的四个操作

导出 TTIR、TTGIR、LLVM IR 和 PTX，分别追踪：

1. program id；
2. range/offset；
3. masked load；
4. floating-point add 与 store。

验收：为每个操作写出各阶段的对应名称，并说明在哪一阶段开始出现 thread id。

### 实验 B：比较 `BLOCK_SIZE`

分别编译：

```text
BLOCK_SIZE = 256, 512, 1024
```

比较：

- TTIR tensor shape；
- TTGIR `sizePerThread`/layout；
- PTX 长度和 load/store 数量；
- kernel latency；
- 是否生成不同 specialization/cache entry。

预期：BLOCK_SIZE 首先改变 TTIR 的静态 tensor shape，随后影响线程分工和资源使用；性能不保证随 block size 单调变化。

### 实验 C：比较 `num_warps`

固定 `BLOCK_SIZE=1024`，分别使用 2、4、8 warps。重点查看 TTGIR module attributes 和 blocked layout，然后 benchmark。

验收：解释“更多 warps”为什么既可能提高并行度，也可能增加资源压力或降低 occupancy。

### 实验 D：检查 Matmul/FlashAttention

对 Lab 02 matmul 或 Lab 03 FlashAttention 导出 TTGIR，搜索：

```text
dot
shared
mma
convert_layout
```

再到 PTX/SASS 查找 matrix instruction。不要只因 source 中存在 `tl.dot` 就断言使用了 Tensor Core。

---

## 17. 面试速记

### Q1：Triton 的完整 NVIDIA 编译链是什么？

```text
Python AST → TTIR → TTGIR → LLVM IR → PTX → cubin
```

如果讨论实际机器指令，可在 cubin 之后补充反汇编得到 SASS。

### Q2：TTIR 与 TTGIR 的关键区别？

TTIR 主要表达 typed tile-level semantics；TTGIR 加入 target、warp/CTA 数量和 tensor layout encoding，说明 tile 如何映射到 GPU execution hierarchy。

### Q3：为什么需要 LLVM IR？

它把 GPU-aware 高层操作降低为 thread-level pointer、control flow、arithmetic 和 target intrinsics，并复用 LLVM 的优化与 target code generation 基础设施。

### Q4：PTX 和 cubin 有什么区别？

PTX 是 NVIDIA virtual ISA 文本；cubin 是面向特定架构、可装载执行的二进制容器。

### Q5：为什么 Triton 第一次调用慢？

第一次新 specialization 需要经过编译、装配和 cache 写入；相同变体的后续调用通常复用 cache。应 warm up 后再计时。

### Q6：为什么 `BLOCK_SIZE` 常写成 `tl.constexpr`？

tile shape、`tl.arange` 范围和许多布局决策必须在编译期已知；constexpr 允许编译器特化、展开和优化对应变体。

### Q7：怎样确认 `tl.dot` 使用了 Tensor Core？

查看 TTGIR 的 dot/matrix-core layout，再检查 PTX/SASS 的实际 matrix instructions，并结合目标 dtype、shape 和架构判断。

---

## 18. 一页总结

```text
Source / AST
  关注：Python DSL、constexpr、JIT specialization
      │
      ▼
TTIR (MLIR)
  关注：typed tile semantics、tt.load/store/dot、静态 shape
      │
      ▼
TTGIR (MLIR)
  关注：target、warps/CTA、layout、shared/dot/mma encoding
      │
      ▼
LLVM IR
  关注：thread id、pointer、predicate、intrinsics、低层优化
      │
      ▼
PTX / AMDGCN
  关注：virtual ISA、load/store、arithmetic、matrix instruction
      │
      ▼
cubin / HSACO
  关注：目标设备可装载的二进制
      │
      ▼
Profiler + SASS
  关注：真实资源使用、访存行为与硬件执行性能
```

最重要的心智模型：

> Triton 编译器不是把 Python 逐行翻译成 GPU 指令，而是在多个 IR 层级中逐步决定“算什么、怎样分块、如何分给 threads/warps、最终使用什么指令”。

---

## 参考资料

- [PyTorch Blog: Triton Kernel Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- [Triton Vector Addition Tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton NVIDIA Backend Compiler](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py)
- [Triton AMD Backend Compiler](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py)
- [Triton Generic Compiler Driver](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py)
- [NVIDIA PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
