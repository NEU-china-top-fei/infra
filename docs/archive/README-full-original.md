# AI Infra Engineering Tutorial 2026

My journey from Backend to AI Infra Engineer.

## 人才画像

这份战略目标书旨在为一名具备扎实系统基础（CMU 研究生级别）及大模型应用经验，但缺乏 AI Infra 工业界实战背书的普通后端工程师，勾勒出蜕变为顶尖大模型公司（如深度求索 DeepSeek）核心系统高级研发工程师的**最终人才画像**。

结合视频中提到的大厂 AI 平台与训练工程师的核心痛点（计算、通信、存储调度优化），以及 DeepSeek 强调的“极致性能榨取”、“软硬件协同”与“专家深度 + 架构广度”，你需要通过自学和深度实践，最终将自己塑造成“具备顶层算法业务感知力，底盘扎根于分布式系统与高性能计算的硬核系统工程师”。

### 第一部分：核心技能领域与具体技能点架构

针对 DeepSeek “专家深度 + 架构视野”的要求，你需要构建 **1 个主攻方向（破局点）+ 2 个辅助方向（护城河）+ 1 个通用底层（基本盘）** 的技能矩阵。由于你拥有传统分布式后端的系统基础，建议以“分布式训练与通信优化”或“大规模集群调度与存储”作为主攻的领域专家切入点。

#### 领域一：大规模分布式训练框架与并行策略（核心主攻 - P0）

*大模型训练的基础设施，直接决定模型迭代的效率。*

- **3D/4D 并行策略底层机制**：深入理解 Data Parallelism (DP/DDP/ZeRO1-3/FSDP)、Tensor Parallelism (TP)、Pipeline Parallelism (PP) 的数学原理与显存/通信开销模型。特别是针对 DeepSeek 核心的 Mixture of Experts (MoE) 架构的 Expert Parallelism (EP)。
- **主流开源框架源码级理解**：`Megatron-LM`、`DeepSpeed`、`vLLM` (推理端)。不能仅仅停留在 API 调用，需要理解其底层的算子切分逻辑、通信原语注入时机。
- **显存优化技术**：Activation Checkpointing (重计算)、CPU Offloading、显存碎片管理（如 PagedAttention 的底层机制）。

#### 领域二：高性能网络通信与拓扑感知（核心主攻 - P0）

*对应视频中提及的 Networking 优化，大模型训练集群最容易成为瓶颈的一环。结合你优秀的系统课背景，这是最好的切入点。*

- **集合通信原语 (Collective Communication)**：彻底弄懂 All-Reduce, All-Gather, Reduce-Scatter, Broadcast 的内部算法实现（如 Ring, Tree, Butterfly），以及它们在不同并行策略中的触发时机和通信量计算。
- **通信与计算重叠 (Overlap)**：掌握通信调度的核心思想（如视频中提到的 ByteScheduler），如何通过 CUDA Stream 和框架层的计算图调度，让前向/反向传播的矩阵乘法与网络通信并行执行。
- **底层网络协议与拓扑**：理解 RDMA (RoCEv2, InfiniBand) 的机制；理解单机内 NVLink/NVSwitch 拓扑，以及机架间胖树 (Fat-Tree) 拓扑对通信路由调度的影响。

#### 领域三：异构计算与算子优化（高阶进阶 - P1）

*DeepSeek JD 强调“榨干硬件点滴性能”，需要深入 GPU 架构。*

- **CUDA 编程模型基础**：GPU 内存层次结构（Global, Shared, Registers）、Thread/Block/Warp 调度机制、Memory Coalescing（内存合并访问）、Bank Conflict。
- **AI 编译器基础**：了解 Triton 语言及其编译机制。Triton 是目前高性价比改写算子的利器，能够编写媲美手写 CUDA 的算子（如 FlashAttention 的 Triton 实现）。
- **Profile 与性能分析**：熟练使用 Nsight Systems (`nsys`)、Nsight Compute (`ncu`)、PyTorch Profiler，能够从底层视角看清系统 Timeline 上的空白 (Idle) 并在代码层定位原因。

#### 领域四：大规模集群调度与容错保障（工程基础 - P1）

*对应视频中的训练平台基本要素，解决数百台机器协同的工程稳定性问题。*

- **多租户调度与资源分配**：理解 Kubernetes, Slurm 等调度系统的核心机制。理解如何针对视频提到的 JCT (Job Completion Time) 和 Makespan 指标进行拓扑感知的任务调度。
- **容错与弹性 (Fault Tolerance)**：大模型训练必然遇到硬件故障。掌握同步训练下的快速 Checkpoint/Restore 机制、异步保存、以及检测到硬件掉线后的自动容灾漂移设计。
- **高吞吐数据流**：应对 Dataloader 在数百张卡上的读取瓶颈，理解并行文件系统，以及从数据预处理到显存的 Zero-copy 数据流。

#### 领域五：底层系统编程与代码品味（基本盘 - P0）

*DeepSeek 基本要求第一条。*

- **Modern C++ (14/17/20)**：大模型 Infra 底层（如 PyTorch C++ 拓展、自定义算子、通信库）的必备语言。要求极高的内存安全意识和极致的性能嗅觉。
- **Python 内部机制与 C/C++ 互操作**：理解 GIL，CPython 内存管理，Pybind11，能够自如地在 Python 框架层与 C++ 性能层穿梭。
- **无锁编程与并发控制**：操作系统级别的线程调度、原子操作、内存屏障。

### 第二部分：各技能领域的学习优先级与战略定位

| 优先级 | 技能领域 | 战略定位与自学重点 |
| :--- | :--- | :--- |
| **P0 (生存基石)** | 分布式框架与并行策略 | **必考题**。自学策略：从手推各种并行的显存占用和通信量公式开始，随后精读 Megatron-LM 源码。利用你的 Agent 背景，思考大模型每一层的输入输出在多卡间如何切分。 |
| **P0 (决胜长板)** | 高性能网络通信 | **利用 CMU 系统课背景实现降维打击**。自学策略：深入研究 NCCL 原理与 RDMA 网络。这是很多纯算法/纯后端工程师的盲区，如果你能清晰阐述如何根据网络拓扑做调度，将极大增加说服力。 |
| **P0 (代码品味)** | 底层系统编程 | **一票否决项**。大厂极为看重代码质量，尤其是 DeepSeek。自学策略：用 C++ 重写一些极简版的轮子，培养对纳秒级延迟和字节级内存的敏感度。 |
| **P1 (潜力展现)** | 异构计算与算子优化 | **证明“榨干性能”潜力的得分点**。不需要成为写 PTX 汇编的极客，但必须掌握 Triton 编写常见算子（如 Fused LayerNorm, Attention）。自学策略：啃透 FlashAttention 原理并复现其简化版。 |
| **P1 (工程视野)** | 集群调度与容错存储 | **体现全局架构观**。自学策略：结合传统后端高可用架构经验，研究大模型场景下（状态极重、同步阻塞）的容错机制有何不同，提出针对性解决方案。 |

### 第三部分：面试场景画像（7 轮面试的实战拆解与能力预期）

在没有任何顶会论文和顶级开源项目背书的情况下，面试官对你的初始假设是“懂传统系统的熟练工”。你需要通过这七轮面试，展现出“理论吃透、能推公式、能看懂源码、有实操洞察”的硬核实力。

#### 1. 编码能力与系统素养局（通常为前 2 轮）

- **考察重点**：并非单纯的 LeetCode。会考察带有系统背景的编程，例如：实现一个多线程安全的高性能内存池；实现一个无锁队列；或者写一个类似 Ring All-Reduce 的拓扑模拟代码。
- **你需要达到的水平**：C++ 代码不仅要 Bug-free，还要展现出对缓存行失效 (Cache Line Bouncing)、内存对齐、零拷贝 (Zero-copy) 的深刻理解。这证明你的 JD 基本要求：“优秀的设计能力和代码品味”。

#### 2. AI 系统架构与计算推演局（通常为第 3-4 轮）

- **考察重点**：纸上谈兵的深度。面试官会给出具体场景：“现在要训练一个千亿参数 MoE 模型，集群有 1024 张 H100，网络是两层胖树，请设计并行策略，并估算每一步的通信时间和显存峰值。”
- **你需要达到的水平**：能够熟练地在白板上（或共享屏幕）写出模型状态（Weights, Gradients, Optimizer states）的显存占用公式。能够准确计算在给定 TP=8, PP=4, DP=32 下，每一次 Forward 和 Backward 阶段网络上需要传输多少 Byte 的数据，耗时大概多少，受限于算力还是受限于带宽。**这是体现你虽然没做过，但已经把理论吃得极透的黄金时刻。**

#### 3. 领域深度与疑难杂症局（通常为第 5-6 轮，资深专家面）

- **考察重点**：排错能力和极端优化（JD 中的“榨干硬件”）。面试官会抛出生产环境的真实问题：“发现训练中途某个 step GPU 利用率突然掉到 0 持续了几百毫秒，你怎么排查？”或者“现有的通信计算重叠策略在某一层失效了，为什么？”
- **你需要达到的水平**：展现出工具链的熟练度（如回答通过 `nsys` 抓取 timeline，查看是 CPU dataloader 阻塞，还是某个通信原语未启动，或是 D2D 内存拷贝造成的 stall）。如果你在自学时用几张消费级显卡或租用云端多卡做过真实的 Profiling 和瓶颈分析，这里的回答会非常具有实战画面感。

#### 4. 技术视野与主管局（第 7 轮）

- **考察重点**：自我驱动力、快速学习能力以及对 AGI 的认知。会挑战你的劣势：“你之前做 Agent 应用层，为什么转到底层 Infra？你觉得自己能适应吗？”
- **你需要达到的水平**：巧妙转化劣势。你的逻辑应该是：“正因为我深度做过 Agent，我知道 LLM 落地时 Context Window 不断扩大对 KV Cache 和推理延迟造成的灾难性影响，也知道 MoE 对于应用的巨大价值。这让我意识到 AGI 的瓶颈已经完全转移到了底层系统优化上。我凭借 CMU 扎实的 OS 和网络基础，在过去几个月里吃透了 Megatron 的源码并精通了 Triton 编程，我具备从业务痛点 (Top-down) 直达硬件底层 (Bottom-up) 的完整视野，这比纯粹做底层的工程师具有更好的目标感，我完全能在 1-2 个月内补齐业务实操的拼图。”

### 结语与执行建议

这幅人才画像的门槛极高，但对于拥有扎实 CS 基础的人来说并非不可逾越。由于你缺乏背书，**你的“投名状”不应该是简历上的空话，而必须是硬核的产出**。

**三个月的破局建议：**
不要泛泛而读。挑一个特定的痛点（例如：优化某种特定的 Transformer 变体的通信开销，或者用 Triton 重新写一个融合算子），在 Github 上开源你的分析过程、公式推导、Timeline 截图和对比代码。把这个硬核的分析报告挂在简历显眼处。当你能在面试中指着自己的分析图表与 DeepSeek 的专家探讨时，你就不再是一个“想转行的后端工程师”，而是一个“带着诚意和实力的准入职者”。

---

## 培养方案

随着大语言模型 (LLM) 与多模态生成式人工智能的参数量突破万亿级别，以及计算集群规模从千卡向十万卡（如 100K+ GPU 集群）迈进，人工智能技术的发展瓶颈已从纯粹的算法架构设计，彻底转移到了底层计算、通信与存储的系统级工程上。在这一历史性拐点，AI 基础设施核心系统工程师 (AI Infrastructure Core Systems Engineer) 成为了决定大模型厂商商业护城河与生死存亡的关键角色。该角色要求工程师不仅具备深厚的 C++ 与 CUDA 底层开发能力，还必须在分布式并行计算、高性能网络 (RDMA/RoCEv2)、显存管理优化以及大规模集群容错调度等维度具备极高的全局架构视野。

传统的软件工程培养路径已无法满足当今万卡集群对极致性能的压榨需求。本方案基于卡内基梅隆大学 (CMU) 系统方向的硬核培养逻辑（汲取 15-418 Parallel Computer Architecture and Programming、15-712 Advanced Operating Systems and Distributed Systems 等核心课程精髓），专为具备一定后端开发或基础架构经验的全职专业人士设计，制定为期一年的业余时间高强度转型路径。方案严格划分为“课程之间的全局战略规划”与“课程内部的微观源码深度剖析”两个维度，并最终将所有知识体系收敛至硅谷顶级科技公司及明星 AI 初创企业的七轮硬核系统面试矩阵中。

### 课程之间的全局战略规划与调度框架

在大规模 AI 基础设施领域，系统组件并非孤立存在，而是一个高度耦合的复杂工程集合。例如，分布式训练中的张量并行 (Tensor Parallelism) 策略直接决定了单节点内 NVLink 的带宽需求，而流水线并行 (Pipeline Parallelism) 则对跨节点 InfiniBand 或 RoCEv2 网络的拓扑结构提出了严苛要求。因此，培养方案必须建立严格的前置依赖条件与并发学习时间轴。本方案定义了五门虚拟核心课程，按重要程度与底层逻辑分为三个阶段：底层算力基石、横向扩展与通信拓扑、以及集群调度与极致推理。

| Course Code | Course Title | Core Engineering Domain | Priority | Dependency |
| :--- | :--- | :--- | :--- | :--- |
| **SYS-601** | GPU Architecture and Operator Optimization | Single-node compute, C++/CUDA, Triton, Memory hierarchy | 1 (Critical) | None |
| **SYS-602** | Distributed Training and Hybrid Parallelism | 3D Parallelism, MoE routing, Memory optimization (ZeRO) | 2 (Critical) | SYS-601 |
| **SYS-603** | High-Performance AI Networking and Collectives | RDMA, RoCEv2, NCCL algorithms, Congestion control | 3 (High) | SYS-601 |
| **SYS-604** | High-Throughput LLM Inference Systems | KV Cache management, Continuous batching, PagedAttention | 4 (High) | SYS-601, SYS-602 |
| **SYS-605** | Large-Scale Cluster Scheduling and Fault Tolerance | Gang scheduling, Checkpointing, Automated failure recovery | 5 (Medium) | SYS-602, SYS-603 |

系统级知识的吸收需要遵循科学的认知路径，上述课程的执行并非完全串行，而是要求在特定阶段进行交替同步学习，以建立跨栈 (Cross-Stack) 的系统直觉。

- **前三个月属于绝对串行期，主攻 SYS-601。** 一切分布式架构的基础在于对单卡算力的极致压榨。在未深刻理解 GPU 内存层次结构（包括高带宽内存 HBM、SRAM/Shared Memory、寄存器 Registers）、Warp 调度机制以及张量核心 (Tensor Cores) 的运作原理之前，研究复杂的分布式系统将沦为纸上谈兵。此阶段需完全沉浸于系统级 C++ 与底层 GPU 编程的思维转换中。
- **第四至第七个月进入高强度的并发交替期，要求同步推进 SYS-602 与 SYS-603。** 分布式训练策略与底层网络通信是“软硬协同设计 (Hardware-Software Co-design)”的经典体现。当剖析 Megatron-LM 的张量并行源码时，必须同步研究 NCCL 的 AllReduce 底层实现与环形/树形拓扑构建。当学习流水线并行与 DeepSeek 最新披露的 DualPipe 机制时，需要结合理解 RDMA 网络的拥塞控制、优先流量控制 (PFC) 导致的死锁风险以及端到端通信延迟。
- **第八至第十个月属于应用深化期，核心聚焦于 SYS-604。** 在大模型全面迈向商业化落地的阶段，推理引擎的运行成本直接决定了企业的毛利率。在掌握了前置的算子优化与模型架构后，研究重点需从训练期的“吞吐量极大化”向推理期的“首字延迟 (TTFT) 与字间延迟 (ITL) 的平衡”转移。必须深刻理解 PagedAttention 机制如何跨界借鉴传统操作系统的虚拟内存与分页机制。
- **最后两个月进入全局架构与兜底保障期，主攻 SYS-605。** 当集群规模扩展至万卡甚至十万卡时，硬件节点的 MTBF（平均故障间隔时间）急剧缩短。此时的学习焦点全面转向宏观的集群作业编排（如 Gang Scheduling）、高频 Checkpointing 引发的存储 I/O 瓶颈突破，以及基于分布式内存的快速故障恢复机制。

### 课程内部源码与文献深度研究计划

针对上述五门核心课程，每一门都必须遵循理论与极致工程并重的原则。方案强制要求工程师深入探究特定顶级开源项目中最核心的代码路径，研读奠基性与前沿性学术论文，并吸收业界顶尖工程师的实战经验总结。

#### SYS-601: GPU Architecture and Operator Optimization

本课程的核心命题是打破深度学习领域的“内存墙 (Memory Wall)”。随着 Transformer 架构的扩张，模型算力需求的增长速度已远远超过了 GPU 物理显存带宽的增长速度，导致标准 Attention 等机制被严重限制在内存带宽瓶颈 (Memory-Bound) 上。算子融合 (Operator Fusion) 与极致的显存局部性优化成为基础设施工程师必须掌握的核心技能。

在源码级研究层面，本课程将剖析 `openai/triton` 编译器项目。研究计划要求绕过表层的 API 调用，深入探究 Triton 编译器如何将高阶的 Python 抽象语法树 (AST) 转换为多级中间表示 (MLIR)，再逐步 Lowering 到 LLVM IR，并最终生成底层的 PTX 汇编代码的完整编译管线。工程实践要求逐行剖析官方库中的 `triton/python/tutorials/06-fused-attention.py` 文件，深刻理解 Triton 特有的 Block 级别内存编程模型如何替代传统且极易出错的 CUDA 线程级编程。

分析 FlashAttention 系列的理论演进可以获得极深的第一性原理洞察。标准 Attention 机制在计算 Query 和 Key 的点积时，需要产生一个空间复杂度为 O(N^2) 的中间激活矩阵，该矩阵必须写入 HBM 后再读出。这造成了极大的硬件闲置。FlashAttention-1 通过创新的 SRAM 分块计算 (Tiling) 和在线 Softmax，将读写复杂度降为线性。FlashAttention-2 重新划分了 Thread Block 与 Warp 的工作负载。更前沿的 FlashAttention-3 则极致利用了 NVIDIA Hopper 架构的 TMA 和 WGMMA 指令，实现了计算与数据搬运的深度异步重叠。

| 必读物 (Required Reading for SYS-601) | 类别 | 核心关注点 |
| :--- | :--- | :--- |
| Triton: an intermediate language and compiler for tiled neural network computations (Tillet et al., 2019) | Academic Paper | MLIR, GPU Compiler, Tiling |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022) | Academic Paper | IO-Awareness, Memory Hierarchy |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023) | Academic Paper | Work Partitioning, Warp Execution |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024) | Academic Paper | Asynchrony, Tensor Cores, FP8 |
| Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs (2025) | Academic Paper | Operator Fusion, Tensor Compilers |
| OpenAI Triton 1.0 Release Blog | Engineering Blog | Triton Origin, Compiler Design |
| Introduction to GPU Programming with Triton | Tutorial | GPU Basics, Warps, SMs |
| How I Wrote FlashAttention-2 from Scratch in Custom Triton Kernels | Technical Blog | Kernel Implementation, Online Softmax |
| Triton Kernel Compilation Stages | Technical Blog | AST to PTX, LLVM IR |
| Warp Specialization in Triton: Design and Roadmap | Engineering Blog | Asynchronous execution, Megakernels |
| Building High-Performance AI/ML Pipelines with C++ and CUDA | Tutorial | C++ Optimization, CUDA Streams |
| Understanding Flash Attention: Writing the algorithm from scratch in Triton | Tutorial | Block-sparse attention, Tiling |
| 10 C++ Concepts Every AI/ML Engineer Must Master in 2026 | Engineering Blog | Memory Management, Smart Pointers |
| Fear and Loathing in Lock-Free Programming | Technical Blog | Lock-free structures, Atomics |
| ZeroIPC: Transforming Shared Memory into an Active Computational Substrate | Technical Blog | Zero-copy memory, IPC |

#### SYS-602: Distributed Training and Hybrid Parallelism

由于单卡显存容量的物理极限，大语言模型必须被科学地拆解并分布到成百上千张 GPU 上。本课程致力于研究如何利用多维度的混合并行策略 (Hybrid Parallelism)，在跨节点通信开销与单卡计算效率之间寻找最优的纳什均衡。

源码级研究计划将深度剖析 `NVIDIA/Megatron-LM` 框架。工程师需要进入 `megatron/core/tensor_parallel` 和 `megatron/core/pipeline_parallel` 核心目录，解构张量并行中列切分与行切分的组合艺术。必须通过代码证明，前向传播和反向传播中的自定义算子是如何在不需要频繁通信的情况下，维持分布式矩阵乘法的数学等价性的。同时，需追踪经典的 1F1B (One Forward One Backward) 以及交错式 1F1B 调度器在流水线并行中的源码流转，理解其如何有效压缩流水线气泡 (Pipeline Bubble)。

大模型的分布式训练本质上是一场关于显存容量与网络通信带宽的极限博弈。基础的数据并行 (DP) 会导致严重的显存溢出；ZeRO 优化框架通过分片平摊了显存压力，但代价是引入了极其庞大的 AllGather 通信开销。近期 DeepSeek-V3 技术报告中披露的 DualPipe 算法，向业界展示了软硬协同调度的巅峰造诣。在采用 MoE 架构并依赖跨节点专家并行通信时，DualPipe 通过创新的双向流水线并行调度，实现了前向计算、反向计算与跨节点 RDMA 通信的完美重叠。

| 必读物 (Required Reading for SYS-602) | 类别 | 核心关注点 |
| :--- | :--- | :--- |
| Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (Shoeybi et al., 2019) | Academic Paper | Tensor Parallelism, 1F1B Scheduling |
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020) | Academic Paper | Memory Sharding, DP Optimization |
| Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning (Zheng et al., 2022) | Academic Paper | Auto-Parallelism, Compiler |
| Distributed Hybrid Parallelism for Large Language Models: Comparative Study and System Design Guide (Amer et al., 2026) | Academic Paper | Strategy Selection, 3D Parallelism |
| DeepSeek-V3 Technical Report (DeepSeek-AI, 2024) | Academic Paper | DualPipe, HAI-LLM, FP8 Training |
| Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B | Engineering Blog | Scaling Laws, Trillion Parameters |
| Megatron-LM: How Model Parallelism is Pushing Language Models to New Heights | Technical Blog | Intra-Layer Parallelism, NLP |
| DeepSpeed ZeRO Tutorial | Tutorial | ZeRO Stage 1/2/3 Configuration |
| Training 175B Parameter Language Models at 1000 GPU scale with Alpa and Ray | Engineering Blog | Ray Integration, Performance |
| Megatron Bridge Documentation & Parallelisms Guide | Architecture Guide | Distributed Optimizer, DDP vs TP |
| DeepSeek-V3 Technical Report Break Down: DualPipe & FP8 | Technical Blog | Architecture Deep Dive, MoE |
| Day 4 of DeepSeek's Open Source Week: From DualPipe to EPLB | Technical Blog | Overlap computation-communication |
| Memory-Efficient Training on Gaudi with DeepSpeed | Engineering Blog | Hardware Accelerators, ZeRO |
| How Meta Optimized Llama 3 Pretraining | Technical Blog | Meta Infrastructure, MFU |
| Distributed Training of LLMs: A Comparative Study and System Design | Research Review | Paradigm Shifts, Trade-offs |

#### SYS-603: High-Performance AI Networking and Collectives

在动辄调动数万张 GPU 的现代集群中，网络系统不再是简单的外围数据传输组件，而是成为了整个分布式 AI 巨型计算机的“内部总线”。在高达数千 Gbps 的吞吐量要求面前，传统 TCP/IP 协议栈显得极其笨重。RDMA 与深层定制的集合通信库是高级系统工程师必须攻克的深水区。

本课程的源码级研究计划将深入拆解 `NVIDIA/nccl` 的内部架构。工程师需要追踪其在初始化拓扑探测阶段，如何动态构建高效的环形 (Ring) 和双树形 (Double-Tree) 算法拓扑。通过研读 `src/collectives/` 下的源码，探究 NCCL 如何将超大型的集合操作数据包切分为细粒度的多个 Chunk，分配给不同的逻辑通道，从而利用精密的 Pipeline 机制实现网络传输与 GPU 计算的重叠并行。

深入理解通信协议的底层演进，才能看清去中心化异构计算架构的未来。尽管 InfiniBand 提供了原生的高质量网络结构，但 RoCEv2 (RDMA over Converged Ethernet) 凭借更加开放的生态，通过引入 PFC（优先流量控制）和 ECMP 等拥塞控制技术，成功支撑了数万卡大模型训练。更前沿的架构重构（如 NCCLX 或 ICCL）已开始将 P2P 通信调度逻辑从 GPU Kernel 中剥离，转而卸载至 CPU 专用线程中执行，实现 SM-free 的零资源占用传输。

| 必读物 (Required Reading for SYS-603) | 类别 | 核心关注点 |
| :--- | :--- | :--- |
| RDMA over Commodity Ethernet at Scale (Guo et al., 2016) | Academic Paper | RoCEv2, PFC, Deadlock avoidance |
| OmniReduce: Efficient Sparse Collective Communication and its application to Accelerate Distributed Deep Learning (Fei et al., 2021) | Academic Paper | Sparse Collectives, Streaming |
| SwitchML: Hardware-Accelerated Distributed Machine Learning (Sapio et al., 2021) | Academic Paper | In-network computing, Switch aggregation |
| Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms (2025) | Academic Paper | NCCL Architecture, Ring/Tree topolgy |
| NCCLX: Scalable, High-Performance Collective Communication for 100k+ GPUs (Zeng et al., 2025) | Academic Paper | Mega-cluster scaling, SM-free transport |
| Unpacking NCCL: A Deep Dive into Multi-GPU Communication | Technical Blog | Channels, Buffer slots, Pipelining |
| Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication | Engineering Blog | Cost models, Algorithm selection |
| Fast Multi-GPU collectives with NCCL | Tutorial | GPUDirect P2P, Broadcast/Reduce |
| NCCL Deep Dive: Cross Data Center Communication and Network Topology Awareness | Engineering Blog | Inter-DC routing, Fabric IDs |
| RDMA over Ethernet for Distributed AI Training at Meta Scale | Engineering Blog | RoCEv2 at Meta, ECMP routing |
| Zettascale OSU NCCL Benchmark on H100 AI Workloads | Technical Blog | TCP tax, Kernel bypass, Latency |
| Enabling Fast Inference and Resilient Training with NCCL 2.27 | Release Notes | Symmetric memory, Latency kernels |
| Understanding RoCEv2: A Beginner's Guide to RDMA over Converged Ethernet | Tutorial | L3 networking, Protocol configuration |
| The Battle of AI Networking: Ethernet vs InfiniBand | Industry Analysis | Hardware trade-offs, Lossless fabrics |
| Enhancing Communication Observability of AI Workloads with NCCL Inspector | Engineering Blog | Profiling, Network anomalies |

#### SYS-604: High-Throughput LLM Inference Systems

如果说训练系统决定了人工智能大模型的智商下限，那么推理引擎的工程架构则直接决定了 AI 企业商业化变现的成本上限。大语言模型基于自回归生成特性，使其长期处于极度的内存带宽受限 (Memory Bandwidth Bound) 状态。如何在毫秒级延迟内实现显存利用率的极致压榨，是本课程的核心系统命题。

在源码实战环节，工程师将深入探究开源项目 `vllm-project/vllm`。聚焦解析 `vllm/core/scheduler.py` 以及 `vllm/core/block_manager.py` 中的核心调度逻辑。系统梳理一次推理请求的完整生命周期，探究系统如何打破传统的静态批处理限制，实现动态的连续批处理 (Continuous Batching)。深入对比研读 `csrc/attention/attention_kernels.cu` 中的 PagedAttention 实现，彻底理解 GPU Kernel 层级如何通过查询非连续的内存页表来抓取并计算 KV Cache 数据。

传统大模型推理架构面临的致命困境在于 KV Cache 的动态且不可预知的膨胀。vLLM 团队提出的 PagedAttention 算法跨界借鉴了现代操作系统的虚拟内存与物理内存分页机制，将庞大的 KV Cache 切分为极小且固定大小的逻辑 Blocks。碎片化的消除直接使得系统的并发批处理大小得以成倍提升。高阶工程师还需深入思考投机解码 (Speculative Decoding) 机制与多级显存调度带来的系统级连锁反应。

| 必读物 (Required Reading for SYS-604) | 类别 | 核心关注点 |
| :--- | :--- | :--- |
| Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023) | Academic Paper | PagedAttention, KV Cache, Virtual Memory |
| FlashInfer: Customizable and Efficient Attention Engine for LLM Serving (Ye et al., 2024) | Academic Paper | Attention Engine, JIT, SGLang |
| Fast Speculative Decoding for vLLM (Snowflake AI Research, 2024) | Academic Paper | Speculative Decoding, Latency Optimization |
| Achieving Platform Portability for vLLM by using Triton Autotuning (IBM Research, 2024) | Academic Paper | Kernel Portability, Triton Autotuning |
| DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (Dai et al., 2024) | Academic Paper | Expert Routing, Sparse Activation |
| vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention | Engineering Blog | Throughput gains, Memory sharing |
| Explaining the source code behind the vLLM fast inference engine | Technical Blog | Source code logic, AsyncLLMEngine |
| Code Review: Deep Dive into vLLM's Architecture and Implementation | Technical Blog | API Server, OpenAI Compatibility |
| The Architecture Behind vLLM: How PagedAttention Improves Memory Utilization | Engineering Blog | Pre-allocation limits, Block management |
| How Prompt Caching Works in vLLM | Technical Blog | Prefix hashing, Radix Attention trees |
| Ultimate Guide to vLLM | Tutorial | Deployment strategies, Workload flexibility |
| vLLM 2024 Wrapped & 2025 Vision | Release Strategy | Community adoption, Future roadmaps |
| Why vLLM is the best choice for AI inference today | Industry Insight | Sustainable deployment, Kubernetes integration |
| Serving LLMs with vLLM: Practical Guide | Tutorial | Multi-GPU inference, Neural net basics |
| Mastering LLM Techniques: Inference Optimization | Technical Blog | Batch size scaling, NVIDIA optimizations |

#### SYS-605: Large-Scale Cluster Scheduling and Fault Tolerance

当 AI 计算集群跨入万卡乃至十万卡级别时，单一组件的性能优化红利将被系统级的可靠性 (Reliability) 瓶颈彻底吞噬。在十万卡级别的超长周期训练作业中，硬件故障几乎每天都会发生。具备自愈能力的容错调度 (Fault Tolerance) 架构构成了 AI 基础设施的最后一道防线。

工程师将深入研究云原生批量调度框架 `volcano-sh/volcano`。探究其如何在 Kubernetes 上实现面向 AI/HPC 工作负载的群组调度算法 (Gang Scheduling)，提供“全有或全无 (All-or-Nothing)”的严格调度保障，杜绝因部分节点启动引发的死锁与资源浪费。从宏观演进视角看，以 Kubernetes 为底座，深度融合 Volcano 与 Ray（形成 KubeRay 架构）的云原生技术栈，正成为行业标配。

分布式模型训练是一个要求极度同步的过程，高频的持久化存储引发的密集 I/O 风暴严重拖垮了 GPU 计算时间。前沿系统如 Gemini 利用集群内闲置的 CPU 内存作为高速 Checkpointing 缓冲介质，大幅缩短恢复时间。更为颠覆性的 Oobleck 项目倡导内建弹性设计，发生严重故障时，系统无需全局回滚，而是智能调度多副本模型状态并实时切换全新的流水线配置，平滑推进训练。

| 必读物 (Required Reading for SYS-605) | 类别 | 核心关注点 |
| :--- | :--- | :--- |
| Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates (Jang et al., 2023) | Academic Paper | Pipeline Templates, Fast Recovery |
| ByteCheckpoint: An Industrial-Grade Checkpointing System for Large-Scale LFM Training (Wan et al., 2025) | Academic Paper | I/O Bottlenecks, State Management |
| Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints (Zhuang et al., 2023) | Academic Paper | CPU Memory buffers, Traffic interference |
| Reliability of AI Supercomputer Clusters (Kokolis et al., Meta FAIR, 2024) | Academic Paper | Failure Taxonomy, Fleet Data Analysis |
| Deadline-Aware Flow Scheduling for AI Clusters with Heterogeneous Latency Requirements (2024) | Academic Paper | QoS, Dynamic network latency |
| Fault-tolerant training: How we build reliable clusters for distributed AI workloads | Engineering Blog | Checkpoint frequency, Hardware MTBF |
| Storage Requirements for AI Clusters: The Hidden Cost of Checkpointing | Technical Blog | Parallel filesystems, Capacity planning |
| Slurm for ML | Industry Insight | HPC vs Cloud-Native trade-offs |
| Volcano: Collision Between Containers and Batch Computing | Engineering Blog | Kubernetes integration, Gang Scheduling |
| Uber's Journey to Ray on Kubernetes | Engineering Case |  |
| Ray vs Kubernetes for AI training scheduling comparison | Technical Blog | Label-based placement, Orchestration |
| Why Scheduling Will Define AI Infrastructure Efficiency in 2026 | Market Analysis | Resource fragmentation, Idle time limits |
| AI Infrastructure Evolution: From Compute Expansion to Efficient Orchestration | Academic Review | Centralized routing, Traffic planning |
| KubeRay vs Ray Clusters on Cloud VMs | Architecture Guide | VM overhead, Pod-level limits |
| Understanding Slurm for AI/ML Workloads | Tutorial | Root controllers, Job limits |

### 结合硅谷顶尖科技公司七轮硬核面试的战略收口体系

历经高强度系统级深潜后，工程师的知识体系必须完全对标当前硅谷顶级人工智能公司（如 NVIDIA, Meta, OpenAI, DeepSeek, Anthropic）长达七轮背靠背的系统面试考核 (Gauntlet)。

#### Round 1: Low-Level Systems Coding (C++/CUDA & Concurrency)

**战略考核映射：SYS-601**

本轮要求候选人在白板中熟练编写无锁并发数据结构 (Lock-free Data Structures)，深刻解释如何通过原子操作 (Atomics) 消除 Mutex 的开销。清晰辨析并应用 `std::memory_order_acquire` 等细粒度内存顺序语义，防范乱序执行隐患。探讨如何在 C++ 服务端与底层网络之间实现零拷贝。也可能被要求当场使用 Triton 或裸写 CUDA C++，手撸基于共享内存优化的矩阵乘法。

#### Round 2: GPU Architecture and Kernel Design

**战略考核映射：SYS-601, SYS-604**

面试官会选取核心算子要求推导其时间与空间复杂度，并分析物理硬件限制。绝对重心落在 FlashAttention 算法核心思想的白板推演能力：为何采用分块计算 (Tiling)、如何利用非对称显存层级、如何处理 Warp 级并行计算与内存交互调度。高阶候选人必须展现对算子融合 (Operator Fusion) 及在 FP8 极低精度量化格式下维持高指令流利用率的深刻理解。

#### Round 3: Distributed Training Architecture

**战略考核映射：SYS-602**

通常以“设计方案训练 1 万亿参数语言模型”为引子。考验数学计算的精确度，候选人必须在白板上流畅展示如何将数据并行 (DP)、张量模型并行 (TP) 以及流水线并行 (PP) 进行多维度组合堆叠。核心难点在于底层通信流量的数学推演，定量计算不同网络拓扑条件下的并行切分最优设定。探讨 DeepSeek DualPipe 等前沿调度策略是关键加分项。

#### Round 4: High-Performance Networking (Collectives & RDMA)

**战略考核映射：SYS-603**

解构以 NCCL 为代表的高性能集合通信库的运行机理。分析 Ring AllReduce 与 Tree AllReduce 在端到端延迟与极限带宽利用率上的数学权衡。论述现代 AI 基建拥抱 RoCEv2 或原生 InfiniBand 的 RDMA 技术以实现内核旁路 (Kernel Bypass) 的必然性。需应对生产级压力测试：如缓解 ECMP 哈希流碰撞引发的致命长尾延迟，或详细阐述由 PFC 引发的死锁现象及规避手段。

#### Round 5: High-Throughput Inference System Design

**战略考核映射：SYS-604**

业务场景锚定为从零设计兼容 OpenAI API 的生产级大模型服务系统。必须在白板上深度剖析 KV Cache 的动态内存膨胀灾难，推演 PagedAttention 架构如何管理非连续物理显存块，并与操作系统页表进行异同对比。针对推理侧瓶颈提出架构级解法：实现跨请求的前缀缓存 (Prefix Caching)、通过连续批处理调度动态插入新请求，以及在 SLA 约束下平衡 TTFT 和 ITL。

#### Round 6: Resilience, Scheduling, and Orchestration

**战略考核映射：SYS-605**

考察“规模效应带来的系统性失效”。面对上万节点极高宕机率的前提，探讨如何通过在 Kubernetes 之上整合 Volcano 实施群组调度 (Gang Scheduling) 防范死锁。设计多层级模型状态 Checkpointing 系统以低损耗实现高频快照。设计在不触发全局强同步回滚的前提下，通过冗余流水线模板实现故障发生时的平滑自愈与动态恢复 (Pipeline Recovery)。

#### Round 7: Scalable AI Mega-Cluster Design (Capstone)

**战略考核映射：五大核心模块的全局综合**

终极系统架构回合：“设计一个包含 10 万台 H100 节点的集群，支撑万亿参数模型预训练与高并发推理”。候选人必须如同总架构师，横跨整个技术栈：从物理数据中心组网拓扑选择、并行策略对节点局部性的约束，到海量数据并行文件系统的极限 I/O 预估与底层负载监控系统搭建。给出算力成本经济学核算方案，精确推演模型浮点运算利用率 (MFU)。

### 参考文献

- [Revisiting Reliability in Large-Scale Machine Learning Research Clusters - arXiv](https://arxiv.org/html/2410.21680v2)
- [Smart Networking Key For Optimum ROI in AI Data Centers - Counterpoint Research](https://counterpointresearch.com/en/insights/smart-networking-key-for-optimum-roi-in-ai-data-centers)
- [Collective Communication for 100k+ GPUs - arXiv](https://arxiv.org/html/2510.20171v3)
- [AI Infrastructure & ML Systems Engineering Roadmap | by Naresh Kumar - Medium](https://medium.com/@nareshns2004/ai-infrastructure-engineering-complete-36-month-mastery-path-c4ef52ffce3b)
- [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms - arXiv](https://arxiv.org/html/2507.04786v1)
- [CMU 15-418/Stanford CS149: Parallel Computing - csdiy.wiki](https://csdiy.wiki/en/%E5%B9%B6%E8%A1%8C%E4%B8%8E%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F/CS149/)
- [CMU 15712 Advanced Operating Systems and Distributed Systems Course Review](https://fanpu.io/blog/2023/advanced-operating-systems-course-review/)
- [Parallel Computer Architecture and Programming - CMU](https://www.csd.cmu.edu/course/15418/f25)
- [Scaling Language Model Training to a Trillion Parameters Using Megatron - NVIDIA Technical Blog](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)
- [RDMA over Ethernet for Distributed AI Training at Meta Scale - Stanford Computer Science](https://cs.stanford.edu/~keithw/sigcomm2024/sigcomm24-final246-acmpaginated.pdf)
- [Introduction to GPU Programming with Triton | by Katherine Oluwadarasimi Olowookere](https://medium.com/@katherineolowookere/introduction-to-gpu-programming-with-triton-d7412289bd51)
- [NCCL Deep Dive: Cross Data Center Communication and Network Topology Awareness](https://developer.nvidia.com/blog/nccl-deep-dive-cross-data-center-communication-and-network-topology-awareness/)
- [Day 4 of Deepseek's Open-Source Week: From DualPipe to EPLB - Medium](https://medium.com/@345490675/day-4-of-deepseeks-open-source-week-from-dualpipe-to-eplb-ed90f2f81d55)
- [RDMA over Commodity Ethernet at Scale - ResearchGate](https://www.researchgate.net/publication/305781076_RDMA_over_Commodity_Ethernet_at_Scale)
- [A Quick Start. Introduction to vLLM | by Okan Yenigün | Towards Dev - Medium](https://medium.com/towardsdev/vllm-a-quick-start-cf1c48aa5890)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention - arXiv](https://arxiv.org/pdf/2309.06180)
- [Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints - Rice Computer Science](https://www.cs.rice.edu/~eugeneng/papers/SOSP23.pdf)
- [ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development - USENIX](https://www.usenix.org/system/files/nsdi25-wan-borui.pdf)
- [Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs - arXiv](https://arxiv.org/html/2510.08726v1)
- [Why FlashAttention? On GPU Memory Bandwidth, Tiling and… | by Katherine Oluwadarasimi Olowookere - Medium](https://medium.com/@katherineolowookere/why-flashattention-4b0f6cca8653)
- [Introducing Triton: Open-source GPU programming for neural networks - OpenAI](https://openai.com/index/triton/)
- [Triton Kernel Compilation Stages - PyTorch](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- [Triton GPU Kernel Programming Guide: CUDA Alternative - RightNow AI](https://www.rightnowai.co/guides/frameworks/triton)
- [How I Wrote FlashAttention-2 from Scratch in Custom Triton Kernels - Medium](https://medium.com/@katherineolowookere/how-i-wrote-flashattention-2-from-scratch-in-custom-triton-kernels-885cac1da357)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning - arXiv](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision - NIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf)
- [Distributed Hybrid Parallelism for Large Language Models: Comparative Study and System Design Guide - arXiv](https://www.arxiv.org/pdf/2602.09109)
- [NVIDIA/Megatron-LM: Ongoing research training transformer models at scale - GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-LM - Hugging Face](https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm)
- [Parallelisms Guide — Megatron Bridge - NVIDIA Documentation](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/parallelisms.html)
- [Zero Redundancy Optimizer - DeepSpeed](https://www.deepspeed.ai/tutorials/zero/)
- [Ultimate Guide To Scaling ML Models - Megatron-LM | ZeRO | DeepSpeed | Mixed Precision - YouTube](https://www.youtube.com/watch?v=hc0u4avAkuM)
- [Technical Blog - NVIDIA Developer](https://developer.nvidia.com/blog/)
- [DeepSeek-V3 Technical Report - arXiv](https://arxiv.org/pdf/2412.19437)
- [DeepSeek Open-Sources DeepSeek-V3, a 671B Parameter Mixture of Experts LLM - InfoQ](https://www.infoq.com/news/2025/01/deepseek-v3-llm/)
- [Zettascale in Practice: OSU and NCCL Benchmark on NVIDIA H100 GPU Clusters for HPC and AI Workloads - Oracle Blogs](https://blogs.oracle.com/cloud-infrastructure/zettascale-osu-nccl-benchmark-h100-ai-workloads)
- [How does NCCL decide which algorithm to use? · Issue #457 - GitHub](https://github.com/NVIDIA/nccl/issues/457)
- [Unpacking NCCL: A Deep Dive into Multi-GPU Communication | by Nitin Kesarwani - Medium](https://medium.com/@nitin966/unpacking-nccl-a-deep-dive-into-multi-gpu-communication-2b667e77d96d)
- [Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication - NVIDIA Technical Blog](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)
- [An Efficient, Reliable and Observable Collective Communication Library in Large-scale GPU Training Clusters - arXiv](https://arxiv.org/html/2510.00991v1)
- [Scaling Distributed Machine Learning with In-Network Aggregation - KAUST Repository](https://repository.kaust.edu.sa/bitstreams/54859003-4fe9-48d0-9c88-fde774704974/download)
- [Collective Communication Optimization(CCO): Use cases, Requirements and Analysis - IETF Datatracker](https://datatracker.ietf.org/meeting/119/materials/slides-119-nfsv4-rdma-in-ai-networking-collective-communication-optimizationcco-use-cases-requirements-and-analysis-00)
- [Mastering LLM Techniques: Inference Optimization - NVIDIA Technical Blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [Why vLLM is the best choice for AI inference today - Red Hat Developer](https://developers.redhat.com/articles/2025/10/30/why-vllm-best-choice-ai-inference-today)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention - arXiv](https://arxiv.org/abs/2309.06180)
- [vllm.core.block_manager - vLLM Docs](https://docs.vllm.ai/en/v0.10.1/api/vllm/core/block_manager.html)
- [Explaining the Code of the vLLM Inference Engine | by Charles L. Chen - Medium](https://medium.com/@crclq2018/explaining-the-source-code-behind-the-vllm-fast-inference-engine-91429f54d1f7)
- [Paged Attention - vLLM Docs](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [The Architecture Behind vLLM: How PagedAttention Improves Memory Utilization - Medium](https://medium.com/@mandeep0405/the-architecture-behind-vllm-how-pagedattention-improves-memory-utilization-2f9b25272110)
- [Code Review: Deep Dive into vLLM's Architecture and Implementation Analysis of OpenAI-Compatible Serving - Zerohertz](https://zerohertz.github.io/vllm-openai-1/)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention - vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training - Snowflake](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
- [Volcano: Collision between containers and batch computing - CNCF](https://www.cncf.io/blog/2021/02/26/volcano-collision-between-containers-and-batch-computing/)
- [Understanding Slurm for AI/ML Workloads - WhiteFiber](https://www.whitefiber.com/blog/understanding-slurm-for-ai-ml-workloads)
- [SLURM, SUNK, SLINKY, SLONK: Chasing Speed, Stability, and Scale for Bleeding-Edge ML - Runhouse](https://www.run.house/blog/slurm-for-ml)
- [Ray on GKE: New features for AI scheduling and scaling - Google Cloud Blog](https://cloud.google.com/blog/products/containers-kubernetes/ray-on-gke-new-features-for-ai-scheduling-and-scaling)
- [Uber's Journey to Ray on Kubernetes: Ray Setup - Uber Blog](https://www.uber.com/blog/ubers-journey-to-ray-on-kubernetes-ray-setup/)
- [Fault-tolerant training: How we build reliable clusters for distributed AI workloads - Nebius](https://nebius.com/blog/posts/how-we-build-reliable-clusters)
- [How checkpointing impacts AI infrastructure storage requirements and cluster size - Cudo Compute](https://www.cudocompute.com/blog/storage-requirements-for-ai-clusters)
- [Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates - arXiv](https://arxiv.org/pdf/2309.08125)
- [The Batch - DeepLearning.AI | AI News & Insights](https://www.deeplearning.ai/the-batch/)
- [Engineering at Meta - Engineering at Meta Blog](https://engineering.fb.com/)
- [How to Build a Lock-Free Data Structure in Rust - OneUptime](https://oneuptime.com/blog/post/2026-01-30-how-to-build-a-lock-free-data-structure-in-rust/view)
- [Learn Lock-Free Programming with iLogtail - Alibaba Cloud Community](https://www.alibabacloud.com/blog/learn-lock-free-programming-with-ilogtail_601461)
- [ZeroIPC: Transforming Shared Memory into an Active Computational Substrate - metafunctor](https://metafunctor.com/post/2025-01-zeroipc/)
- [How to implement zero-copy tcp using lock-free circular buffer in C++ - Stack Overflow](https://stackoverflow.com/questions/11295474/how-to-implement-zero-copy-tcp-using-lock-free-circular-buffer-in-c)
- [Building High-Performance AI/ML Pipelines with C++ and CUDA - Tomato Soup](https://www.wholetomato.com/blog/building-high-performance-ai-ml-pipelines-with-c-and-cuda/)
- [Understanding Flash Attention: Writing the Algorithm from Scratch in Triton - Alex Dremov](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/)
- [Fear and Loathing in Lock-Free Programming | by Tyler Neely - Medium](https://medium.com/@tylerneely/fear-and-loathing-in-lock-free-programming-7158b1cdd50c)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vllm-project/guidellm: Evaluate and Enhance Your LLM Deployments for Real-World Inference Needs - GitHub](https://github.com/vllm-project/guidellm)
- [Top 5 AI/ML Infrastructure Engineer Interview Questions - Interviews Chat](https://www.interviews.chat/questions/aiml-infrastructure-engineer)
- [How to Design a GPU Cluster for AI Training - The Deep Learning System Design Interview - YouTube](https://www.youtube.com/watch?v=o9xAU7KWbvI)

---

## 项目结构

本仓库的目录结构按照上述培养方案进行组织，便于系统性学习和进度追踪。

```
ai-infra-engineering-tutorial-2026/
├── courses/                           # 五大核心课程
│   ├── SYS-601-gpu-architecture/      # GPU 架构与算子优化
│   │   ├── readings/                  # 论文与博客
│   │   ├── notes/                     # 学习笔记
│   │   ├── code/                      # 代码实现
│   │   └── labs/                      # 实验练习
│   ├── SYS-602-distributed-training/  # 分布式训练与混合并行
│   │   ├── readings/
│   │   ├── notes/
│   │   ├── code/
│   │   └── labs/
│   ├── SYS-603-networking/            # 高性能 AI 网络与集合通信
│   │   ├── readings/
│   │   ├── notes/
│   │   ├── code/
│   │   └── labs/
│   ├── SYS-604-inference/             # 高吞吐 LLM 推理系统
│   │   ├── readings/
│   │   ├── notes/
│   │   ├── code/
│   │   └── labs/
│   └── SYS-605-cluster-scheduling/    # 大规模集群调度与容错
│       ├── readings/
│       ├── notes/
│       ├── code/
│       └── labs/
├── interview-prep/                    # 七轮面试准备
│   ├── round-1-systems-coding/        # 底层系统编程
│   ├── round-2-gpu-kernel/            # GPU 架构与 Kernel 设计
│   ├── round-3-distributed-training/  # 分布式训练架构
│   ├── round-4-networking/            # 高性能网络
│   ├── round-5-inference/             # 高吞吐推理系统
│   ├── round-6-resilience/            # 弹性与调度
│   └── round-7-capstone/              # 终极系统设计
├── projects/                          # 实战项目（Portfolio）
│   ├── triton-kernels/                # Triton 算子实现
│   ├── megatron-experiments/          # Megatron-LM 实验
│   ├── nccl-analysis/                 # NCCL 通信分析
│   ├── vllm-exploration/              # vLLM 推理探索
│   └── cluster-simulation/            # 集群调度模拟
└── resources/                         # 学习资源
    ├── papers/                        # 论文 PDF
    ├── blogs/                         # 博客存档
    ├── tools/                         # 工具脚本
    └── datasets/                      # 测试数据集
```

### 学习路径时间轴

```
 Month 1-3    │ Month 4-7           │ Month 8-10   │ Month 11-12
──────────────┼─────────────────────┼──────────────┼────────────
   SYS-601    │ SYS-602 + SYS-603   │   SYS-604    │   SYS-605
 GPU/CUDA     │ Distributed + Net   │  Inference   │  Scheduling
   (Serial)   │    (Parallel)       │   (Serial)   │   (Serial)
```

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourusername/ai-infra-engineering-tutorial-2026.git
cd ai-infra-engineering-tutorial-2026

# 从 SYS-601 开始学习
cd courses/SYS-601-gpu-architecture
cat README.md

# 阅读课程大纲，按照 Progress Tracking 表格推进学习
```

### 进度追踪

每门课程的 README.md 中包含详细的：
- **Learning Objectives** - 可勾选的学习目标
- **Required Reading** - 可勾选的必读材料
- **Progress Tracking** - 周级别的学习进度表

建议使用 GitHub Issues 或 Projects 功能进行个人进度管理。
