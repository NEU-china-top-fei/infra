# 培养方案与课程精读

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

