# SYS-601: GPU Architecture and Operator Optimization

**Priority:** 1 (Critical) | **Duration:** Months 1-3 | **Dependencies:** None

## Overview

本课程的核心命题是打破深度学习领域的"内存墙 (Memory Wall)"。随着 Transformer 架构的扩张，模型算力需求的增长速度已远远超过了 GPU 物理显存带宽的增长速度，导致标准 Attention 等机制被严重限制在内存带宽瓶颈 (Memory-Bound) 上。

### Core Engineering Domains
- Single-node compute optimization
- C++/CUDA programming
- Triton compiler and kernels
- GPU memory hierarchy (HBM, SRAM, Registers)

## Learning Objectives

- [ ] 深刻理解 GPU 内存层次结构（HBM、SRAM/Shared Memory、Registers）
- [ ] 掌握 Warp 调度机制与 Tensor Cores 运作原理
- [ ] 精通 Triton 编译器从 Python AST → MLIR → LLVM IR → PTX 的完整管线
- [ ] 实现 FlashAttention 算法的简化版本
- [ ] 熟练使用 Nsight Systems/Compute 进行性能分析

## Source Code Study

### Primary Repository: `openai/triton`
- **Focus File:** `triton/python/tutorials/06-fused-attention.py`
- **Study Goal:** 理解 Triton Block 级别内存编程模型

### Key Concepts to Master
1. SRAM 分块计算 (Tiling)
2. 在线 Softmax (Online Softmax)
3. 算子融合 (Operator Fusion)
4. Thread Block 与 Warp 工作负载划分

## Required Reading

### Academic Papers
- [ ] Triton: an intermediate language and compiler for tiled neural network computations (Tillet et al., 2019)
- [ ] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
- [ ] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (Dao, 2023)
- [ ] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (Shah et al., 2024)
- [ ] Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs (2025)

### Engineering Blogs & Tutorials
- [ ] [OpenAI Triton 1.0 Release Blog](https://openai.com/index/triton/)
- [✓] [Introduction to GPU Programming with Triton](https://medium.com/@katherineolowookere/introduction-to-gpu-programming-with-triton-d7412289bd51)
- [✓] [How I Wrote FlashAttention-2 from Scratch in Custom Triton Kernels](https://medium.com/@katherineolowookere/how-i-wrote-flashattention-2-from-scratch-in-custom-triton-kernels-885cac1da357)
- [✓] [Triton Kernel Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- [ ] [Building High-Performance AI/ML Pipelines with C++ and CUDA](https://www.wholetomato.com/blog/building-high-performance-ai-ml-pipelines-with-c-and-cuda/)
- [ ] [Understanding Flash Attention: Writing the algorithm from scratch in Triton](https://alexdremov.me/understanding-flash-attention-writing-the-algorithm-from-scratch-in-triton/)
- [ ] [Fear and Loathing in Lock-Free Programming](https://medium.com/@tylerneely/fear-and-loathing-in-lock-free-programming-7158b1cdd50c)
- [ ] [ZeroIPC: Transforming Shared Memory into an Active Computational Substrate](https://metafunctor.com/post/2025-01-zeroipc/)

## Hands-on Labs

### Lab 1: CUDA Basics
```
labs/lab01-cuda-basics/
├── vector_add.cu
├── matrix_mul.cu
└── shared_memory.cu
```

### Lab 2: Triton Fundamentals
```
labs/lab02-triton-fundamentals/
├── vector_add.py
├── softmax.py
└── matmul.py
```

### Lab 3: FlashAttention Implementation
```
labs/lab03-flashattention/
├── naive_attention.py
├── flash_attention_v1.py
└── flash_attention_v2.py   # head-merged batching wrapper
```

## Interview Mapping

This course directly maps to:
- **Round 1:** Low-Level Systems Coding (C++/CUDA & Concurrency)
- **Round 2:** GPU Architecture and Kernel Design

## Progress Tracking

| Week | Topic | Status |
|------|-------|--------|
| 1-2 | GPU Architecture Fundamentals | ⬜ |
| 3-4 | CUDA Programming Model | ⬜ |
| 5-6 | Triton Basics & Compilation | ⬜ |
| 7-8 | FlashAttention Theory | ⬜ |
| 9-10 | FlashAttention Implementation | ⬜ |
| 11-12 | Profiling & Optimization | ⬜ |
