# Notes (SYS-601)

Personal study notes for SYS-601. You may commit them or keep them private—if you use a private branch, add patterns to `.git/info/exclude` or a local-only gitignore.

## Source Code Study

- [`source-code-study-06-fused-attention.md`](source-code-study-06-fused-attention.md) — Triton 官方 fused attention 的前向/反向、online softmax、tiling、causal 分块、autotune、FP8 与 profiling 详解。
- [`FlashattentionInTriton.md`](FlashattentionInTriton.md) — 从零实现 FlashAttention-2 forward：算法、Triton grid/pointer、online softmax、causal stage、autotune、正确性与作者示例工程审计。

## Triton Fundamentals

- [`triton-programming-fundamentals.md`](triton-programming-fundamentals.md) — 从 Vector Add、Fused Softmax 到 Tiled Matmul，系统理解 program/grid、offset/stride/mask、融合、L2 grouping、autotune、benchmark 与调试。
- [`triton-compilation-stages.md`](triton-compilation-stages.md) — 从 Python AST、TTIR、TTGIR、LLVM IR 到 PTX/cubin，理解 JIT specialization、GPU layout、backend 分叉，并包含 Triton 3.8 + RTX 4060 实测与 IR 阅读练习。
- [Lab 02 Triton Coding Problems](../labs/lab02-triton-fundamentals/EXERCISES.md) — 七道递进练习，从 tail-safe vector add 到 mini attention forward，包含测试集、验收标准、性能任务和分级提示。
