# Projects

Hands-on work to build an **AI infrastructure portfolio** (reports + reproducible code + profiling artifacts).

| Directory | README |
|-----------|--------|
| `triton-kernels/` | [README](triton-kernels/README.md) — runnable softmax starter |
| `megatron-experiments/` | [README](megatron-experiments/README.md) |
| `nccl-analysis/` | [README](nccl-analysis/README.md) |
| `vllm-exploration/` | [README](vllm-exploration/README.md) |
| `cluster-simulation/` | [README](cluster-simulation/README.md) |

## Suggested project ideas

### 1. Triton Kernels (`triton-kernels/`)

Custom GPU kernels written in Triton for operator optimization.

**Suggested follow-ons:**
- [ ] Fused LayerNorm kernel
- [ ] FlashAttention simplified implementation
- [ ] Custom activation functions (SiLU, GELU)
- [ ] Efficient softmax with numerical stability

### 2. Megatron Experiments (`megatron-experiments/`)
Distributed training experiments using Megatron-LM.

**Suggested Projects:**
- [ ] TP/PP/DP configuration analysis
- [ ] Pipeline bubble measurement
- [ ] Communication profiling with nsys
- [ ] Memory footprint calculator

### 3. NCCL Analysis (`nccl-analysis/`)
Network communication analysis and optimization.

**Suggested Projects:**
- [ ] AllReduce algorithm comparison (Ring vs Tree)
- [ ] Bandwidth utilization profiling
- [ ] Topology-aware routing simulation
- [ ] ECMP hash collision analysis

### 4. vLLM Exploration (`vllm-exploration/`)
LLM inference optimization experiments.

**Suggested Projects:**
- [ ] PagedAttention memory analysis
- [ ] Continuous batching simulator
- [ ] KV Cache fragmentation study
- [ ] Speculative decoding benchmarks

### 5. Cluster Simulation (`cluster-simulation/`)
Large-scale cluster scheduling and fault tolerance.

**Suggested Projects:**
- [ ] Gang scheduling simulator
- [ ] Checkpoint I/O bottleneck analysis
- [ ] Failure recovery time calculator
- [ ] Elastic training demonstration

## Portfolio Guidelines

每个项目应包含：
1. **分析报告** - 公式推导与理论分析
2. **代码实现** - 可复现的实验代码
3. **Timeline 截图** - 使用 nsys/ncu 的性能分析
4. **对比数据** - 优化前后的性能对比

> "你的投名状不应该是简历上的空话，而必须是硬核的产出。"
