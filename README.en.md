# AI Infra Engineering Tutorial 2026

A structured, self-study path from **backend / distributed systems** toward **AI infrastructure engineering** (large-model training stacks, RDMA/NCCL, inference engines, and cluster scheduling).

- **Main guide (Chinese + full detail):** [README.md](README.md)
- **Documentation index:** [docs/README.md](docs/README.md)

## What you will study

| Course | Focus |
|--------|--------|
| SYS-601 | GPU architecture, CUDA/Triton, operator performance |
| SYS-602 | Distributed training: TP/PP/DP, ZeRO, Megatron-style systems |
| SYS-603 | Collectives, NCCL, RDMA/RoCE |
| SYS-604 | LLM inference: KV cache, continuous batching, PagedAttention / vLLM |
| SYS-605 | Scheduling, checkpointing, fault tolerance at scale |

## Suggested timeline

- **Months 1–3:** SYS-601 (serial).
- **Months 4–7:** SYS-602 and SYS-603 in parallel.
- **Months 8–10:** SYS-604.
- **Months 11–12:** SYS-605.

## Quick start

```bash
git clone https://github.com/<YOUR_GITHUB>/<YOUR_FORK>.git
cd ai-infra-engineering-tutorial-2026
cd courses/SYS-601-gpu-architecture
# Follow README.md; place notes in notes/ and code in labs/ or projects/
```

## Interview prep

Seven rounds mapped to courses live under [`interview-prep/`](interview-prep/). See [`interview-prep/README.md`](interview-prep/README.md).

## Contributing & license

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [LICENSE](LICENSE) (Apache-2.0)
- [NOTICE](NOTICE)
