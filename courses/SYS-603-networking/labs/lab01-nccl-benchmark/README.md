# Lab 01 — NCCL benchmarking

Requires `nccl-tests` binaries on GPU nodes. Scripts fall back to helpful messages if missing.

| File | Role |
|------|------|
| `run_allreduce_bench.sh` | `all_reduce_perf` driver |
| `run_allgather_bench.sh` | `all_gather_perf` driver |
| `analyze_bandwidth.py` | CSV post-process stub |
| `topology_visualization.py` | ASCII fat-tree |
