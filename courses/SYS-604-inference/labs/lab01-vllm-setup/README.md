# Lab 01 — vLLM setup / benchmarks

| File | Role |
|------|------|
| `install_vllm.sh` | venv + `pip install vllm` |
| `benchmark_throughput.py` | Generate many prompts |
| `benchmark_latency.py` | Short max_tokens TTFT-ish probe |
| `analyze_metrics.py` | Log/metrics helper |

vLLM wheels are CUDA-specific; adjust for your driver/toolkit.
