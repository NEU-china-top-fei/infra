# Triton kernels (portfolio)

Small, reviewable GPU kernels with a correctness check against PyTorch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python softmax_triton.py
```

Requires an **NVIDIA GPU** with a working PyTorch CUDA build.

## Contents

| File | Description |
|------|-------------|
| `softmax_triton.py` | Row softmax in Triton vs `torch.softmax` |

Extend with fused LayerNorm, a simplified attention tile, etc., and attach `nsys` profiles in your notes.

## Related course material

- Labs: [`courses/SYS-601-gpu-architecture/labs/lab02-triton-fundamentals/`](../../courses/SYS-601-gpu-architecture/labs/lab02-triton-fundamentals/)
