# Lab 02 — Triton fundamentals

Install dependencies (see also `projects/triton-kernels/requirements.txt`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch triton
python vector_add.py
python softmax.py
python matmul.py
```

Use `TRITON_INTERPRET=1` for CPU-like debugging when supported.

## Coding Problems

See [`EXERCISES.md`](EXERCISES.md) for a progressive problem set covering masked vector operations, fusion, reductions, transpose/coalescing, autotuned matmul, and a mini FlashAttention-style forward kernel.
