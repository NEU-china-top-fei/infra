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
