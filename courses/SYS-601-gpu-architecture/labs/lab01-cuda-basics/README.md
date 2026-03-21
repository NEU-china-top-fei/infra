# Lab 01 — CUDA basics

Build with:

```bash
nvcc -std=c++17 -O2 vector_add.cu -o vector_add
./vector_add
```

Requirements: NVIDIA CUDA toolkit, driver, and a GPU.

Files:

- `vector_add.cu` — minimal elementwise add
- `matrix_mul.cu` — TODO: tiled matmul (starter stub)
- `shared_memory.cu` — TODO: use `__shared__` for a reduction or tile
