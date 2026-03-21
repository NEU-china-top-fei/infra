# Lab 03 — FlashAttention (simplified)

1. Run `naive_attention.py` to establish correctness on small shapes.
2. Implement a **blocked** attention in `flash_attention_v1.py` using Triton (online softmax, SRAM-friendly tiles).
3. Profile with `nsys` / PyTorch profiler and record a short note in `notes/`.

Files:

| File | Role |
|------|------|
| `naive_attention.py` | Reference \(O(L^2)\) attention |
| `flash_attention_v1.py` | Block-wise online softmax (FA-1 style) |
| `flash_attention_v2.py` | Same math, merges heads for larger batched GEMMs |

Reference: Dao et al., *FlashAttention* (2022); official Triton fused-attention tutorial in `openai/triton`.
