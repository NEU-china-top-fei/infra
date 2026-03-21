# Getting started

## Who this is for

Engineers with strong **systems fundamentals** (OS, networking, concurrent programming) and some **LLM exposure**, who want to grow into **AI infrastructure** roles: large-scale training, high-performance networking, inference systems, and cluster orchestration.

## Prerequisites

- **C++**: comfortable with C++14/17, RAII, and at least one concurrent program (threads + atomics or locks).
- **Python**: NumPy/PyTorch basics; you will read framework code, not only call APIs.
- **Linear algebra**: matrix multiply shapes, softmax, attention at a high level.
- **Optional but valuable**: one prior CUDA or Triton tutorial; one profiling session with `nsys` or PyTorch profiler.

## Hardware

| Track | Minimum | Better |
|-------|---------|--------|
| SYS-601 labs | CPU-only for reading; **NVIDIA GPU** for CUDA/Triton | RTX 30xx+, cloud T4/L4/A10, or institutional cluster |
| SYS-602+ | Multi-GPU or cloud multi-node for real runs | Same |

Without a GPU you can still complete **reading, notes, and paper summaries**; schedule rented GPU time for kernel and profiling labs.

## Suggested environment

1. **Linux** (or macOS for notes; CUDA/Triton needs Linux or WSL2 with NVIDIA stack).
2. **NVIDIA driver** matching your CUDA toolkit; follow [NVIDIA CUDA installation](https://docs.nvidia.com/cuda/) for your OS.
3. **Python 3.10+** and a virtualenv per project. Optional umbrella file: [`requirements-labs.txt`](../requirements-labs.txt); per-project pins under `projects/*/requirements.txt`.
4. **Nsight Systems** (`nsys`) when you reach profiling milestones—install with CUDA toolkit or standalone.

## How to work through the repo

1. Open the course README under `courses/SYS-601-gpu-architecture/` and check off **Learning Objectives** and **Required Reading** as you go.
2. Put personal notes in each course’s `notes/` (gitignored patterns optional—see `CONTRIBUTING.md`).
3. Keep **one portfolio-quality artifact** in `projects/` per quarter (kernel, benchmark report, or scheduler simulation).

## Progress tracking

- Use GitHub **Issues** or **Projects** with labels per course code.
- Copy `interview-prep/TEMPLATE.md` when you draft answers for each round.
