# Megatron-LM experiments

Experiments around tensor/pipeline parallelism, bubble measurement, and communication profiling.

**Ideas**

- Script a minimal TP/PP config sweep and log step time.
- Capture `nsys` timelines for one forward/backward.
- Build a small **memory footprint calculator** for weights + optimizer states.

Place scripts under this directory; link upstream Megatron as a git submodule or document a pinned commit in your notes.
