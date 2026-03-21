#!/usr/bin/env bash
set -euo pipefail
if command -v all_reduce_perf >/dev/null 2>&1; then
  all_reduce_perf -b 8 -e 256M -f 2 -g "${CUDA_VISIBLE_DEVICES:-0}"
else
  echo "Install NCCL tests (e.g. nccl-tests) to provide all_reduce_perf, or run on a cluster image."
  echo "Fallback: python ../lab02-comm-patterns/ring_allreduce_sim.py"
fi
