#!/usr/bin/env bash
set -euo pipefail
if command -v all_gather_perf >/dev/null 2>&1; then
  all_gather_perf -b 8 -e 256M -f 2 -g "${CUDA_VISIBLE_DEVICES:-0}"
else
  echo "Install nccl-tests for all_gather_perf."
fi
