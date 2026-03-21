#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install vllm || echo "vLLM install may require matching CUDA wheels; see https://docs.vllm.ai"
