#!/usr/bin/env bash
# Optional: prepare a Python env for Megatron-LM experiments (install upstream separately).
set -euo pipefail
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo "Done. Clone NVIDIA/Megatron-LM and follow its README for full setup."
