"""Parse a simple nsys text export or print hints for profiling Megatron runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsys-stats", type=Path, help="Optional path to nsys stats export")
    args = ap.parse_args()
    if args.nsys_stats and args.nsys_stats.exists():
        text = args.nsys_stats.read_text(errors="ignore")
        print(text[:2000])
    else:
        print(
            "Capture with: nsys profile -o megatron python your_training_script.py\n"
            "Then: nsys stats megatron.nsys-rep --report cuda_gpu_kern_sum\n"
        )


if __name__ == "__main__":
    main()
