"""Parse simple vLLM log lines or print guidance."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path)
    args = ap.parse_args()
    if args.log and args.log.exists():
        print(args.log.read_text()[:4000])
    else:
        print("Point --log at an engine log; otherwise capture TTFT/ITL from vLLM metrics endpoint.")


if __name__ == "__main__":
    main()
