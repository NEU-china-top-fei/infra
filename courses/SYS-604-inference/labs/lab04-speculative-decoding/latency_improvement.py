"""Compare baseline decode vs speculative average latency."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ms", type=float, default=20.0)
    ap.add_argument("--spec-ms", type=float, default=12.0)
    args = ap.parse_args()
    print("speedup:", args.baseline_ms / args.spec_ms)


if __name__ == "__main__":
    main()
