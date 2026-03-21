"""TTFT vs ITL weights for a simple SLA score."""

from __future__ import annotations

import argparse


def score(ttft: float, itl: float, w_ttft: float = 0.4, w_itl: float = 0.6) -> float:
    return w_ttft * ttft + w_itl * itl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ttft-ms", type=float, default=120.0)
    ap.add_argument("--itl-ms", type=float, default=15.0)
    args = ap.parse_args()
    print("weighted score:", score(args.ttft_ms, args.itl_ms))


if __name__ == "__main__":
    main()
