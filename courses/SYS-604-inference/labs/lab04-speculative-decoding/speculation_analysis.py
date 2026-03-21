"""End-to-end latency estimate: draft + verify."""

from __future__ import annotations

import argparse


def latency(draft_ms: float, verify_ms: float, accepted: int) -> float:
    return draft_ms * accepted + verify_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft-ms", type=float, default=2.0)
    ap.add_argument("--verify-ms", type=float, default=8.0)
    ap.add_argument("--accepted", type=int, default=3)
    args = ap.parse_args()
    print("latency ms:", latency(args.draft_ms, args.verify_ms, args.accepted))


if __name__ == "__main__":
    main()
