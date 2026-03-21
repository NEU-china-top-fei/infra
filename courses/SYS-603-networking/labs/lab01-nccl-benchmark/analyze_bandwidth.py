"""Plot bandwidth vs message size from nccl-tests style CSV (two columns: bytes, time)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, help="Optional CSV: bytes,seconds")
    args = ap.parse_args()
    if not args.csv or not args.csv.exists():
        print("Provide --csv with columns bytes,seconds from a benchmark run.")
        return
    rows = list(csv.DictReader(args.csv.open()))
    for r in rows[:5]:
        print(r)


if __name__ == "__main__":
    main()
