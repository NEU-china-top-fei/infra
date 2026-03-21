"""Recovery time = checkpoint interval + restore + catch-up (toy)."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--restore-minutes", type=float, default=3.0)
    args = ap.parse_args()
    print("downtime lower bound (min):", args.ckpt_minutes + args.restore_minutes)


if __name__ == "__main__":
    main()
