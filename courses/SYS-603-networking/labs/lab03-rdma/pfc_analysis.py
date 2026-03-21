"""Toy simulation: ECMP flows colliding on uplinks (qualitative)."""

from __future__ import annotations

import argparse
import random


def collision_rate(num_flows: int, num_uplinks: int, trials: int) -> float:
    hits = 0
    for _ in range(trials):
        choices = [random.randrange(num_uplinks) for _ in range(num_flows)]
        if len(set(choices)) < num_flows:
            hits += 1
    return hits / trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flows", type=int, default=64)
    ap.add_argument("--uplinks", type=int, default=8)
    ap.add_argument("--trials", type=int, default=1000)
    args = ap.parse_args()
    r = collision_rate(args.flows, args.uplinks, args.trials)
    print(f"Collision probability (toy birthday-like): {r:.3f}")


if __name__ == "__main__":
    main()
