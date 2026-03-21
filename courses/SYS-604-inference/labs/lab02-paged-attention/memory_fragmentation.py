"""Toy: external fragmentation when allocating variable-length sequences."""

from __future__ import annotations

import random


def simulate(num_requests: int, pool_blocks: int) -> float:
    free = list(range(pool_blocks))
    random.shuffle(free)
    used = 0
    for _ in range(num_requests):
        need = random.randint(1, 4)
        if len(free) < need:
            return float("nan")
        for _ in range(need):
            free.pop()
        used += need
    return used / pool_blocks


def main() -> None:
    print(f"Utilization (toy): {simulate(10, 32):.3f}")


if __name__ == "__main__":
    main()
