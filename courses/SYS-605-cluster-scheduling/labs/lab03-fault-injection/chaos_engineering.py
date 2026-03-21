"""Randomly drop a \"GPU\" from a set — toy."""

from __future__ import annotations

import random


def main() -> None:
    gpus = list(range(8))
    fail = random.choice(gpus)
    survivors = [g for g in gpus if g != fail]
    print("failed", fail, "survivors", survivors)


if __name__ == "__main__":
    main()
