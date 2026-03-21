"""Swap pipeline stage mapping after node loss (toy lists)."""

from __future__ import annotations


def main() -> None:
    stages = ["s0", "s1", "s2", "s3"]
    stages[2] = "s2_spare"
    print("new mapping:", stages)


if __name__ == "__main__":
    main()
