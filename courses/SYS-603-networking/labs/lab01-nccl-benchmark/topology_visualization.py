"""Print a toy fat-tree ASCII diagram (educational)."""

from __future__ import annotations

TOPO = """
        +------------------+
        |   Core layer     |
        +------------------+
           /    \\
    +------+      +------+
    | Agg  |      | Agg  |
    +------+      +------+
      |  |          |  |
   +--+--+       +--+--+
   |ToR|ToR|     |ToR|ToR|
   +---+---+     +---+---+
    GPUs x8       GPUs x8
"""


def main() -> None:
    print(TOPO)


if __name__ == "__main__":
    main()
