"""Alias experiment for acceptance rate sweeps."""

from __future__ import annotations

import draft_model_setup


def main() -> None:
    for p in [0.5, 0.7, 0.9]:
        avg = sum(draft_model_setup.trial(p, 4) for _ in range(2000)) / 2000
        print(p, avg)


if __name__ == "__main__":
    main()
