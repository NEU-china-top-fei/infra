"""Illustrate sharded state dict merge (single-process mock)."""

from __future__ import annotations

import torch
import torch.nn as nn


def main() -> None:
    m = nn.Linear(16, 16)
    parts = [("w0", m.state_dict())]
    merged = {}
    for _, sd in parts:
        merged.update(sd)
    m2 = nn.Linear(16, 16)
    m2.load_state_dict(merged)
    print("merged keys:", list(merged.keys()))


if __name__ == "__main__":
    main()
