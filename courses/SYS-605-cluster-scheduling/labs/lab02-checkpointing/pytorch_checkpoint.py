"""Minimal torch.save / load round-trip."""

from __future__ import annotations

import tempfile

import torch
import torch.nn as nn


def main() -> None:
    m = nn.Linear(32, 32)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.save(m.state_dict(), path)
    m2 = nn.Linear(32, 32)
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    m2.load_state_dict(sd)
    x = torch.randn(4, 32)
    assert torch.allclose(m(x), m2(x))
    print("checkpoint round-trip ok")


if __name__ == "__main__":
    main()
