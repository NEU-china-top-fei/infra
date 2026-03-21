"""Toy tensor-parallel linear on one GPU: split columns, verify AllGather-style concat."""

from __future__ import annotations

import torch
import torch.nn as nn


def tp_linear_forward(x: torch.Tensor, w_parts: list[torch.Tensor]) -> torch.Tensor:
    """x: (B, S, H_in); each w_k is (H_in, H_out/tp)."""
    outs = [x @ w_parts[k] for k in range(len(w_parts))]
    return torch.cat(outs, dim=-1)


def main() -> None:
    torch.manual_seed(0)
    b, s, h_in, h_out = 2, 16, 128, 256
    tp = 4
    x = torch.randn(b, s, h_in, device="cuda" if torch.cuda.is_available() else "cpu")
    w = torch.randn(h_in, h_out, device=x.device)
    ref = x @ w
    w_parts = torch.chunk(w, tp, dim=1)
    got = tp_linear_forward(x, list(w_parts))
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-4)
    print("TP linear toy ok")


if __name__ == "__main__":
    main()
