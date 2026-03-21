"""Alpha-beta latency model for collectives."""

from __future__ import annotations

import argparse


def ring_time(n: int, data_bytes: int, alpha: float, beta: float) -> float:
    """T ≈ 2(n-1)α + 2(n-1)/n * data * β"""
    return 2 * (n - 1) * alpha + (2 * (n - 1) / n) * data_bytes * beta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--data-mb", type=float, default=256)
    ap.add_argument("--alpha-us", type=float, default=5.0, help="Latency per hop (µs)")
    ap.add_argument("--beta-us-per-byte", type=float, default=1e-3, help="Inv bandwidth µs/byte")
    args = ap.parse_args()
    b = int(args.data_mb * 1024 * 1024)
    t = ring_time(args.world_size, b, args.alpha_us * 1e-6, args.beta_us_per_byte * 1e-6)
    print(f"Ring AllReduce time (rough): {t*1e3:.3f} ms")


if __name__ == "__main__":
    main()
