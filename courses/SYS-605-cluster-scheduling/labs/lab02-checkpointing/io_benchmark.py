"""Sequential write throughput to a temp file."""

from __future__ import annotations

import tempfile
import time


def main() -> None:
    data = b"x" * (64 * 1024 * 1024)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        t0 = time.perf_counter()
        f.write(data)
        f.flush()
        dt = time.perf_counter() - t0
    mb_s = len(data) / 1e6 / dt
    print(f"write throughput (rough): {mb_s:.1f} MB/s")


if __name__ == "__main__":
    main()
