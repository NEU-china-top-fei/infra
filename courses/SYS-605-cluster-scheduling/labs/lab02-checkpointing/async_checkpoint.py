"""Background checkpoint writer (single process toy)."""

from __future__ import annotations

import tempfile
import threading
import time

import torch
import torch.nn as nn


def main() -> None:
    m = nn.Linear(64, 64)
    done = threading.Event()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    def worker() -> None:
        time.sleep(0.05)
        torch.save(m.state_dict(), path)
        done.set()

    t = threading.Thread(target=worker)
    t.start()
    # "training" continues
    for _ in range(10):
        _ = m(torch.randn(8, 64))
    t.join()
    done.wait()
    print("async checkpoint written", path)


if __name__ == "__main__":
    main()
