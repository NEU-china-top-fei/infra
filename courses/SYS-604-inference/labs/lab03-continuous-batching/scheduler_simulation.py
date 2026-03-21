"""FIFO vs shortest-job-first toy for decode batches."""

from __future__ import annotations

import heapq


def fifo(work: list[float]) -> float:
    t = 0.0
    for w in work:
        t += w
    return t


def sjf(work: list[float]) -> float:
    h = list(work)
    heapq.heapify(h)
    t = 0.0
    while h:
        t += heapq.heappop(h)
    return t


def main() -> None:
    w = [3.0, 1.0, 2.0]
    print("FIFO total wait", fifo(w))
    print("SJF total wait", sjf(w))


if __name__ == "__main__":
    main()
