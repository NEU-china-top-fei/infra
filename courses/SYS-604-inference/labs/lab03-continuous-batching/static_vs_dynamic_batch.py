"""Compare static batching vs adding requests over time (toy timeline)."""

from __future__ import annotations

def static_batch(times: list[float]) -> float:
    return max(times) * len(times)


def dynamic_batch(arrival: list[float], work: list[float]) -> float:
    t = 0.0
    for a, w in zip(arrival, work):
        t = max(t, a) + w
    return t


def main() -> None:
    arrival = [0, 0.1, 0.2, 0.3]
    work = [1.0, 1.0, 1.0, 1.0]
    print("dynamic makespan (serial server):", dynamic_batch(arrival, work))


if __name__ == "__main__":
    main()
