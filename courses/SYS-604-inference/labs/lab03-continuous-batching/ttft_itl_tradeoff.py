"""Show that larger prefill batch increases TTFT for queued requests (toy)."""

from __future__ import annotations


def main() -> None:
    prefill_ms = 80.0
    queue = 4
    print("TTFT for last request if serial prefill:", prefill_ms * queue)


if __name__ == "__main__":
    main()
