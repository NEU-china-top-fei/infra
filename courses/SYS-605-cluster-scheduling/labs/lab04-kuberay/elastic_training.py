"""Ray tune elastic idea (runs only if ray installed)."""

from __future__ import annotations


def main() -> None:
    try:
        import ray  # type: ignore
    except Exception as e:
        print("Ray not installed:", e)
        return
    ray.init(ignore_reinit_error=True)

    @ray.remote
    def work(x: int) -> int:
        return x * x

    print(ray.get(work.remote(4)))
    ray.shutdown()


if __name__ == "__main__":
    main()
