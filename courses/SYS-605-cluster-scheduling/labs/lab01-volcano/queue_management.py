"""Print kubectl commands to inspect Volcano queues (requires cluster)."""

from __future__ import annotations


def main() -> None:
    cmds = [
        "kubectl get podgroup -A",
        "kubectl get queue -A",
        "kubectl describe queue default",
    ]
    print("\n".join(cmds))


if __name__ == "__main__":
    main()
