"""Print Prometheus queries useful for GPU training (cheat sheet)."""

from __future__ import annotations

QUERIES = [
    "rate(nvidia_gpu_utilization[5m])",
    "node_network_receive_bytes_total",
    "kube_pod_status_phase",
]


def main() -> None:
    for q in QUERIES:
        print(q)


if __name__ == "__main__":
    main()
