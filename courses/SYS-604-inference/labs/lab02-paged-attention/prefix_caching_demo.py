"""Demonstrate prefix sharing: two prompts with shared system prompt hash."""

from __future__ import annotations

import hashlib


def prefix_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def main() -> None:
    sys_prompt = "You are a helpful assistant.\n"
    p1 = sys_prompt + "Write a haiku about GPUs."
    p2 = sys_prompt + "Write a limerick about NCCL."
    print(prefix_id(sys_prompt), "shared")
    print(prefix_id(p1), prefix_id(p2))


if __name__ == "__main__":
    main()
