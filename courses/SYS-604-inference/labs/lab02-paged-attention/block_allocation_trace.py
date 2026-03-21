"""Simulate block allocation for KV cache pages."""

from __future__ import annotations


class BlockPool:
    def __init__(self, num_blocks: int, block_size_tokens: int) -> None:
        self.free = set(range(num_blocks))
        self.block_size = block_size_tokens

    def alloc(self, need_tokens: int) -> list[int]:
        n = (need_tokens + self.block_size - 1) // self.block_size
        got: list[int] = []
        for _ in range(n):
            b = self.free.pop()
            got.append(b)
        return got

    def free_blocks(self, blocks: list[int]) -> None:
        self.free.update(blocks)


def main() -> None:
    pool = BlockPool(num_blocks=16, block_size_tokens=16)
    b1 = pool.alloc(40)
    print("allocated blocks", b1)


if __name__ == "__main__":
    main()
