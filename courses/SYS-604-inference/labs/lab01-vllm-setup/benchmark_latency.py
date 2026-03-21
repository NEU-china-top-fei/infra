"""Latency-oriented benchmark stub (requires vLLM + GPU)."""

from __future__ import annotations

import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, default="facebook/opt-125m")
    args = ap.parse_args()
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as e:
        print("vLLM not installed:", e)
        return
    llm = LLM(model=args.model)
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.perf_counter()
    llm.generate(["Hello"], sp)
    print(f"One-shot latency (rough): {(time.perf_counter()-t0)*1000:.1f} ms")


if __name__ == "__main__":
    main()
