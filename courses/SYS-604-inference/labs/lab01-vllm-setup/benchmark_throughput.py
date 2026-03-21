"""Optional vLLM throughput harness (skips gracefully if vLLM unavailable)."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, default="facebook/opt-125m")
    args = ap.parse_args()
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        print("vLLM not installed:", e)
        return
    llm = LLM(model=args.model)
    prompts = ["Hello, my name is" for _ in range(16)]
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(prompts, sp)
    print("generated", len(outputs), "outputs")


if __name__ == "__main__":
    main()
