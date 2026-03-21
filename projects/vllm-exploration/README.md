# vLLM exploration

Inference-focused experiments: PagedAttention behavior, batching, KV fragmentation.

**Ideas**

- Trace a request through `vllm` scheduler (reading notes + annotated logs).
- Benchmark TTFT/ITL vs batch size on a small open model.
- Summarize prefix caching / radix attention behavior for interviews.
