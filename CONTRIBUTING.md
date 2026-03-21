# Contributing

This repository is primarily a **personal study syllabus**. If you fork it or open pull requests, keep changes focused and easy to review.

## What to contribute

- **Fixes** to broken links, typos, and unclear wording in Markdown.
- **Small lab improvements**: clearer comments, safer defaults, compatibility notes for Triton/CUDA versions.
- **Optional**: additional `questions.md` prompts for `interview-prep/` (keep them generic, no verbatim proprietary interview content).

## What to avoid

- Large binary blobs (PDFs, datasets, checkpoints) — use links and local `resources/` with `.gitignore`.
- Pasting **confidential** employer code or internal benchmarks.
- Drive-by refactors unrelated to the learning goal.

## Style

- **Markdown**: one sentence per line is fine; keep headings consistent with nearby files.
- **Code**: match the style of neighboring files; run `pre-commit` if configured.
- **Languages**: Chinese and English both appear; prefer clear technical terms in English where standard (NCCL, RDMA, etc.).

## Legal

By contributing, you agree your contributions are licensed under the same license as the repo ([Apache-2.0](LICENSE)).
