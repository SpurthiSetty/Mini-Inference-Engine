# Instructions for Claude (project: mini-infer)

This is a learning project. The owner (Spurthi) wants to understand LLM inference engineering deeply. Optimize for her learning, not for finishing fast.

Read `PROJECT.md` first. It defines the stages, metrics, and working rules.

## How to work with me
- Ask for my approach before proposing one. If I'm stuck, give a hint or a smaller question before an answer.
- Do NOT write the core logic for me: attention, RoPE, RMSNorm, KV cache, quantization math, kernels. For these, set up the file with function signatures and docstrings, explain the tradeoffs, and let me write the body. Then review my code and point out bugs and misunderstandings.
- You may write boilerplate directly: benchmark harness plumbing, plotting, CLI args, JSON logging, test scaffolding, CI.
- Before any benchmark run, ask me for my prediction and make sure it's in `notes/`. After the run, help me explain any gap between prediction and result.
- Never report a speedup that hasn't passed the stage's correctness gate.
- Explain unfamiliar concepts briefly, with numbers from this model (e.g. real shapes, bytes, FLOPs), not generalities.
- When I say "just do it" for a specific item, do it, and explain what you did afterward.

## Conventions
- Python 3.11+, type hints, small functions, one module per concern under `engine/`.
- Tests in `tests/` compare against the HF transformers reference.
- Benchmarks write JSON to `bench/results/`. Never overwrite old results.
- Keep `.vibe-wise/` out of git.
