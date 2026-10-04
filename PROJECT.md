# mini-infer: a from-scratch LLM inference engine

> Goal, non-goals, and rules are mine. Benchmark protocol and the metric definitions were drafted with Claude from my answers and are for me to review. Tolerances and predictions stay empty until I measure.

## Goal (2-3 sentences)
I want to build a mini inference engine, and then compare metrics like latency, decode rate, TTFT, peak memory, memory-bandwidth utilization, and output quality on my engine vs. other runtimes. So I will be writing the logic for the forward pass myself and learn how to manage the memory and operations while doing so. I want to be able to clearly explain in an interview how exactly LLM inference works, the tradeoffs between using different runtimes on different hardware, and the different ways you can optimize it. I will be running this on my MacBook Pro with the M5 Pro (MPS) and one cloud NVIDIA GPU, comparing the metrics for each, and writing up the differences in implementation.

## Non-goals
- I am not fine-tuning an LLM, building a model from scratch, or training anything (no backpropagation or gradient descent).
- I am not building distributed training or inference. It all runs on one device at a time, either my Mac or an NVIDIA GPU, and then I compare them.
- I am not building a harness or RAG system on top of this. I will use the model alone with some basic questions.
- Batch size 1 only. No batching, continuous batching, or serving API.
- I am not writing my own tokenizer. I use the HF tokenizer, with a fixed max context length of 2048 tokens.

## Model and environment
- Model: `Qwen/Qwen3-0.6B` (https://huggingface.co/Qwen/Qwen3-0.6B), pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca`
- Local dev: MacBook Pro, Apple Silicon, PyTorch MPS. I am using PyTorch since it is more widely used in industry than MLX, and it makes it easy to write the CUDA/Triton implementation on the NVIDIA GPU.
- GPU stage: cloud GPU, only for Stage 5. Push work to GitHub before every session, then destroy the instance.
- Reference implementations to compare against: HF transformers, MLX-LM, llama.cpp
- Thinking mode is off for all tests and benchmarks (via the chat template).
- Precision: bf16, since that is what the model's weights use. The HF reference runs in bf16 too. Debug logic in fp32 first, then switch to bf16.
- Record versions in every result file: PyTorch, `transformers` (4.51.0 or newer is required for Qwen3), MLX-LM, llama.cpp.

### Hardware
| Field | MacBook Pro | Cloud GPU (Stage 5) |
|---|---|---|
| Chip / GPU | Apple M5 Pro | TODO: choose when Stage 5 starts |
| CPU / GPU cores | TODO: check System Report | TODO |
| Memory | 48GB unified | TODO (VRAM) |
| Peak memory bandwidth | 307 GB/s (Apple spec) | TODO |
| OS | macOS 26.5.1 | TODO |
| Runtime / backend | PyTorch MPS | PyTorch CUDA + Triton |
| Run conditions | plugged in, note any thermal throttling | n/a |

Bandwidth utilization = (decode tok/s x bytes read per token) / peak memory bandwidth.

## Working rules
1. Correctness gate before speed. A stage doesn't count until its test passes.
2. Write a prediction in `notes/` BEFORE each benchmark run. Record the result after.
3. I write the core logic by hand (attention, RoPE, RMSNorm, KV cache, quant math, kernel). Claude handles boilerplate and reviews.
4. One stage at a time. A slipped stage gets noted and scored, not doubled up.
5. Stage 5 is required, but timeboxed to 2 weeks. If I don't have it working by then, I document what I learned and omit it.
6. Debug in fp32 first, then benchmark in bf16.
7. Comparisons use matching precision, or are clearly labeled when they don't (for example a quantized llama.cpp run).

## Benchmark protocol (decide once, never change mid-project)
- **Prompt set:** 20 fixed prompts, kept in `bench/prompts.json`. 8 short (~32 input tokens), 6 medium (~256), 6 long (~1024). Basic questions and text, no special tricks.
- **Generation:** greedy, 256 new tokens, thinking off, max context 2048. Do not stop early on the end-of-sequence token during benchmarks, so every run produces the same number of tokens.
- **Runs:** 3 warmup + 10 measured runs per prompt. Synchronize the device (`torch.mps.synchronize()` / `torch.cuda.synchronize()`) before reading any timer.
- **Statistics:** report the median and p95 over measured runs.
- **Saved per run** (`bench/results/<stage>-<date>.json`, never overwritten): engine name, git commit, model repo + revision, library versions, hardware, precision, prompt id, input and output token counts, TTFT, prefill tok/s, decode tok/s, peak memory, power/thermal notes, and timestamp.
- **Quality:** perplexity on a fixed WikiText-2 slice, plus greedy-output agreement with the HF reference.

### Metric definitions
- **TTFT:** time from starting the request to the first generated token being available (includes prefill).
- **Prefill tok/s:** input tokens / prefill time.
- **Decode tok/s:** (output tokens - 1) / time between the first and last generated token.
- **Peak memory:** from the backend's memory API; note which one, since MPS and CUDA report differently.

## Stages and metrics

| # | Stage | Correctness gate | Performance metrics | Prediction (fill in first) | Result | Done |
|---|-------|------------------|---------------------|---------------------------|--------|------|
| 1 | Baseline forward pass + generation loop | Logits match HF on identical inputs, max-abs-diff < TODO (set after measuring fp32 and bf16 runs); greedy outputs match HF on the prompt set, with any divergence explained by near-tied tokens | prefill tok/s, decode tok/s, time-to-first-token, peak memory | TODO | | [ ] |
| 2 | KV cache | Output identical to the no-cache version | decode tok/s vs. sequence length, KV bytes/token | TODO (derive KV bytes/token from the model config) | | [ ] |
| 3 | int8 quantization | Perplexity delta vs. bf16 on WikiText-2 < TODO; small task eval | model size, decode tok/s, memory | TODO (estimate speedup from bytes moved) | | [ ] |
| 4 | Profiling | n/a | time per op, achieved memory bandwidth vs. hardware peak (roofline) | TODO: which op dominates decode? | | [ ] |
| 5 | Custom kernel (CUDA/Triton), timeboxed 2 weeks | Matches reference op within TODO tolerance | kernel latency, achieved GB/s, end-to-end tok/s gain | TODO | | [ ] |

## Final deliverable
- Comparison table: my engine vs. HF / MLX-LM / llama.cpp on the same machine, plus Mac vs. NVIDIA GPU results
- Write-up: where the time goes and why (one blog-style post)
- Repo README with reproduce instructions

## Open questions
- ~~Does bf16 run correctly and fast enough on MPS with my PyTorch version? Check before starting Stage 1.~~ **Answered (2026-10-03, PyTorch 2.14.1):** yes for the ops tested. All 7 Qwen3 ops (linear, softmax, SiLU, rsqrt, mul, add, embedding) run natively on MPS in bf16 with no CPU fallback, and errors stay within bf16 precision (softmax is ~3x larger because exp() amplifies input rounding). Decode-shaped matmul reading from DRAM is 1.78x faster than fp32 (~275 GB/s streaming, ~6 us fixed cost per queued op, ~150 us per synchronize). fp16 runs at the same speed, so bf16 is native. End-to-end bf16 accuracy is still to be validated by the Stage 1 logits gate. Details: `notes/stage0-bf16-mps.md`.
- Do MLX-LM and llama.cpp support the pinned Qwen3-0.6B revision at the versions I install?
- Do I include a quantized llama.cpp run as a labeled extra, given the matching-precision rule?
- Which cloud GPU for Stage 5, and what is its peak memory bandwidth?
- If Stage 5 hits its timebox, what do I write up instead?