# mini-infer

A from-scratch LLM inference engine for Qwen3-0.6B, built to learn inference engineering: KV cache, int8 quantization, profiling, and a custom kernel.

See `PROJECT.md` for goals, stages, and metrics.

## Layout
```
engine/   model, kv_cache, quant, kernels (one module per concern)
bench/    harness + results/ (JSON per run, never overwritten)
tests/    correctness tests against the HF transformers reference
notes/    predictions vs. results, one file per stage
```

## Setup
```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock   # exact versions used for results
pip install -e '.[dev]' --no-deps
```
`pyproject.toml` declares minimum versions; `requirements.lock` records the exact ones. Upgrading is deliberate: regenerate the lock (`pip freeze --exclude-editable > requirements.lock`) and rerun baselines.

## Reproduce
TODO: fill in as stages are completed.
