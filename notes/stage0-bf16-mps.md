# Stage 0: bf16 on MPS check

Answers the PROJECT.md open question: does bf16 run correctly and fast enough on MPS with my PyTorch version (2.14.1)?
Script: `bench/bf16_mps_check.py`. Results: `bench/results/stage0-bf16-mps-*.json`.

## Before I start
- What I think will happen (numbers, with reasoning):
  Decode is memory-bound. One MLP projection `[1, 1024] x [1024, 3072]` is ~6.3M FLOPs (well under 1 us of compute) but has to read the whole weight: 12.6 MB in fp32, 6.3 MB in bf16.
  Cost model: time ~= fixed overhead + bytes / bandwidth, with overhead ~10 us per op.
  - At peak 307 GB/s: fp32 ~= 10 + 41 = 51 us, bf16 ~= 10 + 20.5 = 30 us -> ~1.7x.
  - At 70% of peak (~215 GB/s): fp32 ~= 10 + 58.6 = 68.6 us, bf16 ~= 10 + 29.3 = 39.3 us -> ~1.74x.
  Not 2x because the fixed overhead is the same for both precisions; only the bytes term halves.
  Lower achieved bandwidth makes the bytes term a bigger share of the total, so the speedup moves toward 2x.
  With zero overhead the speedup would be 2x at any bandwidth; with huge overhead (e.g. 1000 us) it would approach 1x.
- What I'm unsure about:
  The real per-op overhead and achieved bandwidth on the M5 Pro. Whether every op Qwen3 needs has a bf16 MPS implementation.

## Prediction (written BEFORE running the benchmark)
| Metric | Predicted | Why |
|--------|-----------|-----|
| fp32 decode matmul time | 51-69 us | 10 us overhead + 12.6 MB at 100%-70% of 307 GB/s |
| bf16 decode matmul time | 30-39 us | 10 us overhead + 6.3 MB at 100%-70% of 307 GB/s |
| bf16 speedup over fp32 | ~1.7-1.75x | overhead is fixed, only bytes halve |
| bf16 correctness | all ops run on MPS, error near bf16 precision (~2^-8 = 0.4% relative) | bf16 keeps 8 mantissa bits |

## Prediction 2: batched_dram (written BEFORE the rerun)
First run showed batched mode reused one weight from the on-chip cache (105-129% of peak). Rerun cycles through ~250 MB of distinct weights (~10x cache) so every call reads from DRAM, like real decode.

| Metric | Predicted | Why |
|--------|-----------|-----|
| batched_dram vs batched_cached | slower | every read goes to DRAM instead of the faster cache |
| achieved bandwidth | ~70% of 307 GB/s (~215 GB/s) | DRAM-limited, below peak |
| bf16 speedup over fp32 | ~1.7x, closer to 2x than cached's 1.62x | not using the cache; bytes term bigger share of total |
| implied times (derived from the two rows above) | fp32 ~71 us, bf16 ~42 us | 70% bandwidth + 1.7x implies ~12.6 us fixed cost per call |

## Result
Run: `bench/results/stage0-bf16-mps-20261003-175717.json` (torch 2.14.1, macOS 26.5.1)

| Metric | Predicted | Actual | Gap explained? |
|--------|-----------|--------|----------------|
| fp32 decode matmul time | 51-69 us | per-op 184 us median (p95 254); batched 31.8 us | Yes, because of the cache which speedd it up, since it doesn't have to read from DRAM|
| bf16 decode matmul time | 30-39 us | per-op 176 us median (p95 229); batched 19.6 us ||
| bf16 speedup over fp32 | ~1.7-1.75x | per-op 1.05x; batched 1.62x | Explained becasue the overhead per op is larger than expected, more close ot 100 micro seconds rather than aorudn 10 micro seconds, it is faster batched because it leverages cached|
| achieved bandwidth (bf16) | 70-100% of 307 GB/s | per-op 12%; batched 105% (!) | Batched reused the same weight every call, so it stayed in the on-chip cache and wasn't read from DRAM; that's how it beat the 307 GB/s DRAM peak.† Per-op 12% because the sync overhead dwarfs the bytes term (the large-overhead extreme). |
| bf16 correctness | all ops on MPS, error ~2^-8 | all 7 ops ran on MPS, no errors; errors 1.5e-4 to 3.2e-3, except softmax 1.15e-2 | Matched except softmax. My first guess was that errors compound exponentially. Corrected†: exp() amplifies once, exp(s+d) ~= exp(s)(1+d), so a small absolute rounding error in a score (~0.02 for s~10) becomes a ~2% relative error in the output. Compounding happens across the 28 layers, not inside one softmax. fp16 is also ~3.5x its own eps, so it's the op, not a broken bf16 kernel. |
| fp16 vs bf16 speed | (not predicted) | same within 1% in both modes | |

Run 2: `bench/results/stage0-bf16-mps-20261003-183809.json` (same versions; pool 20 fp32 / 40 bf16 matrices, 252 MB)

| Metric | Predicted | Actual | Gap explained? |
|--------|-----------|--------|----------------|
| batched_dram vs batched_cached | slower | slower: fp32 52.2 vs 31.3 us, bf16 29.3 vs 19.6 us | Matched: every read now comes from DRAM instead of the faster cache. Cycling a pool ~10x the cache size forces eviction before a weight is reused. |
| achieved bandwidth | ~70% of peak | bf16 70.1% (215 GB/s), fp32 78.6% (241 GB/s) | Matched for bf16. Two-equation fit (52.2 = f + 12.6 MB/B, 29.3 = f + 6.3 MB/B): streaming bandwidth B = 275 GB/s (~90% of peak), fixed cost f ~= 6.4 us per call. 70%/79% are effective bandwidth (bytes / total time, including the fixed cost); fp32 is higher because its larger transfer spreads the same fixed cost over more bytes.† |
| bf16 speedup over fp32 | ~1.7x | 1.78x (fp16 1.79x) | Matched: without the cache the bytes term is a bigger share of the total, so the speedup moves toward 2x. |
| implied times | fp32 ~71 us, bf16 ~42 us | fp32 52.2 us (p95 76.8), bf16 29.3 us (p95 50.2) | Overcounted the overhead: the 70% already includes the fixed cost, so adding 12.6 us on top of it inflated the times.† Real fixed cost is ~6.4 us, not 12.6. |
| per_op (repeat of run 1) | n/a | fp32 190 us, bf16 166 us, 1.15x (run 1: 1.05x) | Per-op is noisy (p95 30-130 us above median). Solving my ratio equation (x+41)/(x+20) = 1.05 gave x = 400 us, but near 1x the ratio is too sensitive to noise; subtracting the bytes term from the absolute times gives ~150 us.† |
| batched_cached (repeat) | n/a | 131% / 105% of peak again | Cache effect is reproducible. |
| correctness (repeat) | n/a | identical to run 1 | Same inputs (seed 0) and same kernels, so it's deterministic. |

Gap entries above (except my own run-1 notes) were filled from our chat discussion on 2026-10-03. † = explained by Claude, not first reasoned by me.

## What surprised me
I was surpised by how large the overhead was compared to my expectations, and that made the per op speed up to be much lower than expected. I was also surpised by the impact of caching and how it can significanlty speed things up to be past the originally expevted bandwidth, but it's not really giving you the true picture. 
## What I can now explain without notes
