"""Stage 0: does bf16 run correctly and fast enough on MPS?

Two parts:
  1. Correctness: each op Qwen3 uses, run in bf16 (and fp16) on MPS, compared
     against an fp32 CPU reference on identical inputs.
  2. Speed: the decode-shaped MLP matmul x[1, 1024] @ W[3072, 1024]^T in
     fp32 / bf16 / fp16, timed three ways:
       - per_op:          synchronize around every single call (overhead + GPU time)
       - batched_cached:  synchronize once around `inner` back-to-back calls on the
                          SAME weight; it stays in the on-chip cache, so this
                          overstates bandwidth (kept for contrast)
       - batched_dram:    same, but each call uses the next weight from a pool of
                          distinct matrices totalling `pool_mb` (default 250 MB,
                          ~10x the cache), so every read comes from DRAM, as in decode

Prediction lives in notes/stage0-bf16-mps.md. Results are written to
bench/results/stage0-bf16-mps-<date>-<time>.json and never overwritten.

Usage:
    python bench/bf16_mps_check.py [--warmup 20] [--reps 200] [--inner 100] [--pool-mb 250]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

# Qwen3-0.6B shapes (config.json)
HIDDEN = 1024
INTERMEDIATE = 3072
VOCAB = 151936
N_HEADS = 16
HEAD_DIM = 128

PEAK_BANDWIDTH_GBS = 307.0  # Apple spec, M5 Pro (PROJECT.md)
RESULTS_DIR = Path(__file__).parent / "results"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# ---------------------------------------------------------------- correctness

def make_op_cases(gen: torch.Generator) -> dict[str, tuple[Callable, list[torch.Tensor]]]:
    """Each case: (fn, fp32 CPU inputs). Float inputs get cast per dtype; int inputs don't."""
    x = torch.randn(1, HIDDEN, generator=gen)
    w = torch.randn(INTERMEDIATE, HIDDEN, generator=gen) / HIDDEN**0.5
    h = torch.randn(1, INTERMEDIATE, generator=gen)
    scores = torch.randn(N_HEADS, 128, 128, generator=gen) * 3
    table = torch.randn(VOCAB, HIDDEN, generator=gen)
    ids = torch.randint(0, VOCAB, (32,), generator=gen)
    y = torch.randn(1, HIDDEN, generator=gen)
    return {
        "linear [1,1024]x[3072,1024]^T": (F.linear, [x, w]),
        "softmax [16,128,128] dim=-1": (lambda s: torch.softmax(s, dim=-1), [scores]),
        "silu [1,3072]": (F.silu, [h]),
        "rsqrt(mean(x^2)+eps) [1,1024]": (lambda v: torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6), [x]),
        "mul [1,1024]": (torch.mul, [x, y]),
        "add [1,1024]": (torch.add, [x, y]),
        "embedding [151936,1024] idx[32]": (lambda i, t: F.embedding(i, t), [ids, table]),
    }


def normalized_max_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    """max|out - ref| / max|ref|. Avoids blowing up on near-zero reference elements."""
    out, ref = out.float().cpu(), ref.float().cpu()
    return ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


def check_correctness(device: torch.device) -> dict:
    gen = torch.Generator().manual_seed(0)
    results = {}
    for name, (fn, inputs) in make_op_cases(gen).items():
        ref = fn(*inputs)
        row = {}
        for dname, dtype in DTYPES.items():
            cast = [t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device) for t in inputs]
            try:
                out = fn(*cast)
                row[dname] = {
                    "ok": True,
                    "out_device": str(out.device),
                    "out_dtype": str(out.dtype).removeprefix("torch."),
                    "normalized_max_error": normalized_max_error(out, ref),
                }
            except Exception as e:  # missing MPS kernel shows up here (fallback is disabled)
                row[dname] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        results[name] = row
    return results


# ---------------------------------------------------------------------- speed

def sync() -> None:
    torch.mps.synchronize()


def summarize_us(samples_ns: list[float]) -> dict:
    us = sorted(s / 1e3 for s in samples_ns)
    p95 = us[min(len(us) - 1, int(round(0.95 * (len(us) - 1))))]
    return {"median_us": statistics.median(us), "p95_us": p95, "min_us": us[0], "n": len(us)}


def time_per_op(fn: Callable[[], torch.Tensor], warmup: int, reps: int) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter_ns()
        fn()
        sync()
        samples.append(time.perf_counter_ns() - t0)
    return summarize_us(samples)


def time_batched(fns: list[Callable[[], torch.Tensor]], warmup: int, reps: int, inner: int) -> dict:
    """Cycle through `fns` (one per weight matrix), one sync per `inner` calls."""
    i = 0
    for _ in range(max(warmup, len(fns))):  # touch every weight at least once
        fns[i % len(fns)]()
        i += 1
    sync()
    samples = []
    for _ in range(max(reps // 10, 5)):
        sync()
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fns[i % len(fns)]()
            i += 1
        sync()
        samples.append((time.perf_counter_ns() - t0) / inner)
    return summarize_us(samples)


def bandwidth(bytes_moved: int, t_us: float) -> dict:
    gbs = bytes_moved / (t_us * 1e-6) / 1e9
    return {"achieved_GBs": gbs, "pct_of_peak": 100 * gbs / PEAK_BANDWIDTH_GBS}


SPEED_MODES = ("per_op", "batched_cached", "batched_dram")


def make_weight_pool(device: torch.device, dtype: torch.dtype, pool_mb: float, gen: torch.Generator) -> list[torch.Tensor]:
    """Distinct [3072, 1024] weights whose total size is >= pool_mb."""
    bytes_each = INTERMEDIATE * HIDDEN * torch.finfo(dtype).bits // 8
    n = max(2, -(-int(pool_mb * 1e6) // bytes_each))  # ceil division
    return [
        (torch.randn(INTERMEDIATE, HIDDEN, generator=gen) / HIDDEN**0.5).to(device, dtype)
        for _ in range(n)
    ]


def check_speed(device: torch.device, warmup: int, reps: int, inner: int, pool_mb: float) -> dict:
    gen = torch.Generator().manual_seed(0)
    x32 = torch.randn(1, HIDDEN, generator=gen)
    w32 = torch.randn(INTERMEDIATE, HIDDEN, generator=gen) / HIDDEN**0.5
    results = {}
    for dname, dtype in DTYPES.items():
        x, w = x32.to(device, dtype), w32.to(device, dtype)
        fn = lambda: F.linear(x, w)
        bytes_moved = (w.numel() + x.numel() + INTERMEDIATE) * w.element_size()
        per_op = time_per_op(fn, warmup, reps)
        cached = time_batched([fn], warmup, reps, inner)
        pool = make_weight_pool(device, dtype, pool_mb, gen)
        dram = time_batched([lambda wi=wi: F.linear(x, wi) for wi in pool], warmup, reps, inner)
        results[dname] = {
            "bytes_moved": bytes_moved,
            "pool": {"n_matrices": len(pool), "total_MB": len(pool) * w.numel() * w.element_size() / 1e6},
            "per_op": per_op | bandwidth(bytes_moved, per_op["median_us"]),
            "batched_cached": cached | bandwidth(bytes_moved, cached["median_us"]),
            "batched_dram": dram | bandwidth(bytes_moved, dram["median_us"]),
        }
        del pool
        torch.mps.empty_cache()
    for mode in SPEED_MODES:
        fp32 = results["fp32"][mode]["median_us"]
        for dname in ("bf16", "fp16"):
            results[dname][mode]["speedup_vs_fp32"] = fp32 / results[dname][mode]["median_us"]
    return results


# ------------------------------------------------------------------- metadata

def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # no commits yet


def metadata(args: argparse.Namespace) -> dict:
    return {
        "stage": "stage0-bf16-mps",
        "engine": "torch ops (no model)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "hardware": {
            "chip": "Apple M5 Pro",
            "peak_bandwidth_GBs": PEAK_BANDWIDTH_GBS,
            "platform": platform.platform(),
        },
        "backend": "mps",
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "args": vars(args),
        "reference": "fp32 on CPU, identical inputs (seed 0)",
        "error_metric": "max|out-ref| / max|ref|",
    }


# ------------------------------------------------------------------- printing

def print_report(correctness: dict, speed: dict) -> None:
    print("\nCorrectness (normalized max error vs fp32 CPU; bf16 eps = 2^-8 = 3.9e-3, fp16 eps = 2^-11 = 4.9e-4)")
    print(f"{'op':38s} " + " ".join(f"{d:>14s}" for d in DTYPES))
    for name, row in correctness.items():
        cells = []
        for d in DTYPES:
            r = row[d]
            if not r["ok"]:
                cells.append(f"{'ERROR':>14s}")
            else:
                flag = "" if r["out_device"].startswith("mps") else "!cpu"
                cells.append(f"{r['normalized_max_error']:>10.2e}{flag:>4s}")
        print(f"{name:38s} " + " ".join(cells))
    for name, row in correctness.items():
        for d, r in row.items():
            if not r["ok"]:
                print(f"  {name} [{d}]: {r['error']}")

    print("\nSpeed: linear x[1,1024] @ W[3072,1024]^T (median / p95 us, achieved GB/s, % of peak)")
    for mode in SPEED_MODES:
        pool = f" (pool: {speed['bf16']['pool']['total_MB']:.0f} MB per dtype)" if mode == "batched_dram" else ""
        print(f"  {mode}{pool}:")
        for d in DTYPES:
            r = speed[d][mode]
            sp = f"  {r['speedup_vs_fp32']:.2f}x vs fp32" if "speedup_vs_fp32" in r else ""
            print(f"    {d}: {r['median_us']:7.1f} / {r['p95_us']:7.1f} us  "
                  f"{r['achieved_GBs']:6.1f} GB/s ({r['pct_of_peak']:4.1f}%){sp}")


# ----------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--inner", type=int, default=100, help="calls per sync in batched modes")
    p.add_argument("--pool-mb", type=float, default=250.0, help="total size of distinct weights in batched_dram")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise SystemExit("Unset PYTORCH_ENABLE_MPS_FALLBACK: silent CPU fallback would invalidate the check")
    device = torch.device("mps")

    correctness = check_correctness(device)
    speed = check_speed(device, args.warmup, args.reps, args.inner, args.pool_mb)
    print_report(correctness, speed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"stage0-bf16-mps-{datetime.now():%Y%m%d-%H%M%S}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.write_text(json.dumps(metadata(args) | {"correctness": correctness, "speed": speed}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
