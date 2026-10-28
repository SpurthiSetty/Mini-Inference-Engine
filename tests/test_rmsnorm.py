"""engine.rmsnorm.rms_norm vs. HF Qwen3RMSNorm, with real checkpoint weights.

Same input, same weight, same dtype on both sides. fp32 checks the logic
(torch.testing.assert_close defaults: rtol=1.3e-6, atol=1e-5); bf16 checks
precision choices and is skipped until benchmarking.
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from engine.rmsnorm import rms_norm

SEQ = 7

# (checkpoint weight, input shape): hidden-state norms and per-head QK-norms
CASES = [
    ("model.layers.0.input_layernorm.weight", (SEQ, 1024)),
    ("model.layers.0.post_attention_layernorm.weight", (SEQ, 1024)),
    ("model.norm.weight", (SEQ, 1024)),
    ("model.layers.0.self_attn.q_norm.weight", (SEQ, 16, 128)),
    ("model.layers.0.self_attn.k_norm.weight", (SEQ, 8, 128)),
]

# Input magnitudes. Real residual-stream scale is unmeasured until Stage 1
# runs end to end; these span small to large so the math is checked at both.
SCALES = [1.0, 100.0]


def hf_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    ref = Qwen3RMSNorm(weight.shape[0], eps=eps).to(device=x.device, dtype=x.dtype)
    with torch.no_grad():
        ref.weight.copy_(weight)
        return ref(x)


def run_case(name, shape, scale, dtype, device, config, get_weight):
    eps = config["rms_norm_eps"]
    weight = get_weight(name, dtype, device)
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(shape, generator=gen) * scale).to(device=device, dtype=dtype)

    out = rms_norm(x, weight, eps)
    expected = hf_reference(x, weight, eps)

    assert out.shape == x.shape
    assert out.dtype == expected.dtype
    return out, expected


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name,shape", CASES)
def test_rms_norm_fp32(name, shape, scale, device, config, get_weight):
    out, expected = run_case(name, shape, scale, torch.float32, device, config, get_weight)
    torch.testing.assert_close(out, expected)


@pytest.mark.skip(reason="enable before benchmarking: decide the fp32-upcast question in engine/rmsnorm.py first")
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("name,shape", CASES)
def test_rms_norm_bf16(name, shape, scale, device, config, get_weight):
    out, expected = run_case(name, shape, scale, torch.bfloat16, device, config, get_weight)
    torch.testing.assert_close(out, expected)  # bf16 defaults: rtol=1.6e-2, atol=1e-5
