"""engine.mlp.mlp vs. HF Qwen3MLP, with real checkpoint weights.

Same input and weights on both sides. fp32 checks the logic with assert_close
defaults; bf16 is skipped until benchmarking.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from engine.mlp import mlp

LAYERS = [0, 27]
# The MLP never mixes tokens, so longer sequences add no coverage.
SEQS = [1, 7]

PROJ = ["gate_proj", "up_proj", "down_proj"]


def load_layer(layer, dtype, device, get_weight) -> dict[str, torch.Tensor]:
    return {name: get_weight(f"model.layers.{layer}.mlp.{name}.weight", dtype, device) for name in PROJ}


def hf_reference(x, w, config):
    ref = Qwen3MLP(Qwen3Config(**config)).to(device=x.device, dtype=x.dtype).eval()
    with torch.no_grad():
        for name in PROJ:
            getattr(ref, name).weight.copy_(w[name])
        return ref(x)


def setup(layer, seq, dtype, device, config, get_weight):
    w = load_layer(layer, dtype, device, get_weight)
    gen = torch.Generator().manual_seed(layer * 1000 + seq)
    x = torch.randn((seq, config["hidden_size"]), generator=gen).to(device=device, dtype=dtype)
    return x, w


def run_case(layer, seq, dtype, device, config, get_weight):
    x, w = setup(layer, seq, dtype, device, config, get_weight)
    out = mlp(x, **w)
    expected = hf_reference(x, w, config)
    assert out.shape == x.shape
    assert out.dtype == expected.dtype
    return out, expected


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("layer", LAYERS)
def test_mlp_fp32(layer, seq, device, config, get_weight):
    out, expected = run_case(layer, seq, torch.float32, device, config, get_weight)
    torch.testing.assert_close(out, expected)


def test_mlp_gate_and_up_not_interchangeable(device, config, get_weight):
    """SiLU goes on the gate branch only, so swapping gate_proj and up_proj must change the output."""
    x, w = setup(0, 7, torch.float32, device, config, get_weight)
    out = mlp(x, **w)
    swapped = mlp(x, gate_proj=w["up_proj"], up_proj=w["gate_proj"], down_proj=w["down_proj"])
    assert not torch.allclose(out, swapped)


@pytest.mark.skip(reason="enable before benchmarking, with the rest of the bf16 tests")
@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("layer", LAYERS)
def test_mlp_bf16(layer, seq, device, config, get_weight):
    out, expected = run_case(layer, seq, torch.bfloat16, device, config, get_weight)
    torch.testing.assert_close(out, expected)  # bf16 defaults: rtol=1.6e-2, atol=1e-5
