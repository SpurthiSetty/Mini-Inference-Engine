"""engine.attention.attention vs. HF Qwen3Attention (eager), with real checkpoint weights.

Same input, weights, RoPE tables and causal mask on both sides. fp32 checks the
logic with assert_close defaults; bf16 is skipped until benchmarking.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from engine.attention import attention
from engine.rope import build_rope_table

MAX_SEQ = 2048  # PROJECT.md context limit
LAYERS = [0, 27]
# seq 1 is the decode shape; seq 16 == num_heads (a heads/seq mix-up passes silently)
SEQS = [1, 7, 16, 64]

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]


def load_layer(layer, dtype, device, get_weight) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}.self_attn"
    return {name: get_weight(f"{prefix}.{name}.weight", dtype, device) for name in PROJ}


def run_engine(x, w, cos, sin, config):
    return attention(
        x,
        q_proj=w["q_proj"],
        k_proj=w["k_proj"],
        v_proj=w["v_proj"],
        o_proj=w["o_proj"],
        q_norm_w=w["q_norm"],
        k_norm_w=w["k_norm"],
        eps=config["rms_norm_eps"],
        cos=cos,
        sin=sin,
        head_dim=config["head_dim"],
        num_heads=config["num_attention_heads"],
        num_kv_heads=config["num_key_value_heads"],
    )


def hf_reference(x, w, cos, sin, layer, config):
    """x is [seq, hidden]; HF wants [batch, seq, hidden], cos/sin [batch, seq, head_dim], additive mask."""
    cfg = Qwen3Config(**config)
    cfg._attn_implementation = "eager"
    ref = Qwen3Attention(cfg, layer_idx=layer).to(device=x.device, dtype=x.dtype).eval()
    with torch.no_grad():
        for name in PROJ:
            getattr(ref, name).weight.copy_(w[name])
        seq = x.shape[0]
        mask = torch.full((seq, seq), float("-inf"), device=x.device, dtype=x.dtype).triu(1)
        out, _ = ref(x[None], (cos[None, :seq], sin[None, :seq]), mask[None, None])
    return out[0]


def setup(layer, seq, dtype, device, config, get_weight):
    w = load_layer(layer, dtype, device, get_weight)
    cos, sin = build_rope_table(config["head_dim"], config["rope_theta"], MAX_SEQ, dtype)
    cos, sin = cos.to(device), sin.to(device)
    gen = torch.Generator().manual_seed(layer * 1000 + seq)
    x = torch.randn((seq, config["hidden_size"]), generator=gen).to(device=device, dtype=dtype)
    return x, w, cos, sin


def run_case(layer, seq, dtype, device, config, get_weight):
    x, w, cos, sin = setup(layer, seq, dtype, device, config, get_weight)
    out = run_engine(x, w, cos, sin, config)
    expected = hf_reference(x, w, cos, sin, layer, config)
    assert out.shape == x.shape
    assert out.dtype == expected.dtype
    return out, expected


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("layer", LAYERS)
def test_attention_fp32(layer, seq, device, config, get_weight):
    out, expected = run_case(layer, seq, torch.float32, device, config, get_weight)
    torch.testing.assert_close(out, expected)


def test_attention_is_causal(device, config, get_weight):
    """Changing tokens 5.. must leave outputs for tokens 0..4 unchanged."""
    x, w, cos, sin = setup(0, 8, torch.float32, device, config, get_weight)
    x2 = x.clone()
    x2[5:] = torch.randn(x2[5:].shape, generator=torch.Generator().manual_seed(1)).to(device)

    out, out2 = run_engine(x, w, cos, sin, config), run_engine(x2, w, cos, sin, config)
    torch.testing.assert_close(out[:5], out2[:5])
    assert not torch.allclose(out[5:], out2[5:])


@pytest.mark.skip(reason="enable before benchmarking: decide the softmax-precision question in engine/attention.py first")
@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("layer", LAYERS)
def test_attention_bf16(layer, seq, device, config, get_weight):
    out, expected = run_case(layer, seq, torch.bfloat16, device, config, get_weight)
    torch.testing.assert_close(out, expected)  # bf16 defaults: rtol=1.6e-2, atol=1e-5
