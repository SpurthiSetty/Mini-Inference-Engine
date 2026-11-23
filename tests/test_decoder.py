"""engine.decoder.decoder_layer vs. HF Qwen3DecoderLayer (eager), with real checkpoint weights.

Weights come from the test fixture, not engine.loader, so a loader bug can't
show up here. bf16 is skipped until benchmarking.

fp32 tolerance: rtol=1e-3, atol=1e-4 instead of the defaults. Layer 27 outputs
reach ~1292 (a massive-activation channel), where fp32 spacing is 1.2e-4, so
the default atol of 1e-5 is below one rounding step. A correct implementation
measured max rel diff 1.5e-4; the residual-from-normed-input bug differs by ~26.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from engine.decoder import decoder_layer
from engine.rope import build_rope_table
from tests.test_loader import LAYER_KEYS

MAX_SEQ = 2048  # PROJECT.md context limit
FP32_TOL = dict(rtol=1e-3, atol=1e-4)
LAYERS = [0, 27]
SEQS = [1, 7, 16]


def load_layer(i, dtype, device, get_weight) -> dict[str, torch.Tensor]:
    return {key: get_weight(f"model.layers.{i}.{key}.weight", dtype, device) for key in LAYER_KEYS}


def hf_reference(h, layer, cos, sin, i, config):
    cfg = Qwen3Config(**config)
    cfg._attn_implementation = "eager"
    ref = Qwen3DecoderLayer(cfg, layer_idx=i).to(device=h.device, dtype=h.dtype).eval()
    ref.load_state_dict({f"{key}.weight": t for key, t in layer.items()})
    seq = h.shape[0]
    mask = torch.full((seq, seq), float("-inf"), device=h.device, dtype=h.dtype).triu(1)
    with torch.no_grad():
        out = ref(h[None], attention_mask=mask[None, None], position_embeddings=(cos[None, :seq], sin[None, :seq]))
    return out[0]


def run_case(i, seq, dtype, device, config, get_weight):
    layer = load_layer(i, dtype, device, get_weight)
    cos, sin = build_rope_table(config["head_dim"], config["rope_theta"], MAX_SEQ, dtype)
    cos, sin = cos.to(device), sin.to(device)
    gen = torch.Generator().manual_seed(i * 1000 + seq)
    h = torch.randn((seq, config["hidden_size"]), generator=gen).to(device=device, dtype=dtype)

    out = decoder_layer(h, layer, cos, sin, config)
    expected = hf_reference(h, layer, cos, sin, i, config)
    assert out.shape == h.shape
    assert out.dtype == expected.dtype
    return out, expected


@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("layer_idx", LAYERS)
def test_decoder_layer_fp32(layer_idx, seq, device, config, get_weight):
    out, expected = run_case(layer_idx, seq, torch.float32, device, config, get_weight)
    torch.testing.assert_close(out, expected, **FP32_TOL)


@pytest.mark.skip(reason="enable before benchmarking, with the rest of the bf16 tests")
@pytest.mark.parametrize("seq", SEQS)
@pytest.mark.parametrize("layer_idx", LAYERS)
def test_decoder_layer_bf16(layer_idx, seq, device, config, get_weight):
    out, expected = run_case(layer_idx, seq, torch.bfloat16, device, config, get_weight)
    torch.testing.assert_close(out, expected)  # bf16 defaults: rtol=1.6e-2, atol=1e-5
