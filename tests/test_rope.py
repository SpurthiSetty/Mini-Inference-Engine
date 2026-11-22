"""engine.rope vs. HF Qwen3RotaryEmbedding + apply_rotary_pos_emb.

Table built once (build_rope_table), sliced by start offset in apply_rope.
fp32 checks the logic with assert_close defaults; this assumes inv_freq is
computed with HF's recipe (1 / theta ** (arange(0, d, 2) / d) in fp32), since
a 1-ulp difference becomes ~1.2e-4 rad at position 2047. bf16 is skipped
until benchmarking.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding, apply_rotary_pos_emb

from engine.rope import apply_rope, build_rope_table

MAX_SEQ = 2048  # PROJECT.md context limit

# (heads, seq, start): q/k prefill, single-token decode, seq == heads
# (wrong broadcast passes silently), and positions near the table end.
CASES = [
    (16, 7, 0),
    (8, 7, 0),
    (16, 1, 57),
    (8, 1, 57),
    (16, 16, 0),
    (16, 8, MAX_SEQ - 8),
]


def hf_reference(x: torch.Tensor, start: int, config: dict) -> torch.Tensor:
    """x is [seq, heads, head_dim]; HF wants [batch, seq, heads, head_dim] with unsqueeze_dim=2."""
    rotary = Qwen3RotaryEmbedding(Qwen3Config(**config)).to(x.device)
    position_ids = torch.arange(start, start + x.shape[0], device=x.device)[None]
    cos, sin = rotary(x[None], position_ids)
    out, _ = apply_rotary_pos_emb(x[None], x[None], cos, sin, unsqueeze_dim=2)
    return out[0]


def run_case(heads, seq, start, dtype, device, config):
    head_dim = config["head_dim"]
    cos, sin = build_rope_table(head_dim, config["rope_theta"], MAX_SEQ, dtype)
    assert cos.shape == sin.shape == (MAX_SEQ, head_dim)
    assert cos.dtype == sin.dtype == dtype
    cos, sin = cos.to(device), sin.to(device)

    gen = torch.Generator().manual_seed(0)
    x = torch.randn((seq, heads, head_dim), generator=gen).to(device=device, dtype=dtype)

    out = apply_rope(x, cos, sin, start)
    expected = hf_reference(x, start, config)

    assert out.shape == x.shape
    assert out.dtype == expected.dtype
    return out, expected


@pytest.mark.parametrize("heads,seq,start", CASES)
def test_rope_fp32(heads, seq, start, device, config):
    out, expected = run_case(heads, seq, start, torch.float32, device, config)
    torch.testing.assert_close(out, expected)


def test_rope_table_dtype_is_output_cast_only(config):
    """Angles must be computed in fp32 regardless of dtype; only the returned tables are cast.

    Guards against computing positions/inv_freq in bf16 (257 -> 256, angle error of
    several radians by position ~300), which the fp32 tests can't see.
    """
    args = (config["head_dim"], config["rope_theta"], MAX_SEQ)
    cos32, sin32 = build_rope_table(*args, torch.float32)
    cos16, sin16 = build_rope_table(*args, torch.bfloat16)
    assert cos16.dtype == sin16.dtype == torch.bfloat16
    torch.testing.assert_close(cos16, cos32.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(sin16, sin32.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("start,seq", [(MAX_SEQ, 1), (MAX_SEQ - 4, 8)])
def test_rope_out_of_range_raises(start, seq, device, config):
    head_dim = config["head_dim"]
    cos, sin = build_rope_table(head_dim, config["rope_theta"], MAX_SEQ, torch.float32)
    x = torch.randn((seq, 16, head_dim), device=device)
    with pytest.raises(ValueError):
        apply_rope(x, cos.to(device), sin.to(device), start)


@pytest.mark.skip(reason="enable before benchmarking, with the rest of the bf16 tests")
@pytest.mark.parametrize("heads,seq,start", CASES)
def test_rope_bf16(heads, seq, start, device, config):
    out, expected = run_case(heads, seq, start, torch.bfloat16, device, config)
    torch.testing.assert_close(out, expected)  # bf16 defaults: rtol=1.6e-2, atol=1e-5
