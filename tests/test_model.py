"""Full model: engine.loader + engine.model vs. HF Qwen3ForCausalLM, both fp32 eager.

Start of the Stage 1 logits gate. Greedy argmax must match HF exactly at every
position. Logits use the fp32 working tolerance rtol=1e-3, atol=1e-4 (a correct
implementation measured max abs diff <= 6.4e-5, failing the default atol=1e-5
on near-zero logits). PROJECT.md sets the real gate value after measuring fp32
and bf16 runs. Run with -s to see the max-abs-diff per prompt.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from engine.loader import load_weights
from engine.model import model
from engine.rope import build_rope_table
from tests.conftest import MODEL_REPO, MODEL_REVISION

MAX_SEQ = 2048  # PROJECT.md context limit
FP32_TOL = dict(rtol=1e-3, atol=1e-4)
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Large language models generate text one token at a time, so the cost of decoding",
]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)


@pytest.fixture(scope="module")
def hf_model(device):
    ref = Qwen3ForCausalLM.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, dtype=torch.float32, attn_implementation="eager"
    )
    return ref.to(device).eval()


@pytest.fixture(scope="module")
def engine_fp32(checkpoint_path, config, device):
    weights = load_weights(checkpoint_path, torch.float32, device, config["num_hidden_layers"])
    cos, sin = build_rope_table(config["head_dim"], config["rope_theta"], MAX_SEQ, torch.float32)
    return weights, cos.to(device), sin.to(device)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_hf_fp32(prompt, tokenizer, hf_model, engine_fp32, config, device):
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
    weights, cos, sin = engine_fp32

    logits = model(ids, weights, cos, sin, config)
    with torch.no_grad():
        expected = hf_model(ids[None]).logits[0]

    assert logits.shape == (ids.shape[0], config["vocab_size"])
    max_abs = (logits - expected).abs().max().item()
    print(f"\n[{prompt[:30]!r}] seq={ids.shape[0]} max-abs-diff={max_abs:.3e}")

    assert torch.equal(logits.argmax(-1), expected.argmax(-1)), "greedy next-token prediction differs"
    torch.testing.assert_close(logits, expected, **FP32_TOL)
