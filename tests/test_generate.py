"""engine.generate.generate vs. HF Qwen3ForCausalLM.generate (greedy), both fp32 eager.

Prompts go through the chat template with thinking off, as in the benchmark.
HF decodes with its KV cache and the engine re-runs the full sequence, so the
two take different floating-point paths. Exact id agreement is expected here;
a mismatch must be explained by a near-tied token before relaxing anything.
The full 20-prompt, 256-token agreement check is part of the Stage 1 gate.
"""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3ForCausalLM

import engine.generate as generate_module
from engine.engine import MODEL_REPO, MODEL_REVISION, STOP_IDS, encode_chat, load_engine
from engine.generate import generate

NEW_TOKENS = 32
PROMPTS = [
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "Explain in two sentences why the sky is blue.",
]


@pytest.fixture(scope="module")
def eng(device):
    return load_engine(torch.float32, device)


@pytest.fixture(scope="module")
def hf_model(device):
    ref = Qwen3ForCausalLM.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, dtype=torch.float32, attn_implementation="eager"
    )
    # generate() falls back to the model's eos ids when none are passed, so
    # clear them here; tests that want stopping pass eos_token_id explicitly.
    ref.generation_config.eos_token_id = None
    return ref.to(device).eval()


def hf_greedy(hf_model, prompt_ids: torch.Tensor, max_new_tokens: int, stop_ids: set[int] | None) -> list[int]:
    kwargs = dict(do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=151643)
    if stop_ids is not None:
        kwargs["eos_token_id"] = sorted(stop_ids)
    with torch.no_grad():
        out = hf_model.generate(prompt_ids[None], **kwargs)
    return out[0, prompt_ids.shape[0]:].tolist()


def run_engine(eng, prompt_ids, max_new_tokens, stop_ids):
    return generate(
        prompt_ids,
        eng["weights"],
        eng["cos"],
        eng["sin"],
        eng["config"],
        max_new_tokens=max_new_tokens,
        stop_ids=stop_ids,
    )


def prompt_tensor(eng, text, device):
    return torch.tensor(encode_chat(eng["tokenizer"], text), device=device)


@pytest.mark.parametrize("text", PROMPTS)
def test_greedy_matches_hf_no_stop(text, eng, hf_model, device):
    prompt_ids = prompt_tensor(eng, text, device)
    out = run_engine(eng, prompt_ids, NEW_TOKENS, None)
    expected = hf_greedy(hf_model, prompt_ids, NEW_TOKENS, None)

    assert len(out) == NEW_TOKENS, "stop_ids=None must never stop early"
    assert all(type(t) is int for t in out), "return Python ints, not tensors"
    first_diff = next((i for i, (a, b) in enumerate(zip(out, expected)) if a != b), None)
    assert out == expected, f"first divergence at new token {first_diff}"


def test_stops_at_stop_id_inclusive(eng, hf_model, device):
    """HF ends this reply as [17 '2', 151645 <|im_end|>]; the stop token is kept."""
    prompt_ids = prompt_tensor(eng, "What is 2+2? Reply with just the number.", device)
    out = run_engine(eng, prompt_ids, NEW_TOKENS, STOP_IDS)
    expected = hf_greedy(hf_model, prompt_ids, NEW_TOKENS, STOP_IDS)

    assert out == expected
    assert out[-1] in STOP_IDS
    assert len(out) < NEW_TOKENS


def test_context_limit_raises_before_any_forward(eng, device, monkeypatch):
    """prompt + max_new_tokens past the RoPE table must fail up front, not partway through."""

    def no_forward(*args, **kwargs):
        raise AssertionError("model() was called before the context check")

    monkeypatch.setattr(generate_module, "model", no_forward)
    short_cos, short_sin = eng["cos"][:64], eng["sin"][:64]
    prompt_ids = torch.zeros(60, dtype=torch.long, device=device)
    with pytest.raises(ValueError):
        generate(
            prompt_ids,
            eng["weights"],
            short_cos,
            short_sin,
            eng["config"],
            max_new_tokens=8,
            stop_ids=None,
        )
