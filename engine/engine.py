"""Load-time setup and the text-in/text-out wrapper around engine.generate.

load_engine runs once per process (config, 1.5 GB of weights, RoPE tables,
tokenizer). chat runs per request and does only per-request work.
"""

from __future__ import annotations

import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from engine.generate import generate
from engine.loader import load_weights
from engine.rope import build_rope_table

MODEL_REPO = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_SEQ = 2048  # PROJECT.md context limit
STOP_IDS = {151645, 151643}  # <|im_end|>, <|endoftext|> (generation_config.json)


def load_engine(dtype: torch.dtype, device: torch.device, max_seq: int = MAX_SEQ) -> dict:
    """Load everything generate needs, once.

    Returns:
        {"config", "weights", "cos", "sin", "tokenizer"}; tensors in `dtype` on `device`.
    """
    with open(hf_hub_download(MODEL_REPO, "config.json", revision=MODEL_REVISION)) as f:
        config = json.load(f)
    checkpoint = hf_hub_download(MODEL_REPO, "model.safetensors", revision=MODEL_REVISION)
    weights = load_weights(checkpoint, dtype, device, config["num_hidden_layers"])
    cos, sin = build_rope_table(config["head_dim"], config["rope_theta"], max_seq, dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    return {
        "config": config,
        "weights": weights,
        "cos": cos.to(device),
        "sin": sin.to(device),
        "tokenizer": tokenizer,
    }


def encode_chat(tokenizer, text: str) -> list[int]:
    """One user turn through the chat template, thinking off (PROJECT.md)."""
    messages = [{"role": "user", "content": text}]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    return encoded["input_ids"]


def chat(engine: dict, text: str, max_new_tokens: int = 256) -> str:
    """Text in, text out. Stops at the end-of-turn token."""
    device = engine["weights"]["embed_tokens"].device
    prompt_ids = torch.tensor(encode_chat(engine["tokenizer"], text), device=device)
    new_ids = generate(
        prompt_ids,
        engine["weights"],
        engine["cos"],
        engine["sin"],
        engine["config"],
        max_new_tokens=max_new_tokens,
        stop_ids=STOP_IDS,
    )
    return engine["tokenizer"].decode(new_ids, skip_special_tokens=True)
