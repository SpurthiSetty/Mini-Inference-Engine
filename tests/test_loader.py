"""engine.loader.load_weights: structure, shapes, dtype/device, and values vs. the raw checkpoint."""

from __future__ import annotations

import pytest
import torch

from engine.loader import load_weights

LAYER_KEYS = {
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "post_attention_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}


def expected_layer_shapes(config: dict) -> dict[str, tuple[int, ...]]:
    hidden, inter, d = config["hidden_size"], config["intermediate_size"], config["head_dim"]
    q, kv = config["num_attention_heads"] * d, config["num_key_value_heads"] * d
    return {
        "input_layernorm": (hidden,),
        "self_attn.q_proj": (q, hidden),
        "self_attn.k_proj": (kv, hidden),
        "self_attn.v_proj": (kv, hidden),
        "self_attn.o_proj": (hidden, q),
        "self_attn.q_norm": (d,),
        "self_attn.k_norm": (d,),
        "post_attention_layernorm": (hidden,),
        "mlp.gate_proj": (inter, hidden),
        "mlp.up_proj": (inter, hidden),
        "mlp.down_proj": (hidden, inter),
    }


@pytest.fixture(scope="module")
def weights(checkpoint_path, config, device):
    # bf16 is the stored dtype, so values can be compared exactly.
    return load_weights(checkpoint_path, torch.bfloat16, device, config["num_hidden_layers"])


def all_tensors(weights: dict):
    yield "embed_tokens", weights["embed_tokens"]
    yield "norm", weights["norm"]
    for i, layer in enumerate(weights["layers"]):
        for key, t in layer.items():
            yield f"layers[{i}][{key}]", t


def test_top_level_structure(weights, config):
    assert set(weights) == {"embed_tokens", "norm", "layers"}
    assert len(weights["layers"]) == config["num_hidden_layers"]
    assert weights["embed_tokens"].shape == (config["vocab_size"], config["hidden_size"])
    assert weights["norm"].shape == (config["hidden_size"],)


def test_layer_keys_and_shapes(weights, config):
    shapes = expected_layer_shapes(config)
    for i, layer in enumerate(weights["layers"]):
        assert set(layer) == LAYER_KEYS, f"layer {i}"
        for key, shape in shapes.items():
            assert tuple(layer[key].shape) == shape, f"layer {i} {key}"


def test_dtype_and_device(weights, device):
    for name, t in all_tensors(weights):
        assert t.dtype == torch.bfloat16, name
        assert t.device.type == device.type, name


def test_values_match_checkpoint(weights, get_weight, device):
    """Every tensor is the one its key claims (catches off-by-one layers and swapped names)."""
    for i in [0, 13, 27]:
        for key in LAYER_KEYS:
            expected = get_weight(f"model.layers.{i}.{key}.weight", torch.bfloat16, device)
            assert torch.equal(weights["layers"][i][key], expected), f"layer {i} {key}"
    assert torch.equal(weights["embed_tokens"], get_weight("model.embed_tokens.weight", torch.bfloat16, device))
    assert torch.equal(weights["norm"], get_weight("model.norm.weight", torch.bfloat16, device))


def test_lm_head_not_loaded(weights):
    names = [name for name, _ in all_tensors(weights)]
    assert not any("lm_head" in n for n in names)


def test_checkpoint_lm_head_is_tied(config, get_weight):
    """Justifies skipping lm_head.weight: config says tied and the stored copy is identical."""
    assert config["tie_word_embeddings"] is True
    cpu = torch.device("cpu")
    embed = get_weight("model.embed_tokens.weight", torch.bfloat16, cpu)
    lm_head = get_weight("lm_head.weight", torch.bfloat16, cpu)
    assert torch.equal(embed, lm_head)
