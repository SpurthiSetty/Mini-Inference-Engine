"""One Qwen3 decoder layer: pre-norm attention and MLP, each with a residual add."""

from __future__ import annotations

from torch import Tensor
from engine.attention import attention 
from engine.rmsnorm import rms_norm
from engine.mlp import mlp


def decoder_layer(h: Tensor, layer: dict, cos: Tensor, sin: Tensor, config: dict) -> Tensor:
    """h -> h + attention(norm(h)) -> h + mlp(norm(h)).

    Data flow:

        h                                                  [seq, 1024]
        a = attention(rms_norm(h, input_layernorm), ...)   [seq, 1024]
        h = h + a
        m = mlp(rms_norm(h, post_attention_layernorm), ...)
        h = h + m                                          [seq, 1024]

    This is where the layer dict and config are unpacked into the building
    blocks' flat keyword arguments (e.g. q_norm_w=layer["self_attn.q_norm"]).

    Args:
        h: [seq, hidden], the residual stream entering this layer.
        layer: one entry of load_weights(...)["layers"].
        cos, sin: [max_seq, head_dim] tables from build_rope_table, built once at load.
        config: parsed config.json (rms_norm_eps, head_dim, num_attention_heads,
            num_key_value_heads).

    Returns:
        [seq, hidden], the residual stream leaving this layer.
    """
    rms1 = rms_norm(h, layer["input_layernorm"], config["rms_norm_eps"])
    a = attention(rms1,
    q_proj = layer["self_attn.q_proj"],
    k_proj = layer["self_attn.k_proj"],
    v_proj = layer["self_attn.v_proj"],
    o_proj = layer["self_attn.o_proj"],
    q_norm_w = layer["self_attn.q_norm"],
    k_norm_w = layer["self_attn.k_norm"],
    eps = config["rms_norm_eps"],
    cos = cos,
    sin = sin,
    head_dim = config["head_dim"],
    num_heads = config["num_attention_heads"],
    num_kv_heads = config["num_key_value_heads"]
    )
    h = h + a # first residual connection 

    rms2 = rms_norm(h, layer["post_attention_layernorm"], config["rms_norm_eps"])
    m = mlp(rms2, gate_proj = layer["mlp.gate_proj"], up_proj = layer["mlp.up_proj"], down_proj = layer["mlp.down_proj"])
    h = h + m # second residual connection 
    return h 

