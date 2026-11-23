"""Weight loader: model.safetensors -> nested dict of tensors, ready for engine.model.

Runs once at engine load. dtype conversion and the move to the device happen
here, never in the forward pass.
"""

from __future__ import annotations

import torch
from safetensors import safe_open
import re

def load_weights(path: str, dtype: torch.dtype, device: torch.device, num_layers: int) -> dict:
    """Read the checkpoint into the structure engine.model expects.

    Returned structure (Qwen3-0.6B shapes):

        {
            "embed_tokens": [151936, 1024]      model.embed_tokens.weight (also the output matrix)
            "norm":         [1024]              model.norm.weight
            "layers": [                         index i = model.layers.{i}
                {
                    "input_layernorm":          [1024]
                    "self_attn.q_proj":         [2048, 1024]
                    "self_attn.k_proj":         [1024, 1024]
                    "self_attn.v_proj":         [1024, 1024]
                    "self_attn.o_proj":         [1024, 2048]
                    "self_attn.q_norm":         [128]
                    "self_attn.k_norm":         [128]
                    "post_attention_layernorm": [1024]
                    "mlp.gate_proj":            [3072, 1024]
                    "mlp.up_proj":              [3072, 1024]
                    "mlp.down_proj":            [1024, 3072]
                },
                ...                             28 entries
            ],
        }

    Layer keys are the checkpoint name with "model.layers.{i}." and ".weight"
    stripped. lm_head.weight is not loaded: tie_word_embeddings is true and it
    is an identical copy of embed_tokens (311 MB in bf16).

    Args:
        path: local path to model.safetensors (pinned revision).
        dtype: dtype for every returned tensor (checkpoint is stored in bf16).
        device: device for every returned tensor.
        num_layers: number of decoder layers 

    Returns:
        The dict above, every tensor in `dtype` on `device`.
    """
    tensors = {}
    tensors["layers"] = [{} for _ in range(num_layers)]
    with safe_open(path, framework="pt", device=str(device)) as f:
        for k in f.keys():
            if k == "model.embed_tokens.weight": 
                tensors["embed_tokens"] = f.get_tensor(k).to(dtype)
            elif k == "model.norm.weight":
                tensors["norm"] = f.get_tensor(k).to(dtype)
            elif k == "lm_head.weight":
                pass
            else:
                m = re.match(r"model\.layers\.(\d+)\.(.+)\.weight$", k)
                layer_idx = int(m.group(1))   # 0
                param = m.group(2) 
                tensors["layers"][layer_idx][param] = f.get_tensor(k).to(dtype)

    return tensors 