"""
    Gated MLP (SwiGLU), as used by Qwen3 (28 layers, one call each per forward pass).

    gate_proj  [3072, 1024]
    up_proj    [3072, 1024]
    down_proj  [1024, 3072]

No size arguments: nothing is reshaped, so every size comes from the weights.
"""

from __future__ import annotations

from torch import Tensor
import torch


def mlp(x: Tensor, *, gate_proj: Tensor, up_proj: Tensor, down_proj: Tensor) -> Tensor:
    """down( silu(gate(x)) * up(x) ).

    Data flow (Qwen3-0.6B):

        x                          [seq, 1024]   already through post_attention_layernorm
        g = silu(x @ gate_proj.T)  [seq, 3072]
        u = x @ up_proj.T          [seq, 3072]
        g * u                      [seq, 3072]   elementwise
        @ down_proj.T              [seq, 1024]

    Args:
        x: [seq, hidden], normalized hidden states.
        gate_proj, up_proj, down_proj: weights from the checkpoint, stored [out, in].
            config.json hidden_act = "silu".

    Returns:
        [seq, hidden], the MLP output before the residual add.
    """
    g = torch.nn.functional.silu(x @ gate_proj.T)
    u = x @ up_proj.T  
    res = (g * u) @ down_proj.T  
    return res 
