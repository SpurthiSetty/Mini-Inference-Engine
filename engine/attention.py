"""Grouped-query self-attention, as used by Qwen3 (28 layers, one call each per forward pass).

    q_proj  [2048, 1024]   16 query heads x 128
    k_proj  [1024, 1024]    8 key/value heads x 128
    v_proj  [1024, 1024]    8 key/value heads x 128
    o_proj  [1024, 2048]   back to hidden size
    q_norm / k_norm [128]  QK-norm, shared across heads

Architecture sizes (head_dim, num_heads, num_kv_heads) come from config.json;
input sizes (seq) come from the tensors.
"""

from __future__ import annotations

from torch import Tensor
import torch

from engine.rmsnorm import rms_norm
from engine.rope import apply_rope
import math


def attention(
    x: Tensor,
    *,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_norm_w: Tensor,
    k_norm_w: Tensor,
    eps: float,
    cos: Tensor,
    sin: Tensor,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> Tensor:
    """Causal grouped-query self-attention over a full sequence starting at position 0.

    Data flow (shapes for Qwen3-0.6B, heads = 16, kv_heads = 8, head_dim = 128):

        x                      [seq, 1024]        already through input_layernorm
        q = x @ q_proj.T       [seq, 2048]   ->   [seq, 16, 128]
        k = x @ k_proj.T       [seq, 1024]   ->   [seq, 8, 128]
        v = x @ v_proj.T       [seq, 1024]   ->   [seq, 8, 128]
        q, k: rms_norm, then apply_rope(..., start=0)       (v untouched)
        k, v: each kv head repeated for its 2 query heads (0,0,1,1,...)  -> [seq, 16, 128]
        heads first                                          [16, seq, 128]
        scores = q @ k^T / sqrt(head_dim)                    [16, seq, seq]
        -inf above the diagonal, softmax over the last (key) axis
        scores @ v                                           [16, seq, 128]
        back to [seq, 16, 128], merge heads                  [seq, 2048]
        @ o_proj.T                                           [seq, 1024]

    Args:
        x: [seq, hidden], normalized hidden states.
        q_proj, k_proj, v_proj, o_proj: projection weights from the checkpoint, stored [out, in].
        q_norm_w, k_norm_w: [head_dim] QK-norm weights (checkpoint self_attn.q_norm.weight,
            self_attn.k_norm.weight).
        eps: rms_norm_eps from config.json (1e-6).
        cos, sin: [max_seq, head_dim] tables from engine.rope.build_rope_table.
        head_dim: config.json head_dim (128).
        num_heads: config.json num_attention_heads (16).
        num_kv_heads: config.json num_key_value_heads (8); num_heads // num_kv_heads
            query heads share each kv head.

    Returns:
        [seq, hidden], the attention output before the residual add.

    Stage 1 has no KV cache: every call recomputes from position 0. Stage 2 adds
    `start` and cached keys/values.

    Open question (decide before the bf16 test): HF computes the softmax in fp32
    and casts back (eager_attention_forward, dtype=torch.float32). Do you?
    """

    seq = x.shape[0]
    group_size = num_heads // num_kv_heads
    q = x @ q_proj.T 
    k = x @ k_proj.T
    v = x @ v_proj.T 

    q_head = torch.reshape(q,(seq, num_heads, head_dim)) # [seq, 16, 128] is expected form for rope
    q_norm = rms_norm(q_head, q_norm_w, eps)
    q_rope = apply_rope(q_norm, cos, sin, start=0)

    k_head = torch.reshape(k,(seq, num_kv_heads, head_dim)) # [seq, 8, 128]
    k_norm = rms_norm(k_head, k_norm_w, eps)
    k_rope = apply_rope(k_norm, cos, sin, start=0)

    v_head = torch.reshape(v,(seq, num_kv_heads, head_dim)) # [seq, 8, 128]
    v_att = v_head.transpose(0,1)  # [8, seq, 128]
    v_att_group = torch.repeat_interleave(v_att, repeats=group_size, dim=0) # [16, seq, 128]

    q_att = q_rope.transpose(0,1) # [16, seq, 128] to batch accross all the heads
    k_att = k_rope.transpose(0,1) # [8, seq, 128]
    k_att_group = torch.repeat_interleave(k_att, repeats=group_size, dim=0) # [16, seq, 128]

    att_num = q_att @ k_att_group.transpose(1, 2) # [16, seq, seq]
    scores = att_num / math.sqrt(head_dim)
    mask = torch.tril(torch.ones(seq, seq, device=x.device)) # makes upper triangle all 0s 
    w = scores.masked_fill_(mask == 0, float("-inf")) # [16, seq, seq] will it automatically apply it to the last 2 dimentions? 1st is batch
    att_head = torch.softmax(w, dim=-1) @ v_att_group # [16, seq, 128]
    att_head = att_head.transpose(0,1)
    att_full = torch.reshape(att_head, (seq, -1)) # [seq, 2048]
    output = att_full @ o_proj.T #[seq, 1024]
    return output
