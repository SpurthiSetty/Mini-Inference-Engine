"""Greedy generation loop: prompt token ids -> new token ids.

Stage 1 has no KV cache. Every step re-runs engine.model on the whole sequence
so far, so a 1024-token prompt with 256 new tokens costs forward passes over
1024, 1025, ..., 1279 tokens.
"""

from __future__ import annotations

from torch import Tensor
import torch 

from engine.model import model


def generate(
    prompt_ids: Tensor,
    weights: dict,
    cos: Tensor,
    sin: Tensor,
    config: dict,
    *,
    max_new_tokens: int,
    stop_ids: set[int] | None,
) -> list[int]:
    """Greedily generate up to `max_new_tokens` tokens after `prompt_ids`.

    Each step picks the argmax of the logits row that predicts the next token.

    Stopping:
        - Always stops after `max_new_tokens` new tokens.
        - If `stop_ids` is a set (normal use: {151645 <|im_end|>, 151643 <|endoftext|>}),
          also stops right after generating any id in it. The stop token IS
          included in the output, matching HF generate.
        - If `stop_ids` is None (benchmarks), never stops early: always returns
          exactly `max_new_tokens` ids.

    Context limit: the RoPE table has cos.shape[0] rows (2048). A prompt that
    would run past it must fail before any generation, not partway through.

    Args:
        prompt_ids: [prompt_len] int64 token ids on the weights' device.
        weights: output of engine.loader.load_weights.
        cos, sin: [max_seq, head_dim] tables from build_rope_table, built once at load.
        config: parsed config.json.
        max_new_tokens: upper bound on generated tokens (256 in the benchmark protocol).
        stop_ids: token ids that end generation, or None to never stop early.

    Returns:
        The new token ids only (not the prompt), as Python ints.

    Raises:
        ValueError: if len(prompt_ids) + max_new_tokens exceeds the RoPE table.
    """
    response_ids = []
    if len(prompt_ids) + max_new_tokens > len(cos): 
        raise ValueError
    token_ids = prompt_ids

    while len(response_ids) < max_new_tokens:
        llm = model(token_ids, weights, cos, sin, config) # passes ongoing accumulation of all the tokes
        next_token_id = torch.argmax(llm[-1]).unsqueeze(0) # take the id maximum value of the last row 
        response_ids.append(int(next_token_id))
        if stop_ids and next_token_id.item() in stop_ids:
            break
        token_ids = torch.concat((token_ids, next_token_id)) 
    return response_ids

