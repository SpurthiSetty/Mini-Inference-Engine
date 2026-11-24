"""Full Qwen3-0.6B forward pass: token ids -> logits for every position."""

from __future__ import annotations

from torch import Tensor
from engine.decoder import decoder_layer
from engine.rmsnorm import rms_norm



def model(token_ids: Tensor, weights: dict, cos: Tensor, sin: Tensor, config: dict) -> Tensor:
    """Embedding -> 28 decoder layers -> final norm -> logits.

    Data flow:

        token_ids                                  [seq]           int64
        h = embed_tokens[token_ids]                [seq, 1024]     row lookup, not a matmul
        28x: h = decoder_layer(h, layer, ...)      [seq, 1024]
        h = rms_norm(h, norm, eps)                 [seq, 1024]
        logits = h @ embed_tokens.T                [seq, 151936]   tied: same matrix as the embedding

    Stage 1 has no KV cache: generation calls this on the whole sequence for
    every new token.

    Args:
        token_ids: [seq] int64 token ids on the same device as the weights.
        weights: output of engine.loader.load_weights.
        cos, sin: [max_seq, head_dim] tables from build_rope_table, built once at load.
        config: parsed config.json.

    Returns:
        [seq, vocab] logits; row t predicts token t+1.
    """
    h = weights["embed_tokens"][token_ids] #lookup table - [seq, 1024]
    for i in range(config["num_hidden_layers"]): 
        h = decoder_layer(h, weights["layers"][i], cos, sin, config)
    h = rms_norm(h, weights["norm"], config["rms_norm_eps"])
    logits = h @ weights["embed_tokens"].T
    return logits #  [seq, 151936]


    
