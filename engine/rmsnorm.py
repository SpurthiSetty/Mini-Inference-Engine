"""RMSNorm, as used by Qwen3 (4 places, 113 calls per forward pass).

    input_layernorm           x [seq, 1024]       weight [1024]   (28x)
    post_attention_layernorm  x [seq, 1024]       weight [1024]   (28x)
    q_norm / k_norm (QK-norm) x [seq, heads, 128] weight [128]    (56x, per head)
    model.norm (final)        x [seq, 1024]       weight [1024]   (1x)
"""

from __future__ import annotations

from torch import Tensor

def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Normalize `x` over its last dimension, then scale by `weight`.

        rms = sqrt(mean(x^2 over the last dim) + eps)
        out = (x / rms) * weight

    Args:
        x: any shape [..., d]. Each length-d vector along the last dim is
            normalized independently; leading dims are untouched
            (e.g. [seq, 1024] for hidden states, [seq, 16, 128] for QK-norm).
        weight: [d], learned per-channel scale from the checkpoint.
        eps: added inside the sqrt to avoid division by zero
            (config.json rms_norm_eps = 1e-6).

    Returns:
        Tensor with the same shape as `x`.

    Open question (decide before the bf16 test): when `x` is bf16, should the
    mean/rsqrt be computed in fp32 and cast back? The HF reference's choice
    determines what "matching" means in tests/test_rmsnorm.py::test_rms_norm_bf16.
    """
    # Compute the root mean square (RMS) of the input tensor along the last dimension
    denom = eps + (x * x).mean(dim=-1, keepdim=True)
    rms = denom.sqrt()
    return (x / rms) * weight