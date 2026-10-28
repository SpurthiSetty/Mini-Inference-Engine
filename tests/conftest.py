"""Shared fixtures: pinned Qwen3-0.6B checkpoint and config, test device.

Test-only loading. The engine gets its own loader later; tests can switch to it then.
"""

from __future__ import annotations

import json
from typing import Callable

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_REPO = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device("mps")


@pytest.fixture(scope="session")
def config() -> dict:
    with open(hf_hub_download(MODEL_REPO, "config.json", revision=MODEL_REVISION)) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def checkpoint_path() -> str:
    return hf_hub_download(MODEL_REPO, "model.safetensors", revision=MODEL_REVISION)


@pytest.fixture(scope="session")
def get_weight(checkpoint_path: str) -> Callable[..., torch.Tensor]:
    """get_weight(name, dtype, device) -> tensor. Reads only the named tensor, not the whole file."""
    handle = safe_open(checkpoint_path, framework="pt", device="cpu")

    def _get(name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return handle.get_tensor(name).to(device=device, dtype=dtype)

    return _get
