"""Randomness controls and RNG state persistence."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state so a resumed run can continue reproducibly."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints loaded with map_location="cuda" can contain the CPU RNG
    # tensor on CUDA, but torch.set_rng_state requires a CPU tensor.
    torch.set_rng_state(state["torch"].detach().cpu())
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            rng_state.detach().cpu().to(dtype=torch.uint8)
            for rng_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
