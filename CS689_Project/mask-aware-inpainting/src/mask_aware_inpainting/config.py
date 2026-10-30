"""Configuration and environment helpers shared by scripts and notebooks."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and fail clearly for malformed configurations."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return loaded


def make_experiment_id(config: Mapping[str, Any]) -> str:
    """Return a stable short ID for a configuration."""
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    name = str(config.get("experiment_name", "experiment")).replace(" ", "-")
    return f"{name}-{digest}"


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a requested device without assuming that Colab has a GPU."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Check the runtime type and "
            "torch.cuda.is_available() before starting a GPU experiment."
        )
    return torch.device(requested)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    """Collect execution details for reproducible experiment artifacts."""
    metadata: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        metadata["cuda_version"] = torch.version.cuda
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
    return metadata
