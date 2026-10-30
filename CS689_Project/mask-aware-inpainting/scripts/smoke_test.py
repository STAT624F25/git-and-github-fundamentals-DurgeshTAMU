"""Run a tiny infrastructure-only PyTorch forward/backward/checkpoint test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mask_aware_inpainting.checkpointing import load_checkpoint, save_checkpoint
from mask_aware_inpainting.config import (
    environment_metadata,
    load_config,
    make_experiment_id,
    resolve_device,
)
from mask_aware_inpainting.logging_utils import JsonlLogger, configure_logging
from mask_aware_inpainting.reproducibility import seed_everything


class SmokeModel(nn.Module):
    """Deliberately small stand-in; this is not an inpainting architecture."""

    def __init__(self, input_features: int, hidden_features: int, output_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, output_features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    experiment_id = make_experiment_id(config)
    output_dir = Path(config["output_dir"]) / experiment_id
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    logger = configure_logging(output_dir)
    events = JsonlLogger(output_dir / "events.jsonl")
    metadata = environment_metadata(device)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    logger.info("Starting smoke test on %s", device)
    events.log("start", experiment_id=experiment_id, **metadata)

    model = SmokeModel(
        int(config["input_features"]),
        int(config["hidden_features"]),
        int(config["output_features"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(int(config["batch_size"]), int(config["input_features"]), device=device)
    targets = torch.randn(int(config["batch_size"]), int(config["output_features"]), device=device)
    loss = nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()

    checkpoint_path = output_dir / str(config["checkpoint_name"])
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=1,
        config=config,
        metadata=metadata,
    )
    restored_model = SmokeModel(
        int(config["input_features"]),
        int(config["hidden_features"]),
        int(config["output_features"]),
    ).to(device)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint_metadata = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        map_location=device,
    )
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(expected, actual)
    assert checkpoint_metadata["step"] == 1
    events.log(
        "checkpoint_verified",
        checkpoint=str(checkpoint_path),
        loss=float(loss.detach().cpu()),
    )
    logger.info("Checkpoint save/reload verified: %s", checkpoint_path)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "smoke.yaml"))
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
