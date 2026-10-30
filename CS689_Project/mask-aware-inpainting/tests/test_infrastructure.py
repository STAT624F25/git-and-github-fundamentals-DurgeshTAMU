from pathlib import Path

import torch
from torch import nn

from mask_aware_inpainting.checkpointing import load_checkpoint, save_checkpoint
from mask_aware_inpainting.config import make_experiment_id
from mask_aware_inpainting.reproducibility import seed_everything


def test_experiment_id_is_stable() -> None:
    config = {"experiment_name": "smoke", "seed": 689}
    assert make_experiment_id(config) == make_experiment_id(config)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    seed_everything(689)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 3)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=7, config={"seed": 689})

    restored = nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    metadata = load_checkpoint(path, restored, restored_optimizer)
    assert metadata["step"] == 7
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)
