# Mask-Aware Image Inpainting

Phase 1 provides project infrastructure only. It does not contain the
inpainting architecture, datasets, masks, or training loop. Those belong to
later phases in [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## Layout

- `configs/`: versioned YAML experiment configurations.
- `src/mask_aware_inpainting/`: reusable configuration, environment,
  reproducibility, logging, and checkpoint helpers.
- `scripts/smoke_test.py`: tiny CPU/GPU-safe forward/backward and checkpoint
  round-trip test; `SmokeModel` is not a research model.
- `tests/`: infrastructure tests.
- `notebooks/colab_runner.ipynb`: thin Colab entry point that installs and
  executes the Python script.
- `artifacts/`: local experiment output, ignored by Git.

## Local verification

From this directory, install the package and run the focused checks:

```powershell
python -m pip install -e . --no-deps
python scripts/smoke_test.py
python -m pytest tests/test_infrastructure.py -q
```

The smoke test records `config.json`, `environment.json`, `run.log`,
`events.jsonl`, and a checkpoint under `artifacts/smoke/<experiment-id>/`.
The environment record reports the actual PyTorch version and device; no GPU
is assumed.

## Colab execution

1. In Colab, select **Runtime > Change runtime type > T4 GPU** (or another
   available GPU) if GPU testing is desired.
2. Open `notebooks/colab_runner.ipynb`.
3. Set `REPO_URL` to the GitHub URL containing this project. The default
   matches the current repository layout:
   `STAT624F25/git-and-github-fundamentals-DurgeshTAMU`.
4. Run the cells in order. The first cell clones the source; the second prints
   PyTorch, CUDA availability, and the GPU name before installation; the third
   runs `scripts/smoke_test.py`.
5. `/content` is temporary. For later experiments, mount Google Drive and set
   a copied config's `output_dir` to a Drive directory before starting
   training. Do not commit datasets, checkpoints, credentials, or API keys.

The notebook deliberately does not download data or start model training.
After a successful run, verify that the printed checkpoint path exists and
that `events.jsonl` contains `start` and `checkpoint_verified` events.
