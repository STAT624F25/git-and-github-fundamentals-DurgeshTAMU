# Codex Instructions — Mask-Aware Image Inpainting Research

Read `PROJECT_CONTEXT.md` in the repository root **before changing code**. If `references/Project_Report_CSCE689(2).pdf` is present, consult it for the original Milestone 1 research commitments. The project is a solo course research project. The priority is valid, reproducible, controlled experiments for Midterm Report 2 and final report.

## Engineering rules

1. Work incrementally; do not implement or train the entire project in a single task. Respect the current phase and exit criteria in `PROJECT_CONTEXT.md`.
2. Keep the `local`, `full_global`, and `selective_global` experiments comparable: shared backbone and decoder, matched hyperparameters, data splits/masks and optimization steps. Clearly document unavoidable parameter/computation differences.
3. Prevent ground-truth leakage through masks and partial patches; test it explicitly. Use mask convention `1=missing`, `0=visible` everywhere. Ensure final output preserves visible pixels exactly.
4. Use configuration files, fixed seeds, stable experiment IDs, train/validation/test split manifests, persistent checkpoints, and reproducible metrics. Keep a thin Colab notebook; put reusable logic in Python modules.
5. Do not assume VS Code's local terminal is the Colab GPU. Display and log the executing machine's actual device, storage paths, and framework versions. Never assume GPU type or Colab access/availability.
6. Prefer small, testable changes. Run relevant unit tests after implementing a feature. Include a tiny overfit test before substantive training.
7. Avoid expensive GPU experiments, downloading large datasets, or modifying/deleting checkpoints without explicit user approval. Do not expose API keys or commit private datasets, datasets or large checkpoints.
8. Never fabricate experiments, metrics, references, plots or claims. Use `TBD` for pending results and mark preliminary work clearly.
9. When finished, explain files changed, tests executed and their outcomes, reproducible run commands, assumptions, and what is next. Escalate genuine blockers; otherwise make sensible defaults and document them.

## Current task scope

Start with **Phase 1: infrastructure and Colab execution only**. Set up a lean repository and demonstrate a minimal PyTorch smoke test with checkpoint save/reload and a working remote execution path. Do NOT start model training or implement the complete architecture unless the user explicitly advances to Phase 2.
