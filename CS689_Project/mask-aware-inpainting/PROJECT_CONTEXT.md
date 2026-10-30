# Research Project Context and Execution Plan

**Title:** Mask-Aware Local–Global Attention for Efficient Image Inpainting  
**Course:** CSCE 689, Generative AI for Computer Vision (Fall 2026)  
**Team:** Individual/solo project  
**Purpose:** Implement controlled experiments for Midterm Report (Milestone 2), then expand for the final report.  
**Primary written source:** `Project_Report_CSCE689(2).pdf` (the submitted Milestone 1 proposal; place a local copy in `references/` if available). The proposal is the authority for existing commitments. This document provides implementation recommendations; explicitly distinguish proposed choices from fixed commitments.

## 1. Research question and hypothesis

**Research question:** Does image inpainting require every image token to perform global self-attention, or can we maintain similar reconstruction quality more efficiently by allowing only missing-region tokens to query globally over visible-region features?

**Hypothesis:** Global context primarily benefits masked tokens. Most visible features can be computed using efficient local/shifted-window processing. Letting masked queries perform global cross-attention over visible keys and values may preserve much of full attention's quality while reducing attention computation, particularly at low and moderate mask ratios. This is a hypothesis, not a confirmed result.

Core objectives:
- RQ1: Compare reconstruction quality of local-only, full-global, and proposed selective-global attention.
- RQ2: Compare theoretical and *measured* computational costs (latency, VRAM, throughput, FLOPs/MACs when practical).
- RQ3: Analyze how effects change with mask ratio and geometry.

The research contribution is the **asymmetric allocation of global queries** and a rigorous quality/efficiency comparison, not a claim that mask-aware attention, visible-feature retrieval, or window attention were invented here. In particular, contextual attention and MAT already perform related forms of mask-aware feature aggregation.

## 2. Architecture and math

For original image `x` and binary mask `M` (1=missing, 0=visible):

- Observed input: `x_obs = (1-M) * x`.
- Create patch embeddings and positional embeddings. Missing patches are **learned mask tokens**, not visible pixel features.
- Shared front end: efficient local/window or shifted-window attention, including local communication among masked tokens.
- Experimental attention stage: one of three alternatives below.
- Shared lightweight spatial decoder predicts RGB values; final image is `x_final = (1-M)*x + M*x_pred`.

Proposed global stage (per attention head):

`Q_M = H_M W_Q`, `K_V = H_V W_K`, `V_V = H_V W_V`

`Delta_H_M = softmax(Q_M K_V^T / sqrt(d_head)) V_V`.

Apply output projection/residual to masked features and scatter results back to their spatial token locations. Visible features are not globally updated in this stage. Masked tokens can interact locally before/after the global stage.

**Three primary controlled attention variants** (same patch embedding, common local feature extractor, shared decoder, width, depth, optimization protocol, and matched stage projections/FFN as far as practical):

- `local`: local/window-only attention at experimental stage.
- `full_global`: global self-attention over all N tokens at experimental stage.
- `selective_global`: masked queries attend to only visible keys/values at experimental stage.

Optional later baselines: CNN U-Net; MAT-style mask-aware window attention. These are separate references, not the primary mechanism-isolation experiment.

**Initial working configuration (subject to GPU feasibility; not mandated by proposal):** image 256x256; patch size 16 -> 16x16 grid, N=256 tokens; embedding width 384; 6 heads, 64 dim/head; 4x4-token local windows; 4 shared local/shifted-window Transformer blocks; 1 experimental attention block; shared convolutional upsampling decoder. Smaller models/128x128 may be used for debugging, but clearly record differences from final configuration.

**Complexity:** Full global stage has O(N^2) token-pair interactions; masked-to-visible global stage has O(K(N-K)) interactions with K masked tokens, *plus* local attention and all projections/decoder. This does not by itself imply an actual wall-clock speedup. At K=N/2 masked-to-visible interactions are maximal; at 75% masked ratio interactions decrease even though the reconstruction problem becomes harder.

## 3. Mask correctness and data leakage

Initial experiments should use patch-aligned binary masks: generate masks at token-grid resolution and expand to pixels with nearest-neighbor. Begin with rectangular masks and add irregular connected masks. Initial ratios 25% and 50%; eventual full study 10%, 25%, 50%, and 75%.

When extending to arbitrary pixel-level irregular masks, define handling of partially observed patches rigorously. **Never expose original missing pixels to the encoder by accident**, even inside partially masked patches. The model should see only `x_obs` and mask metadata, never the unmasked ground-truth image as input.

Training masks may vary; validation/test image-mask pairs must be *fixed and exactly reused across all model variants*. Preserve mask seeds/definitions. Group or pad variable masked/visible token counts correctly for batched attention. Handle empty masked or empty visible sets defensively.

## 4. Datasets and reproducibility

Primary experiment: a manageable natural-scene dataset/subset (ideally a properly accessible subset of Places2/Places365); the exact dataset and access terms must be verified before downloading. Alternative small image dataset is acceptable if documented. An initial 4k–8k-image experiment budget is a **proposal**, not a fixed requirement. Use a small (~100–500-image) subset for debugging and pilot training.

Use explicit deterministic train/validation/test manifests with no split leakage. Example pilot: 5000 train, 500 val, 500 test images if data permit. Choose and document actual counts before main runs. Keep test held out until comparisons are finalized. Train with varied masks; evaluate on fixed masks. Do not present fabricated quantitative metrics.

## 5. Training protocol

Suggested starting settings: AdamW; LR 1e-4; weight decay 1e-2; batch size 8 initially; mixed precision BF16 if available otherwise AMP/FP16; warmup + cosine learning-rate schedule; a fixed initial random seed. Adjust based on actual GPU memory and throughput. Use the same **number of optimizer steps**, dataset, mask distribution, augmentations, loss, and optimizer settings across the three attention variants.

Primary training loss is masked-region L1/MAE:

`L_mask = sum_{h,w,c} M[h,w] * abs(x[h,w,c]-x_pred[h,w,c]) / (C * sum_{h,w} M[h,w])`.

Optional later perceptual loss must be added consistently to all models if compared. Keep baseline experiment L1-only to isolate attention effects. Save checkpoint with model, optimizer, scheduler, grad scaler (if applicable), global step, seed, config, and RNG states where practical. Training must resume across Colab disconnects.

Training progression: (1) synthetic forward/backward and unit tests, (2) overfit 8–16 fixed images, (3) short pilot run and GPU profiling, (4) matched training of three variants, (5) controlled evaluation, (6) optional extensions.

## 6. Evaluation

Quality: masked-region L1, masked-region PSNR as primary numerical metric, whole-image PSNR, SSIM, LPIPS (full-image initially), and qualitative comparisons. For masked PSNR, divide MSE by number of masked pixel-channel entries; define intensity range consistently. Don't treat masked-image SSIM or LPIPS as ordinary region metrics without specifying and validating a protocol. Preserve original visible pixels in final composites.

Efficiency: stage interaction counts, parameter counts, full-model inference latency (synchronized GPU events / appropriate synchronization and warmup), peak CUDA memory, training throughput, and optionally FLOPs/MACs. Record hardware, PyTorch/CUDA, precision, batch size, image resolution, mask ratio. Benchmark *actual* gather/scatter and masking overhead; do not claim theoretical O(K(N-K)) guarantees practical speedup. Benchmark batch=1 latency and practical batch throughput separately.

Compare using **identical test images and masks**. Save reproducible CSV/JSON summaries, logs, plots, and reconstruction grids with filenames/configs tied to experiment IDs. Report negative and inconclusive results honestly.

## 7. Milestone 2 (midterm) requirements

Single PDF using a major conference paper template, suggested approximately 4 pages:
1. Introduction & Project Overview ~0.5 page: problem, motivation, scope, single researcher.
2. Literature Review ~0.5 page: contextual attention (Yu et al.), partial convolutions (Liu et al.), LaMa, MAT, TFill, T-former, CMT, ProPainter as relevant. Identify the gap carefully without exaggerating novelty.
3. Proposed Method ~1 page: implementable architecture with flow diagram, token/mask math, attention equation, decoding, loss, complexity, exact hyperparameters.
4. Preliminary Investigations ~2 pages: dataset/setup; actually completed pilot/baseline experiments; training curves; quantitative table; visual reconstructions; measured efficiency; mask analysis; failure cases; what preliminary findings imply for next steps.
5. References as needed. No code submission for this milestone (code still required to produce real experiments).

**Minimum credible midterm:** working/tested pipeline; local, full-global and selective-global variants; one fair comparison (prioritize masks 25/50%); actual quantitative quality + latency/VRAM measurements; example reconstructions; limitations and future plan. Do not invent results or imply an untrained model is complete.

Suggested report exhibits:
- architecture figure explaining asymmetric attention;
- reconstruction grid: ground truth / masked input / local / full global / selective;
- quality/efficiency table and, if helpful, training-curve or tradeoff plot.

## 8. Final report extensions

After Milestone 2: full 10/25/50/75% ratios; rectangular and irregular masks; repeat seeds; `masked_queries -> all_keys` vs `masked_queries -> visible_keys` ablation; vary windows/global blocks; CNN U-Net reference; possible MAT-style comparison; perceptual objective; larger/second dataset. Prioritize the most scientifically informative extensions based on midterm results. Keep the identical evaluation pipeline.

## 9. Computing and repository workflow

- Main editing environment: VS Code with Codex extension.
- Training environment: Google Colab Pro GPU accessed from VS Code / notebook.
- **Local VS Code terminal is not automatically Colab GPU.** Verify `torch.cuda.is_available()`, GPU name, and actual remote filesystem before GPU training.
- Keep Git repository as source of truth. Use a thin Colab runner notebook that syncs an explicit code revision and launches Python scripts; don't duplicate training logic inside the notebook.
- Use Colab local runtime disk for active workloads where feasible; persist checkpoints/results on Google Drive or other persistent storage, because Colab runtime storage is temporary.
- Never commit large datasets/checkpoints or secrets to Git. No secrets/service credentials in notebooks.
- Suggested code layout: `configs/`, `data/`, `models/`, `training/`, `evaluation/`, `scripts/`, `tests/`, `notebooks/`, `reports/milestone2/`. Keep the layout appropriately lightweight; don't create empty boilerplate unnecessarily.

## 10. Implementation stages and acceptance criteria

**Phase 1 — Infrastructure only.** Set up repository, dependency management, configs, reproducibility/logging, checkpoints, tests, and thin Colab runner. Provide dummy-model smoke test and instructions to verify remote GPU and persistence. **Do not implement the complete inpainting model or launch expensive training yet.** Acceptance: notebook can execute project modules, and checkpoint written to persistent storage can be reloaded.

**Phase 2 — Data/masks/shared backbone.** Implement data loaders/splits, mask generation, patch embedding, local/shifted attention, decoder, masked loss, unit tests. Acceptance: missing pixels never leak; exact visible-pixel preservation; forward/backward finite; tiny-subset overfit works.

**Phase 3 — Controlled variants.** Implement `local`, `full_global`, `selective_global`, matched training configs, validation snapshots. Acceptance: all three train for identical update budgets and produce restorable checkpoints.

**Phase 4 — Metrics and report exhibits.** Reproducible quality metrics, real synchronized efficiency benchmarks, plots, reconstruction grids, ablations. Acceptance: every number and figure in Report 2 traces to experiment artifacts.

**Phase 5 — Final report extensions.** Extend mask coverage, seeds, datasets, baselines and ablations according to findings.

## 11. Research integrity and project constraints

- Never invent performance results, citations, experimental outcomes, dataset access, or GPU/runtime availability.
- Preserve core research hypothesis; if architecture deviates from proposal, document clearly and get user approval for material changes.
- Don't conflate theoretical interactions with runtime/FLOPs or assume selective attention wins.
- Distinguish completed experiments from proposed future work.
- Make edits incrementally, test after each phase, summarize changed files, command outputs and blockers, and do not train large models without explicit instruction.
