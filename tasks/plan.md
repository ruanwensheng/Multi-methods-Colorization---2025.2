# Implementation Plan: Deep Learning Colorization Pipeline

> **Branch:** `feature/deep`
> **Created:** 2026-04-13
> **Last updated:** 2026-05-19 — switched evaluation benchmark from val2017 subset to test2017 subset (1000 imgs, seed=42) to remove val/test contamination. See SPEC.md §6.3.
> **Status:** Approved

---

## Current State Assessment

### What's DONE (code exists, fully implemented)
- **Core module** (`src/deep_learning/`): 9 files, ~2000 LOC — Zhang16Net, quantizer, loss, training loop, inference API, pretrained wrappers
- **CLI tools** (`tools/`): 5 scripts — download_coco, download_pretrained, train_deep, evaluate_deep, compare_methods
- **Unit tests** (`tests/`): 9 files, ~980 LOC — covers all modules
- **Config** (`configs/config.yaml`): DL hyperparameters defined
- **Workflow** (`workflows/deep_learning_pipeline.md`): Step-by-step SOP
- **Notebook 03** (`03_deep_learning_pipeline.ipynb`): Full pipeline walkthrough (18 KB, well-structured)
- **Reference papers** (`references/`): All 5 theory papers present (58 MB)
- **Infrastructure**: requirements.txt, pytest.ini, .gitignore all configured

### What's MISSING
1. **LaTeX report** — entire `reports/deep_learning/` directory (SPEC section 8)
2. **Execution artifacts** — no COCO data, pretrained weights, trained models, evaluation metrics, or comparison outputs
3. **Notebook 04 content mismatch** — currently implements cross-method (scribble/example/DL) comparison, but SPEC says it should show the 5-model DL comparison. Needs rewrite.

### Issues Found
| Issue | Location | Impact |
|-------|----------|--------|
| Config has scribble/example sections | `config.yaml` lines 24-33 | SPEC says "only deep learning settings" |
| Notebook 04 is cross-method comparison | `04_method_comparison.ipynb` | Should be 5-model DL comparison per SPEC |
| num_classes: 233 in config vs 313 in SPEC/code | `config.yaml` line 37 | Code uses `quantizer.num_bins` dynamically from `pts_in_hull.npy` — config value is fallback only. Need to verify consistency |
| requirements.txt has fastapi/uvicorn + encoding issue | `requirements.txt` lines 33-34 | SPEC says "Only DL-relevant dependencies (no FastAPI/uvicorn)". File also has spaces between every character (encoding bug). |
| Tests not yet verified passing | `tests/` | Must pass before any execution phase |

---

## Dependency Graph

```
Phase 0: Verify & Fix ──────────────────────┐
  T0.1 Run tests, fix failures              │
  T0.2 Clean config.yaml                    │
  T0.3 Rewrite notebook 04                  │
                                             ▼
Phase 1: Data & Weights ────────────────────┐
  T1.1 Download COCO 2017 ──────────┐       │
  T1.2 Download pretrained weights ──┤       │
                                     ▼       │
Phase 2: Training ──────────────────────────┐│
  T2.1 Fine-tune Zhang16 on COCO           ││
                                     ┌──────┘│
                                     ▼       ▼
Phase 3: Evaluation (parallel per model) ───┐
  T3.1 Eval Zhang16 Pretrained              │
  T3.2 Eval Zhang16 Fine-tuned              │ ← blocked by T2.1
  T3.3 Eval Zhang17 (Interactive CNN)       │
  T3.4 Eval DeOldify (GAN)                 │
  T3.5 Eval ControlNet (Diffusion)          │
                                             ▼
Phase 4: Comparison & Visualization ────────┐
  T4.1 Run compare_methods.py               │
  T4.2 Generate figures                      │
                                             ▼
Phase 5: LaTeX Report ─────────────────────┐│
  T5.1 Create structure + references.bib   ││ ← can start early
  T5.2 Write intro + related_work          ││ ← can start early
  T5.3 Write method (Zhang16 architecture) ││ ← can start early
  T5.4 Write experiments + results         ││ ← blocked by Phase 4
  T5.5 Write discussion + conclusion       ││ ← blocked by T5.4
  T5.6 Compile PDF                         ││
                                            ▼▼
Phase 6: Final Verification ────────────────
  T6.1 Verify notebook 03 end-to-end
  T6.2 Verify notebook 04 end-to-end
  T6.3 Final acceptance criteria audit
```

---

## Phase 0: Verify & Fix (Foundation)

**Goal:** Ensure existing code works and fix known issues before execution.

### T0.1 — Run unit tests, fix failures
- **Command:** `conda activate AI && pytest tests/ -v --timeout=60`
- **Acceptance:** All tests pass (0 failures, 0 errors)
- **Verification:** Green pytest output

### T0.2 — Clean config.yaml and requirements.txt
- **config.yaml:** Remove `scribble:` and `example_based:` sections. Verify `num_classes` consistency with `pts_in_hull.npy`.
- **requirements.txt:** Remove `fastapi` and `uvicorn` lines. Fix character-encoding issue (spaces between every character).
- **Why:** SPEC acceptance criteria: "Config: only deep learning settings" and "Requirements: Only DL-relevant dependencies (no FastAPI/uvicorn)"
- **Acceptance:** Config has only `project`, `paths`, `image`, `deep_learning`, `evaluation`, `device`. Requirements has no fastapi/uvicorn, no encoding artifacts.
- **Verification:** `python -c "from src.deep_learning.utils import load_config; cfg = load_config('configs/config.yaml'); print('OK')"` and `pip install -r requirements.txt --dry-run`

### T0.3 — Rewrite notebook 04 for 5-model DL comparison
- **What:** Replace cross-method content with 5-model DL comparison (Zhang16 Pretrained, Zhang16 Fine-tuned, Zhang17, DeOldify, ControlNet)
- **Why:** Current content compares scribble/example/DL — that belongs on `main`, not this branch
- **Structure:** Load comparison results, show metrics table, bar charts, qualitative grids, per-model analysis
- **Acceptance:** Notebook references all 5 DL models, loads from `results/deep_learning/comparison/`, produces visualizations
- **Verification:** Notebook cells run without import errors (data-dependent cells can show "no data" gracefully)

**CHECKPOINT 0:** Tests green, config clean, notebook 04 corrected. No execution artifacts yet.

---

## Phase 1: Data & Weights (Prerequisites)

**Goal:** Download all required data and pretrained weights.

### T1.1 — Download COCO 2017 dataset
- **Command:** `python tools/download_coco.py --source local-zip --split all --benchmark-size 1000`
  (or `--source kaggle` / `--source http` if `archive.zip` isn't present)
- **Downloads/extracts:** train2017 (~18 GB, 118K imgs), val2017 (~0.8 GB, 5K imgs),
  test2017 (~6.2 GB, 40,670 imgs), annotations (~0.8 GB, 6 JSONs),
  and a 1,000-image benchmark subset of **test2017** (seed=42).
- **Why test2017 for benchmark:** val2017 is used by `Trainer.fit` for best-ckpt selection;
  using a val2017 subset as the test set would leak. test2017 is COCO's held-out split. See SPEC §6.3.
- **Risk:** Long extraction; idempotent on re-run.
- **Acceptance:** train2017 ≈118,287, val2017 = 5,000, test2017 = 40,670, benchmark = 1,000.
- **Verification:** `python tools/_verify_dataset.py` (after extending it to print test2017 + check that
  benchmark is a subset of test2017, not val2017).

### T1.2 — Download pretrained weights
- **Command:** `python tools/download_pretrained.py --model zhang16`
- **Downloads:** `models/pretrained/zhang16_eccv.pth`, `models/pretrained/zhang16_siggraph.pth`
- **Acceptance:** Both .pth files exist and are non-zero size
- **Verification:** `ls -la models/pretrained/*.pth`

**CHECKPOINT 1:** Data and weights in place. Ready for training.

---

## Phase 2: Training

**Goal:** Fine-tune Zhang16 on COCO 2017, produce best checkpoint.

### T2.1 — Fine-tune Zhang16 on COCO 2017
- **Command:** `python tools/train_deep.py --config configs/config.yaml`
- **Key params:** 50 epochs, batch 8, lr 0.0002, Adam, StepLR(20, 0.5), AMP enabled, class-rebalanced CE loss
- **Output:** `models/deep_learning/best_model.pth`, `models/deep_learning/last_model.pth`
- **MLflow:** Experiment `deep-colorization` with logged params, per-epoch metrics, best model artifact
- **Risk:** VRAM overflow on GTX 1660 → reduce batch_size to 4 or 2. Training may take hours.
- **Acceptance:** best_model.pth exists, MLflow run has training curves (train_loss decreasing, val_loss not diverging)
- **Verification:** `python -c "import torch; m = torch.load('models/deep_learning/best_model.pth', map_location='cpu'); print(type(m))"`

**CHECKPOINT 2:** Trained model available. Ready for evaluation.

---

## Phase 3: Evaluation (Per Model)

**Goal:** Compute PSNR, SSIM, LPIPS for all 5 models on the **1,000-image test2017 benchmark** (seed=42).
The benchmark images were never seen during fine-tuning or best-checkpoint selection.

### T3.1 — Evaluate Zhang16 Pretrained
- **Command:** `python tools/evaluate_deep.py --model-path models/pretrained/zhang16_eccv.pth --tag pretrained`
- **Output:** `results/deep_learning/metrics/zhang16_pretrained_metrics.json`
- **Acceptance:** JSON has psnr_mean, ssim_mean, lpips_mean fields with numeric values

### T3.2 — Evaluate Zhang16 Fine-tuned
- **Blocked by:** T2.1
- **Command:** `python tools/evaluate_deep.py --model-path models/deep_learning/best_model.pth --tag finetuned`
- **Output:** `results/deep_learning/metrics/zhang16_finetuned_metrics.json`
- **Acceptance:** Same as T3.1. Fine-tuned PSNR should be >= pretrained (or close).

### T3.3 — Evaluate Zhang17 (Interactive CNN)
- **Prerequisites:** `pip install colorizers` (or manual weight download per pretrained.py fallback)
- **Command:** `python tools/evaluate_deep.py --model zhang17 --tag zhang17`
- **Output:** `results/deep_learning/metrics/zhang17_metrics.json`
- **Risk:** `colorizers` package may have compatibility issues. pretrained.py has S3 fallback.
- **Acceptance:** JSON with metrics or documented "N/A" with reason

### T3.4 — Evaluate DeOldify (GAN)
- **Prerequisites:** `pip install deoldify fastai`
- **Command:** `python tools/evaluate_deep.py --model deoldify --tag deoldify`
- **Output:** `results/deep_learning/metrics/deoldify_metrics.json`
- **Risk:** DeOldify + fastai dependency conflicts. May need separate env (document in workflow).
- **Acceptance:** JSON with metrics or documented "N/A"

### T3.5 — Evaluate ControlNet (Diffusion)
- **Prerequisites:** `pip install diffusers transformers accelerate`, ~10GB VRAM (RTX 2080 Ti)
- **Command:** `python tools/evaluate_deep.py --model controlnet --tag controlnet`
- **Output:** `results/deep_learning/metrics/controlnet_metrics.json`
- **Risk:** Needs RTX 2080 Ti. Very slow (30 inference steps per image). May need `--max-images 50`.
- **Acceptance:** JSON with metrics or documented "N/A"

**CHECKPOINT 3:** All 5 models evaluated. Individual metrics available.

---

## Phase 4: Comparison & Visualization

**Goal:** Cross-model comparison table and visual outputs.

### T4.1 — Run compare_methods.py
- **Command:** `python tools/compare_methods.py --max-images 50`
- **Output:**
  - `results/deep_learning/comparison/comparison_summary.json`
  - `results/deep_learning/comparison/comparison_metrics.csv`
  - `results/deep_learning/comparison/comparison_grids/` (visual grids)
- **Acceptance:** Summary JSON has all 5 models, CSV has per-image rows

### T4.2 — Generate figures
- **Output:**
  - `results/deep_learning/figures/metrics_bar_chart.png`
  - `results/deep_learning/figures/qualitative_grid.png`
  - `results/deep_learning/figures/training_curves.png`
- **Acceptance:** All 3 PNG files exist and are visually correct
- **Note:** training_curves.png may need to be generated from MLflow data separately

**CHECKPOINT 4:** All quantitative and qualitative results ready. Report can be written.

---

## Phase 5: LaTeX Report

**Goal:** Write the DL method report per SPEC section 8 (IEEE format).

### T5.1 — Create report directory structure + references.bib
- **Create:** `reports/deep_learning/` with `main.tex`, `sections/`, `figures/`, `references.bib`
- **main.tex:** IEEEtran class, `\input{}` for each section
- **references.bib:** All 8 required BibTeX entries (Zhang16, Zhang17, ChromaGAN, DeOldify, Palette, ControlNet, COCO, LPIPS)
- **Acceptance:** `pdflatex main.tex` runs without missing-file errors (sections can be empty stubs)

### T5.2 — Write introduction + related_work
- **Can start early** (no dependency on execution results)
- **introduction.tex:** Problem statement, motivation for DL approaches, chapter overview
- **related_work.tex:** 4-paradigm taxonomy (CNN → Interactive → GAN → Diffusion), cite all 6 papers
- **Acceptance:** Compiles, citations resolve, ~2-3 pages combined

### T5.3 — Write method (Zhang16 architecture)
- **Can start early** (architecture is known)
- **method.tex:** Network diagram reference, Eq. 2-4 (loss), ab quantization, annealed-mean decoding, training procedure
- **Acceptance:** Compiles, equations render, ~2 pages

### T5.4 — Write experiments + results
- **Blocked by:** Phase 4 (needs actual metric numbers and figures)
- **experiments.tex:** COCO 2017 description, 5 models table, 3 metrics, hardware specs, hyperparameters
- **results.tex:** Quantitative table (PSNR/SSIM/LPIPS per model), qualitative grids, analysis
- **Acceptance:** Compiles, figures referenced, ~3 pages combined

### T5.5 — Write discussion + conclusion
- **Blocked by:** T5.4
- **discussion.tex:** Speed vs quality trade-offs, saturation analysis, failure cases, paradigm strengths/weaknesses
- **conclusion.tex:** Summary, practical recommendations
- **Acceptance:** Compiles, ~1.5 pages combined

### T5.6 — Compile final PDF
- **Command:** `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- **Acceptance:** `main.pdf` renders cleanly with all sections, tables, figures, citations, no warnings

**CHECKPOINT 5:** Report complete. Ready for final verification.

---

## Phase 6: Final Verification

**Goal:** End-to-end validation against SPEC acceptance criteria.

### T6.1 — Verify notebook 03 runs end-to-end
- Run all cells in `03_deep_learning_pipeline.ipynb`
- **Acceptance:** No errors, visualizations render, training/evaluation sections work with real data

### T6.2 — Verify notebook 04 runs end-to-end
- Run all cells in `04_method_comparison.ipynb`
- **Acceptance:** No errors, loads comparison results, shows 5-model metrics and grids

### T6.3 — Final acceptance criteria audit
- Walk through all 17 items in SPEC section 12
- **Acceptance:** All boxes checked, branch ready to merge

---

## Parallelism Opportunities

| Tasks that can run in parallel | Notes |
|-------------------------------|-------|
| T0.1, T0.2, T0.3 | All independent foundation fixes |
| T1.1 and T1.2 | Independent downloads |
| T3.1, T3.3, T3.4, T3.5 | Independent model evaluations (T3.2 waits for T2.1) |
| T5.1, T5.2, T5.3 | Report structure + intro sections don't need results |
| T6.1 and T6.2 | Independent notebook verification |

## Time Estimates (Rough)

| Phase | Wall Clock | Notes |
|-------|-----------|-------|
| Phase 0 | 30 min | Code fixes, fast |
| Phase 1 | 1-4 hours | Download speed dependent (18GB COCO) |
| Phase 2 | 4-12 hours | GPU training time (50 epochs on 118K images) |
| Phase 3 | 2-6 hours | ControlNet is slowest (~1 min/image × 500 images) |
| Phase 4 | 1-2 hours | compare_methods on 50 images |
| Phase 5 | 3-5 hours | Writing + compiling LaTeX |
| Phase 6 | 1 hour | Verification |

**Total: ~12-30 hours** depending on GPU speed and download bandwidth.

## Risks & Contingencies

| Risk | Probability | Contingency |
|------|------------|-------------|
| Tests fail | Medium | Fix issues discovered; may need code changes |
| COCO download fails/slow | Low | Idempotent script; retry on university network |
| Training diverges | Low | Reduce LR, increase batch norm momentum, check data pipeline |
| VRAM overflow (GTX 1660) | Medium | Reduce batch_size to 4→2, enable gradient checkpointing |
| DeOldify deps conflict | High | Install in separate conda env, document in workflow |
| ControlNet too slow / OOM | High | Use RTX 2080 Ti, reduce max-images to 50, or mark N/A |
| LaTeX compilation fails | Low | Use basic IEEEtran template, fix one error at a time |
