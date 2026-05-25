# Task List: Deep Learning Colorization Pipeline

> **Branch:** `feature/deep`
> **Last updated:** 2026-05-25 — Phases 0, 1, 2 complete. Best model: epoch 5, val_loss 3.23, PSNR 22.85, SSIM 0.914.

---

## Phase 0: Verify & Fix (Foundation)

- [x] **T0.1** Run unit tests, fix any failures
  - `pytest tests/ -v --timeout=60`
  - All tests green (113 passed, 0 failures) — includes 9 new guard tests added in T0.2 / T0.3

- [x] **T0.2** Clean config.yaml and requirements.txt
  - config.yaml: removed scribble/example_based sections; only DL settings remain
  - requirements.txt: already clean (no fastapi/uvicorn, no encoding artifacts) — no change needed
  - Added guard tests: `test_no_scribble_or_example_sections`, `test_only_expected_top_level_keys`

- [x] **T0.3** Rewrite notebook 04 for 5-model DL comparison
  - Replaced cross-method content (scribble/example/DL) with 5-model DL comparison scaffold
  - Cells load from `results/deep_learning/comparison/`, gracefully no-op until Phase 4 produces artifacts
  - Added `tests/test_notebook_04.py` (7 structural + content guards)

**CHECKPOINT 0** — [x] Tests green, config clean, notebook 04 corrected

---

## Phase 1: Data & Weights

- [x] **T1.1** Download COCO 2017 (train + val + test + 1,000-img benchmark from test2017)
  - `python tools/download_coco.py --source local-zip --split all --benchmark-size 1000`
  - Verified: val2017 = 5,000, test2017 = 40,670, benchmark = 1,000 (subset of test2017)

- [x] **T1.2** Download Zhang16 pretrained weights
  - `python tools/download_pretrained.py --model zhang16`
  - Verified: `models/pretrained/zhang16_eccv.pth` (123 MB), `models/pretrained/zhang16_siggraph.pth` (130.5 MB)

**CHECKPOINT 1** — [x] Data and weights in place

---

## Phase 2: Training

- [x] **T2.1** Fine-tune Zhang16 on COCO 2017
  - `python tools/train_deep.py --config configs/config.yaml --epochs 5 --max-train-batches 2000`
  - Settings: AMP off + batch_size=4 (the AMP=True path NaN'd at iter 1649 with fp16 overflow; see commits 412f48c, 2de761b)
  - Results: 5 epochs × 2000 batches, ~13 min/epoch + ~6 min/val
  - Best: epoch 5, val_loss 3.23, PSNR 22.85 dB, SSIM 0.914 — saved to `models/deep_learning/best_model.pth`
  - MLflow run: `1753768c996c473e989676f3ef3a1d1a` in experiment `deep-colorization`
  - Note: epoch 2 val_loss spiked to 16.85 (one bad batch); training otherwise monotonically improved

**CHECKPOINT 2** — [x] Trained model available

---

## Phase 3: Evaluation

- [ ] **T3.1** Evaluate Zhang16 Pretrained on benchmark
  - `python tools/evaluate_deep.py --model-path models/pretrained/zhang16_eccv.pth --tag pretrained`
  - _Blocked by: T1.1, T1.2_

- [ ] **T3.2** Evaluate Zhang16 Fine-tuned on benchmark
  - `python tools/evaluate_deep.py --model-path models/deep_learning/best_model.pth --tag finetuned`
  - _Blocked by: T2.1_

- [ ] **T3.3** Install & evaluate Zhang17 (Interactive CNN)
  - Install: `pip install colorizers`
  - _Blocked by: T1.1_

- [ ] **T3.4** Install & evaluate DeOldify (GAN)
  - Install: `pip install deoldify fastai`
  - _Blocked by: T1.1_

- [ ] **T3.5** Install & evaluate ControlNet (Diffusion)
  - Install: `pip install diffusers transformers accelerate`
  - Needs RTX 2080 Ti (~10GB VRAM)
  - _Blocked by: T1.1_

**CHECKPOINT 3** — [ ] All 5 models evaluated with metrics

---

## Phase 4: Comparison & Visualization

- [ ] **T4.1** Run 5-model comparison
  - `python tools/compare_methods.py --max-images 50`
  - Output: comparison_summary.json, comparison_metrics.csv, visual grids
  - _Blocked by: T3.1-T3.5_

- [ ] **T4.2** Generate result figures
  - metrics_bar_chart.png, qualitative_grid.png, training_curves.png
  - _Blocked by: T4.1_

**CHECKPOINT 4** — [ ] All results and figures ready

---

## Phase 5: LaTeX Report

- [ ] **T5.1** Create report directory structure + main.tex + references.bib
  - `reports/deep_learning/` with IEEEtran format, 8 BibTeX entries
  - _No blockers — can start anytime_

- [ ] **T5.2** Write introduction.tex + related_work.tex
  - Problem statement, 4-paradigm taxonomy, all citations
  - _No blockers — can start anytime_

- [ ] **T5.3** Write method.tex
  - Zhang16 architecture, loss equations (Eq. 2-4), quantization, training
  - _No blockers — can start anytime_

- [ ] **T5.4** Write experiments.tex + results.tex
  - COCO 2017, 5 models, hyperparameters, metrics table, qualitative grids
  - _Blocked by: Phase 4 (needs actual numbers and figures)_

- [ ] **T5.5** Write discussion.tex + conclusion.tex
  - Trade-off analysis, recommendations
  - _Blocked by: T5.4_

- [ ] **T5.6** Compile final PDF
  - `pdflatex main && bibtex main && pdflatex main && pdflatex main`
  - _Blocked by: T5.5_

**CHECKPOINT 5** — [ ] Report compiles to clean PDF

---

## Phase 6: Final Verification

- [ ] **T6.1** Verify notebook 03 runs end-to-end
  - _Blocked by: Phase 4_

- [ ] **T6.2** Verify notebook 04 runs end-to-end
  - _Blocked by: Phase 4_

- [ ] **T6.3** Final acceptance criteria audit (SPEC section 12)
  - Walk through all 17 checklist items
  - _Blocked by: T6.1, T6.2, T5.6_

**CHECKPOINT 6** — [ ] Branch ready to merge into `main`

---

## Quick Reference: What Can Run In Parallel

```
Parallel Group A (Phase 0):  T0.1 | T0.2 | T0.3
Parallel Group B (Phase 1):  T1.1 | T1.2
Parallel Group C (Phase 3):  T3.1 | T3.3 | T3.4 | T3.5  (T3.2 waits for T2.1)
Parallel Group D (Phase 5):  T5.1 | T5.2 | T5.3  (can start during Phase 2-3)
Parallel Group E (Phase 6):  T6.1 | T6.2
```
