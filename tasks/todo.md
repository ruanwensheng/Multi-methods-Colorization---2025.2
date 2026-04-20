# Task List: Deep Learning Colorization Pipeline

> **Branch:** `feature/deep`
> **Last updated:** 2026-04-13

---

## Phase 0: Verify & Fix (Foundation)

- [ ] **T0.1** Run unit tests, fix any failures
  - `pytest tests/ -v --timeout=60`
  - All tests green (0 failures, 0 errors)

- [ ] **T0.2** Clean config.yaml and requirements.txt
  - config.yaml: remove scribble/example sections, keep only DL settings
  - requirements.txt: remove fastapi/uvicorn, fix character-encoding issue
  - Verify num_classes consistency with pts_in_hull.npy

- [ ] **T0.3** Rewrite notebook 04 for 5-model DL comparison
  - Replace cross-method content (scribble/example/DL)
  - New content: 5-model DL comparison (Zhang16 Pretrained/Fine-tuned, Zhang17, DeOldify, ControlNet)

**CHECKPOINT 0** — [ ] Tests green, config clean, notebook 04 corrected

---

## Phase 1: Data & Weights

- [ ] **T1.1** Download COCO 2017 (train + val + 500 benchmark)
  - `python tools/download_coco.py --split both --benchmark-size 500`
  - Verify: ~118K train, ~5K val, 500 benchmark images

- [ ] **T1.2** Download Zhang16 pretrained weights
  - `python tools/download_pretrained.py --model zhang16`
  - Verify: `models/pretrained/zhang16_eccv.pth` exists

**CHECKPOINT 1** — [ ] Data and weights in place

---

## Phase 2: Training

- [ ] **T2.1** Fine-tune Zhang16 on COCO 2017
  - `python tools/train_deep.py --config configs/config.yaml`
  - Verify: `models/deep_learning/best_model.pth` exists, MLflow run logged
  - _Blocked by: T1.1, T1.2_

**CHECKPOINT 2** — [ ] Trained model available

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
