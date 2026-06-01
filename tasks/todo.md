# Task List: Deep Learning Colorization Pipeline

> **Branch:** `feature/deep`
> **Last updated:** 2026-06-02 — **All six phases complete; CHECKPOINT 6 reached.** Phase 6: notebook 03 patched (skip-if-trained guard + small fixes) and verified end-to-end via `nbconvert`; notebook 04 verified end-to-end against canonical Phase-4 artifacts; SPEC §12 audit walked. The scope was subsequently reduced from four paradigms to three: the diffusion slot (ControlNet + SD 2.1) was excluded after the upstream SD 2.x deprecation made every available open-source colorization checkpoint unusable — see `reports/deep_learning/sections/experiments.tex`. `reports/deep_learning/Report.pdf` committed. Branch ready to merge.

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

- [x] **T2.1** Fine-tune Zhang16 on COCO 2017 — 5 capped + 3 full-data epochs total
  - **Capped pass (epochs 1-5):** `python tools/train_deep.py --config configs/config.yaml --epochs 5 --max-train-batches 2000`
    - AMP off + batch_size=4 (the AMP=True path NaN'd at iter 1649 with fp16 overflow; see commits 412f48c, 2de761b)
    - ~13 min/epoch + ~6 min/val. End of epoch 5: val_loss 3.23, val_psnr 22.85 dB.
    - MLflow run: `1753768c996c473e989676f3ef3a1d1a`
  - **Full-data resume (epochs 6-8):** two sessions, both `--resume` from prior `last_model.pth`, no batch cap
    - 29,571 batches/epoch × bs=4 ≈ 118K image-views/epoch (~2 hr/epoch on GTX 1660 SUPER, AMP off)
    - Epoch 6: val_loss 3.17, val_psnr 23.24 → won best-checkpoint
    - Epoch 7: val_loss 13.58 (single pathological batch — outlier-filtered, see commit 50d798d), val_psnr 23.38
    - Epoch 8: val_loss 3.62, val_psnr **23.64 dB**, val_ssim 0.917
    - MLflow runs: `4bd4839f35a7460cb5108236799f838b`, `c3d3c19ce5034990b63a2664e69d1f9c`
  - Final artifacts: `models/deep_learning/best_model.pth` (epoch 6, lowest val_loss), `last_model.pth` (epoch 8, most-trained)

**CHECKPOINT 2** — [x] Trained model available

---

## Phase 3: Evaluation

- [x] **T3.1** Evaluate Zhang16 Pretrained on benchmark
  - `python tools/evaluate_deep.py --model-path models/pretrained/zhang16_eccv.pth --tag pretrained`
  - **PSNR 17.14 [16.97, 17.30]**, SSIM 0.700, LPIPS 0.438. **Misleading number**: our 233-bin head can't accept the official 313-bin ECCV head, so this measures "ECCV encoder + RANDOM head" — not a real Zhang16 baseline.

- [x] **T3.2** Evaluate Zhang16 Fine-tuned on benchmark
  - `python tools/evaluate_deep.py --model-path models/deep_learning/last_model.pth --tag finetuned`
  - **Final (epoch 8 `last_model.pth`, after 5 capped + 3 full-data resume epochs): PSNR 23.29 [23.02, 23.55], SSIM 0.919 [0.915, 0.923], LPIPS 0.192 [0.187, 0.197]**
  - First pass (5 capped epochs only, 40K image-views) was PSNR 22.97; +0.32 dB lift from the ~355K additional full-data image-views. CIs are disjoint → real improvement, but smaller than the 24-25 dB target — train_loss curve had already flattened (3.05 → 3.00 across epochs 7-8) so further full epochs likely yield diminishing returns.

- [x] **T3.3** Install & evaluate Zhang17 (Interactive CNN, automatic mode)
  - Vendored `colorizers` package at `src/vendor/colorizers/` (BSD-licensed, ~30 KB; not on PyPI).
  - `python tools/evaluate_deep.py --method zhang2017 --tag zhang17`
  - **PSNR 18.82 [18.64, 19.00]**, SSIM 0.832, LPIPS 0.309. Below Zhang16 because Zhang17 was designed for *user-guided* mode; in zero-hint auto mode it underperforms.

- [x] **T3.4** Install & evaluate DeOldify (GAN)
  - `pip install deoldify` (PyPI), `pip install fastprogress`, `pip install "setuptools<81"` (deoldify needs pkg_resources).
  - Downloaded `ColorizeArtistic_gen.pth` (243 MB) into `models/`.
  - `python tools/evaluate_deep.py --method deoldify --tag deoldify`
  - **PSNR 24.00 [23.75, 24.25]**, SSIM 0.919, LPIPS 0.148. SOTA-class anchor.

> **Note on the originally-planned diffusion slot (T3.5).** The original plan
> included a fourth paradigm — Diffusion (ControlNet + SD 2.1). That slot was
> dropped before the comparison was finalised: Stability AI deprecated the
> entire SD 2.x line upstream (all `stabilityai/stable-diffusion-2*` URLs return
> HTTP 401), and every open-source colorization checkpoint we surveyed is
> anchored to SD 2.1. No alternative pipeline tested (SD 1.5 brightness-ControlNet,
> SD-InstructPix2Pix, Flux) produced a faithful colorization under the
> benchmark's identical-conditions constraint. The diffusion paradigm is therefore
> excluded from the benchmark entirely rather than represented by a degraded
> substitute. The exclusion rationale lives in
> `reports/deep_learning/sections/experiments.tex` (Sec. "Why diffusion is
> excluded from the comparison").

**CHECKPOINT 3** — [x] Four-model comparison evaluated with metrics on the 1,000-image test2017 benchmark. Phase 4 (cross-model comparison + figures) unblocked.

| Model | PSNR (95% CI) | SSIM | LPIPS | Time/img |
|---|---|---|---|---|
| Zhang16 Pretrained (broken head — see T3.1) | 17.14 [16.97, 17.30] | 0.700 | 0.438 | 0.17s |
| Zhang17 (auto mode) | 18.82 [18.64, 19.00] | 0.832 | 0.309 | 0.22s |
| **Zhang16 Fine-tuned (ours)** | **23.29 [23.02, 23.55]** | **0.919** | **0.192** | 0.18s |
| DeOldify (GAN, SOTA-class anchor) | 24.00 [23.75, 24.25] | 0.919 | 0.148 | 0.51s |

---

## Phase 4: Comparison & Visualization

- [x] **T4.1** Build 4-model comparison from canonical metrics
  - `python tools/compare_methods.py --from-metrics`
  - **Methodology change from plan**: the plan's literal `--max-images 50` re-run
    would have produced a *second, weaker* set of numbers (50-img subset, default
    `best_model.pth` ep6 not the canonical `last_model.pth` ep8). Instead
    `--from-metrics` aggregates the existing per-model `evaluate_deep.py` outputs
    (full 1,000-image benchmark), keeping the single source of truth — see
    `docs/benchmark_methodology.md`.
  - Output: `comparison/comparison_summary.json` (4 models),
    `comparison/comparison_metrics.csv` (4,000 per-image rows, `model` column).
    Numbers match the T3 table exactly.

- [x] **T4.2** Generate result figures (`results/deep_learning/figures/`)
  - `metrics_bar_chart.png` — PSNR/SSIM/LPIPS with 95% bootstrap CI error bars.
  - `training_curves.png` — train/val loss + val PSNR over 8 epochs; the two
    val-loss outliers (ep 2, 7) are masked off the line and annotated.
  - `qualitative_grid.png` — `--qualitative-grid 5`; Input | Zhang16 (Ours) |
    Zhang 2017 | DeOldify | GT over 5 benchmark images. Run in conda env `AI`.

**CHECKPOINT 4** — [x] All results and figures ready. Comparison artifacts + 3 figures
generated from the 1,000-image benchmark; consistent with the Phase-3 numbers.

---

## Phase 5: LaTeX Report

- [x] **T5.1** Create report directory structure + main.tex + references.bib
  - `reports/deep_learning/` (IEEEtran conference class), references.bib with all 8 entries.
  - Figures: `architecture.png` + `ab_quantization.png` generated from the actual model/quantizer
    (confirmed Q=233 in-gamut bins); `training_curves/metrics_comparison/qualitative_grid` copied from results.
  - Verified by `tests/test_report.py` (structural guards + a cite/figure lint standing in for the compiler).

- [x] **T5.2** Write introduction.tex + related_work.tex
  - Problem statement (ill-posed/multimodal, classification vs regression), 4-paradigm taxonomy table, all citations.

- [x] **T5.3** Write method.tex
  - Zhang16 architecture (Fig.), class-rebalanced CE loss + rebalancing weight + annealed-mean decode (T=0.38), Q=233 quantization, training.

- [x] **T5.4** Write experiments.tex + results.tex
  - COCO 2017, leakage-safe 1,000-img test2017 benchmark, hyperparameter table, metrics table with 95% CIs, bar chart + qualitative grid. Numbers match Phase-3/4 exactly.

- [x] **T5.5** Write discussion.tex + conclusion.tex
  - Speed/quality trade-offs, saturation/failure analysis, diffusion-exclusion-as-finding, per-paradigm recommendations.

- [~] **T5.6** Compile final PDF — **handed off to user (no LaTeX engine in this env)**
  - `pdflatex` / `bibtex` / `tectonic` / `latexmk` are all absent locally; user opted to compile on their own toolchain.
  - Compile: `cd reports/deep_learning && pdflatex main && bibtex main && pdflatex main && pdflatex main`
    (or upload the folder to Overleaf). Structural integrity already validated by `tests/test_report.py`.

**CHECKPOINT 5** — [~] Report authored & structurally verified; final PDF compile pending the user's LaTeX toolchain.

---

## Phase 6: Final Verification

- [x] **T6.1** Verify notebook 03 runs end-to-end
  - Patched cell 14 with a *skip-if-`best_model.pth`-exists* guard so the demo notebook is
    re-runnable without re-doing 50 epochs (it now defers to `tools/train_deep.py` for
    real training). Fixed a typo (`total_mem` → `total_memory`) in cell 1 and the stale
    "scribble/example-based" mention in cell 21 (Summary).
  - `tests/test_notebook_03.py` (new, 7 structural + 1 execution-gated): pipeline-stage
    coverage, config-from-YAML, locked-in skip guard, and full-notebook execution via
    `nbconvert` (gated on AI env). End-to-end execution: **PASS** (~50s).

- [x] **T6.2** Verify notebook 04 runs end-to-end
  - Added `test_notebook_04_executes_end_to_end` (gated on `nbconvert`). The notebook
    loads `comparison_summary.json` + `comparison_metrics.csv` (`model` column),
    renders the per-model table + bar charts, and shows `qualitative_grid.png` — all
    against the canonical Phase-4 artifacts. End-to-end execution: **PASS** (~10s).

- [x] **T6.3** Final acceptance criteria audit (SPEC §12)

| # | Item | Status |
|---|------|--------|
| 1 | Data: train+val+test+1k stratified test2017 benchmark (seed=42) | ✅ (T1.1) |
| 2 | Pretrained weights: Zhang16 ECCV downloaded | ✅ (T1.2) |
| 3 | Fine-tuning: Zhang16 fine-tuned on COCO 2017, best ckpt saved | ✅ (T2.1) |
| 4 | Eval Zhang16 Pretrained: PSNR/SSIM/LPIPS on benchmark | ✅ (T3.1; misleading row caveat documented) |
| 5 | Eval Zhang16 Fine-tuned: PSNR/SSIM/LPIPS on benchmark | ✅ (T3.2) |
| 6 | Eval Zhang17: PSNR/SSIM/LPIPS on benchmark | ✅ (T3.3) |
| 7 | Eval DeOldify: PSNR/SSIM/LPIPS on benchmark | ✅ (T3.4) |
| 8 | Comparison: summary table + visual grids for all 4 models | ✅ (T4.1+T4.2) |
| 9 | MLflow: all training experiments logged | ✅ (training runs `1753768c…`, `4bd4839f…`, `c3d3c19c…`) |
| 10 | Notebook 03: runs end-to-end | ✅ (T6.1) |
| 11 | Notebook 04: 4-model comparison with metrics + visualizations | ✅ (T6.2) |
| 12 | Tests: all unit tests pass (`pytest tests/`) | ✅ |
| 13 | Report: LaTeX report compiles to PDF | ✅ `reports/deep_learning/Report.pdf` (committed) |
| 14 | Config: only deep-learning settings | ✅ (T0.2) |
| 15 | Requirements: no FastAPI/uvicorn | ✅ (T0.2) |
| 16 | Clean git: no large data/weights tracked | ✅ `data/raw/`, `models/*.pth`, `mlruns/`, `results/` all in `.gitignore` |

**CHECKPOINT 6** — [x] Branch ready to merge into `main`. Scope is three paradigms (CNN, Interactive CNN, GAN) — the diffusion paradigm was scoped in originally but excluded after the upstream SD 2.1 deprecation; the exclusion is documented in the report and presentation, and no diffusion code or evaluation artifact remains in the repository.

- [x] **T6.4** Deepen Methodology section + author presentation script (post-checkpoint addition)
  - `tools/make_arch_figures.py`: generates three consistent-style schematic
    architecture figures (`zhang16_arch.png`, `zhang17_arch.png`,
    `deoldify_arch.png`) into `reports/deep_learning/figures/`, plus a copy of
    the shared Lab-pipeline overview (`Deep_Learning_method.png`) pulled from
    `references/`.
  - `sections/method.tex` rewritten as a per-paradigm deep dive: for each of
    the three benchmarked paradigms (Zhang16, Zhang17, DeOldify) it inserts the
    paradigm's architecture figure, walks block-by-block through what each part
    does, and explains *why* each design choice matters for the result. The
    Zhang16 subsection still covers the loss, quantization, and annealed-mean
    decoder mathematically; the other two are explanatory-only since we did not
    retrain them.
  - `sections/related_work.tex` trimmed so it no longer duplicates the
    architecture detail now in `method.tex`; it remains the historical taxonomy
    map for the three benchmarked paradigms plus a brief note on the excluded
    diffusion paradigm.
  - `tests/test_report.py`: `REQUIRED_FIGURES` updated to list the new
    architecture figures + pipeline overview.
  - `reports/deep_learning/presentation_script.md`: 20-slide speaker script
    with full talking points and a Q&A prep section, written for a CV professor
    audience; includes a dedicated slide explaining *why* diffusion is excluded.

---

## Quick Reference: What Can Run In Parallel

```
Parallel Group A (Phase 0):  T0.1 | T0.2 | T0.3
Parallel Group B (Phase 1):  T1.1 | T1.2
Parallel Group C (Phase 3):  T3.1 | T3.3 | T3.4  (T3.2 waits for T2.1)
Parallel Group D (Phase 5):  T5.1 | T5.2 | T5.3  (can start during Phase 2-3)
Parallel Group E (Phase 6):  T6.1 | T6.2
```
