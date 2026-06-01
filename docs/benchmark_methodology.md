# Benchmark Methodology

> **Branch:** `feature/deep`
> **Status:** Authoritative. Anything that conflicts with this document is wrong.
> **Last updated:** 2026-05-19

This document is the **single source of truth** for how the 5-model evaluation benchmark is
constructed and how metrics are reported. Read it before touching anything in the evaluation
pipeline. If you skip this and "just look at the code," you will mis-explain the numbers.

---

## TL;DR

We evaluate all 5 model variants on **the same 1,000-image set built from COCO 2017 `test2017`**
(40,670 images total). The sample is:

- **Held out** from training and from best-checkpoint selection (so no model has "seen" it)
- **Stratified** over a 4 × 4 grid of brightness × saturation, proportionally allocated
  (so the sample's color-difficulty distribution matches the full `test2017`)
- **Deterministic** (seed=42; rerunning produces the same 1,000 images)

Metrics are reported as **mean + 95% percentile bootstrap CI** (10,000 resamples, seed=42).

| Reproduce in one command | |
|--------------------------|---|
| `python tools/download_coco.py --source local-zip --split all --benchmark-size 1000 --benchmark-strategy stratified` | rebuilds `data/raw/coco2017/benchmark/` |
| `python tools/_verify_dataset.py` | confirms 1000 imgs, all subset of test2017, zero overlap with val2017, KS-style diversity check |

---

## 1. The Splits

| Directory | Count | Role | Touched during fine-tuning? |
|-----------|------:|------|:--:|
| `data/raw/coco2017/train2017/`    | 118,287 | Fine-tune Zhang16 (gradient updates)                                | **yes** |
| `data/raw/coco2017/val2017/`      |   5,000 | In-training validation — `Trainer.fit` picks `best_model.pth` here  | **yes** (no gradients, but used for model selection — this is the leak source) |
| `data/raw/coco2017/test2017/`     |  40,670 | Source pool for the held-out evaluation benchmark                   | **no**  |
| `data/raw/coco2017/benchmark/`    |   1,000 | The actual evaluation set (subset of `test2017`)                    | **no**  |
| `data/raw/coco2017/annotations/`  |       6 | COCO JSON metadata (bundled with the Kaggle archive). Not used by colorization. | — |

The benchmark/ directory is what `tools/evaluate_deep.py` and `tools/compare_methods.py` read by
default. Re-pointing them elsewhere is a deliberate change.

---

## 2. Why `test2017`, not `val2017` — the leakage explanation

In `src/deep_learning/train.py`, `Trainer.fit` saves the best checkpoint based on validation
loss over `val2017`:

```python
val_loss, val_metrics = self.validate()              # runs on val_loader = val2017
...
if val_loss < self.best_val_loss:
    self.best_val_loss = val_loss
    self.save_checkpoint(os.path.join(self.save_dir, "best_model.pth"), epoch)
```

If the 1,000-image evaluation benchmark were sampled from `val2017`, those 1,000 images would
have contributed (along with the other 4,000) to deciding **which checkpoint became
`best_model.pth`**. Their evaluation metrics would then be implicitly optimistic — classic
**val/test contamination**.

`test2017` is COCO's official held-out split. The Zhang16 fine-tuner never sees it during
either gradient updates or checkpoint selection, so it gives a clean estimate of generalization.

**Important nuance:** the leakage applies *specifically to our Zhang16 Fine-tuned model*. The
other 3 models (Zhang16 Pretrained, Zhang17, DeOldify) were trained on their own
datasets (ImageNet, web crawl, etc.), and they have whatever contamination they have regardless
of whether we benchmark on val2017 or test2017. Switching the benchmark to test2017 does **not**
change the picture for those models — it just gives Zhang16 Fine-tuned a fair shake without
hurting anyone else.

**Why this matters for the report:** when you describe "PSNR = X dB on benchmark," you can
write the methodology section truthfully: "1,000-image held-out subset of COCO 2017 test2017,
never seen during fine-tuning or checkpoint selection." That sentence wouldn't be true if the
benchmark came from val2017.

---

## 3. Why 1,000 images

Compute budget at the speed of the slowest benchmarked model (DeOldify, ~0.5 s/image):

| n  | All-4-models eval wall-clock | Standard error of PSNR mean (σ≈5 dB) |
|---:|---|---|
|   500 | ~10 min | ±0.224 dB |
| **1,000** | **~20 min** | **±0.158 dB** |
| 2,000 | ~40 min | ±0.112 dB |
| 40,670 (full test2017) | ~14 h | ±0.025 dB |

At 1,000 images, the standard error of the PSNR mean is already 15× narrower than typical
between-model gaps (0.5–3 dB). Going larger gives diminishing returns on statistical power and
makes the comparison infeasible on a single RTX 2080 Ti. Going smaller stays statistically
sound but reduces our headroom against unlucky draws.

---

## 4. Sampling strategy: stratified by brightness × saturation

### 4.1 Why stratify at all

Pure random sampling at n=1,000 is statistically unbiased (E[sample mean] = population mean),
but a single random draw can still under-represent specific content modes — e.g., all bright
outdoor shots, or all dim indoor shots. That doesn't bias the mean, but it makes the sample
less *defensible* as "representative." Stratified proportional sampling guarantees coverage of
the stratification axes and **reduces variance** by removing between-stratum variance, which is
strictly better than random when the strata correlate with the metric.

For **colorization specifically**, per-image PSNR/SSIM/LPIPS varies strongly with:
- **scene brightness** — affects how easy the L channel structure is to decode and how grey
  the chrominance prediction is
- **scene color saturation** — determines whether the target ab distribution is concentrated
  near zero (easy) or wide-spread (hard)

Image aspect ratio and scene category are weaker confounders. We stratify on brightness +
saturation only.

### 4.2 Features (cheap, no model inference required)

For every image in `test2017`, `_compute_brightness_saturation_features()` in
`tools/download_coco.py` does:

```python
with Image.open(path) as img:
    thumb = img.convert("RGB").resize((32, 32), Image.BILINEAR)
arr = np.asarray(thumb, dtype=np.float32)               # (32, 32, 3) in [0, 255]
brightness  = float(arr.mean())                          # 0..255  — overall lightness
saturation  = float((arr.max(axis=2) - arr.min(axis=2)).mean())  # 0..255 — color span per pixel
```

Two scalars per image. The pass over 40,670 images takes ~1–3 min and is cached to
`data/raw/coco2017/test2017_features.npz` so subsequent runs are instant.

### 4.3 Binning + proportional allocation

```
N_BINS = 4 per feature  →  4 × 4 = 16 cells
bright_edges = np.quantile(features[:, 0], [0, 0.25, 0.5, 0.75, 1.0])   # equal-population edges
sat_edges    = np.quantile(features[:, 1], [0, 0.25, 0.5, 0.75, 1.0])
b_idx = np.digitize(features[:, 0], bright_edges[1:-1])  # 0..3
s_idx = np.digitize(features[:, 1], sat_edges[1:-1])     # 0..3
cell_idx = b_idx * 4 + s_idx                              # 0..15
```

Each cell has ~2,500 images by construction (40,670 / 16 ≈ 2,542). Proportional allocation:

```
for c in range(16):
    cell_files = files where cell_idx == c
    n_take    = round((|cell_files| / 40,670) * 1,000)   ≈ 62
    selected  ← rng.choice(cell_files, size=n_take, replace=False, seed=42)
```

Truncate or top-up to exactly 1,000.

### 4.4 Observed result (verified by `tools/_verify_dataset.py`)

|                     |   mean   |  std  | p10   | p90    |
|---------------------|---------:|------:|------:|-------:|
| **Full test2017** brightness | 112.6 | 30.5 | 74.8 | 150.2 |
| **Benchmark** brightness     | 112.4 | 30.5 | 73.8 | 150.5 |
| **Full test2017** saturation |  35.2 | 23.4 | 11.4 |  65.3 |
| **Benchmark** saturation     |  35.0 | 22.8 | 11.4 |  66.4 |

Max quantile gap (population vs benchmark): **1.25** for brightness, **2.00** for saturation
on a 0–255 scale. The 1,000-image sample's color-difficulty distribution is statistically
indistinguishable from the full 40,670-image test2017.

All 16 cells are populated, per-cell counts 52–79, summing to exactly 1,000.

### 4.5 Falling back to random

For ablation purposes: `--benchmark-strategy random` does
`random.sample(files, 1000)` with seed=42. Use this only to verify that the stratified
benchmark gives stable metrics — never as the production sample.

---

## 5. Bootstrap 95% confidence intervals

### 5.1 Why

A single number like "PSNR = 25.3 dB" is a point estimate. Reviewers reasonably ask: "how
sure are you?" The percentile bootstrap converts the point estimate into an interval estimate
without assuming any parametric distribution.

### 5.2 The math

Given per-image metric values `x = (x_1, …, x_n)` with `n = 1,000`:

```
for b in 1..10,000:
    idx_b   = rng.integers(0, n, size=n)            # sample with replacement
    means_b = x[idx_b].mean()
ci_lo, ci_hi = quantile(means_b, [0.025, 0.975])     # 95% percentile CI
```

Implementation in `src/deep_learning/stats.py` is fully vectorised:

```python
idx = rng.integers(0, n, size=(n_boot, n))           # (10,000, 1,000) — ~80 MB
boot_means = values[idx].mean(axis=1)                # (10,000,)
```

Returns `{mean, std, ci_lo, ci_hi, n, n_boot}`. Smoke-tested to match CLT predictions to
within ~5% (see commit history).

### 5.3 Seed and reproducibility

Both bootstrap calls (in `evaluate_deep.py` and `compare_methods.py`) use:

```
ci=0.95, n_boot=10_000, seed=42
```

Re-running on the same per-image metric file produces byte-identical CIs.

### 5.4 What you'll see in output

`results/deep_learning/metrics/aggregate_metrics.json` after a single-model eval:

```json
{
  "num_images": 1000,
  "psnr_mean":   25.32,
  "psnr_std":    4.78,
  "psnr_ci_lo":  25.03,
  "psnr_ci_hi":  25.61,
  "ssim_mean":   0.872,
  "ssim_std":    0.063,
  "ssim_ci_lo":  0.868,
  "ssim_ci_hi":  0.876,
  "lpips_mean":  0.142,
  "lpips_std":   0.052,
  "lpips_ci_lo": 0.139,
  "lpips_ci_hi": 0.145,
  "ci_level":    0.95,
  "n_bootstrap": 10000,
  "avg_time_sec": 0.087
}
```

Console output prints `PSNR 25.32 [95% CI 25.03, 25.61]` lines.

---

## 6. Reproduction commands

```bash
# Conda env (required for all of the below)
conda activate AI

# (one-time) Download / extract everything, build the stratified benchmark
python tools/download_coco.py --source local-zip --split all \
       --benchmark-size 1000 --benchmark-strategy stratified

# Verify dataset integrity + leakage check + diversity check
python tools/_verify_dataset.py

# Evaluate one model
python tools/evaluate_deep.py --model-path models/pretrained/zhang16_eccv.pth \
       --tag pretrained

# Compare all 5 models
python tools/compare_methods.py --max-images 1000
```

`_verify_dataset.py` is intended to be re-runnable on demand. It prints exactly the invariants
this document promises.

---

## 7. Invariants (what must always be true)

| Invariant | Where it's checked |
|---|---|
| `benchmark/` contains exactly 1,000 images | `tools/_verify_dataset.py` |
| Every image in `benchmark/` is a member of `test2017/` | `tools/_verify_dataset.py` (set intersection) |
| Zero overlap between `benchmark/` and `val2017/` | `tools/_verify_dataset.py` (set intersection) |
| All 16 stratification cells are non-empty | `download_coco.py` cell-coverage report |
| Per-cell allocation is proportional (52–79 imgs each) | `download_coco.py` cell-coverage report |
| Bootstrap CIs use n_boot=10,000, seed=42, ci=0.95 | hard-coded in `evaluate_deep.py` and `compare_methods.py` |

If any of these fails, the benchmark numbers are not the ones this document describes.
Don't quote them as such.

---

## 8. Trouble shooting

**Q: I re-ran `download_coco.py` and got a different benchmark.**
A: You changed something — likely `--benchmark-strategy random`, a different `--benchmark-size`,
or a different seed. The default invocation in §6 is deterministic.

**Q: The cached `test2017_features.npz` exists but my benchmark is empty.**
A: Cell coverage requires `test2017/` to be populated. Run `--split all` first.

**Q: Should I delete `archive.zip` after extraction?**
A: Yes — it's ~25 GB and you only need it once. `tools/download_coco.py` extracts it on first
run and is idempotent afterwards.

**Q: My new model's PSNR is 24.5 [95% CI 24.0, 25.0] and the baseline is 25.0 [24.6, 25.4].
Did the new model improve?**
A: The CIs overlap heavily — the difference is not statistically significant at α=0.05.
Treat them as a tie. (Formal test: paired bootstrap on the per-image differences.)

**Q: Can I add a new comparison model?**
A: Yes, but document where it sits in the taxonomy (CNN / Interactive CNN / GAN are
benchmarked; Diffusion is the explicitly-excluded fourth category — see SPEC §2). Run it
on the same `benchmark/` with the same bootstrap settings. Update SPEC.md.

**Q: Can I change `n=1,000` or the strata?**
A: That invalidates all prior numbers. If you do, bump SPEC §6.3, this document, and add a
row to the decision log below. Then re-run every model.

---

## 9. Decision log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-13 | Initial SPEC: 500 random imgs from val2017 as benchmark | First draft, didn't notice val/test contamination |
| 2026-05-19 | Switch benchmark from val2017 → 1,000 random imgs from test2017 | Found that `Trainer.fit` selects checkpoints by val_loss, leaking 500-img val subset evaluation for Zhang16 Fine-tuned |
| 2026-05-19 | Random → stratified by brightness × saturation (4×4 cells, proportional) | Random draw doesn't *guarantee* color-difficulty coverage; stratification removes between-stratum variance |
| 2026-05-19 | Add 95% bootstrap CIs (n_boot=10,000, seed=42) to all metric reporting | Point estimates aren't defensible without uncertainty; bootstrap is cheap and non-parametric |

---

## 10. Files involved

| File | Role |
|---|---|
| `tools/download_coco.py` | Extracts COCO 2017, computes thumbnail features, builds stratified `benchmark/` |
| `src/deep_learning/stats.py` | `bootstrap_ci()` — numpy-only, no torch/skimage import |
| `tools/evaluate_deep.py` | Single-model eval, reports `mean ± 95% CI` |
| `tools/compare_methods.py` | 5-model comparison, reports `mean ± 95% CI` per model |
| `tools/_verify_dataset.py` | Asserts every invariant in §7 |
| `data/raw/coco2017/test2017_features.npz` | Cached `(brightness, saturation)` features for all 40,670 test imgs |
| `data/raw/coco2017/benchmark/` | The 1,000 images themselves (full JPEGs, ~160 MB) |

`SPEC.md` §6.3 contains the short-form version of everything in this document. If they ever
disagree, this document wins and SPEC.md needs to be updated.
