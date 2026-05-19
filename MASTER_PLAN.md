# Master Plan: Multi-Methods Colorization (CV 2025.2)

## Week 1 (Days 1-7): Foundation + Implementation Start

| Day | Deep (2 people) | Scribble (1 person) | Example (2 people) | Lead |
|-----|-----------------|---------------------|--------------------|------|
| 1-3 | Download COCO 2017 (~18 GB) + weights | Implement core solver | Implement feature extraction + KD-tree | Scaffold `app/demo.py` |
| 4-7 | Start Zhang16 fine-tuning (long-running) | Add ScribbleColorizer class + tests | Add ExampleColorizer class + tests | Docker polish, start LaTeX structure |

## Week 2 (Days 8-14): Complete + Evaluate

| Day | Deep | Scribble | Example | Lead |
|-----|------|----------|---------|------|
| 8-10 | Evaluate all 5 models on benchmark | Evaluate on benchmark images | Evaluate on benchmark images | Merge feature/deep -> develop |
| 11-12 | Generate comparison figures, write report sections | Write notebook 01 + report section | Write notebook 02 + report section | Merge scribble + example -> develop |
| 13-14 | Review + polish | Review + polish | Review + polish | Wire Gradio demo with all 3 methods |

## Week 3 (Days 15-21): Integration + Report + Polish

| Day | All | Lead |
|-----|-----|------|
| 15-17 | Review cross-method comparison results | Write report: Introduction, Related Work, Experiments, Results |
| 17-19 | Final bug fixes, test all notebooks | Write Discussion + Conclusion, create Slides |
| 19-21 | Final review pass | Merge develop -> main, tag v1.0, deploy Docker |

---

# 5. Technical Specs per Method

## 5.0 Input Preprocessing

### Shared — caller's responsibility

Tất cả ba method đều nhận `gray_image` là **grayscale BGR image**. Caller (demo app, evaluation scripts) có trách nhiệm chuyển ảnh màu sang grayscale trước khi gọi `colorize()`.

Chuẩn chuyển đổi: `cv2.COLOR_BGR2GRAY` (luminance formula: 0.299R + 0.587G + 0.114B). Mỗi `colorize()` phải chấp nhận cả hai dạng `(H, W)` và `(H, W, 3)`.

### Per-method — nằm trong `colorize()`

| Method | Color space dùng nội bộ | Preprocessing bên trong |
|--------|------------------------|------------------------|
| Scribble | YUV | BGR → YUV; dùng Y làm affinity weights; sau khi solve ghép lại Y + U + V → BGR |
| Example | CIE Lab | Target BGR → Lab; Reference BGR → Lab; feature = (L, std của L trong cửa sổ NxN) |
| Deep Learning | CIE Lab | BGR → Lab; L normalize về [-1, 1]; resize 256×256 cho model input; output là kênh ab |

Mỗi method **tự xử lý preprocessing** bên trong.

---

## 5.1 Scribble-based (Levin 2004)

**Paper:** "Colorization using Optimization" - Levin, Lischinski, Weiss (SIGGRAPH 2004)  
**Core idea:** Neighboring pixels with similar luminance should have similar color. User scribbles provide hard constraints. Solve as a sparse linear system.

**Algorithm steps:**
1. Convert input image to YUV (Y = luminance, U/V = chrominance)
2. Detect scribble pixels -> extract their U, V values as constraints
3. Build sparse affinity matrix `W` where `W(r,s) = exp(-(Y(r)-Y(s))^2 / (2*sigma^2))` normalized per row
4. Build linear system `(I - W_modified) * u = b` where scribble pixels are fixed in `b`
5. Solve with `scipy.sparse.linalg.spsolve` for U channel, then V channel
6. Combine Y (original) + solved U + solved V -> convert back to BGR

**File structure required:**
```
src/scribble/
├── __init__.py          # exports ScribbleColorizer
├── colorizer.py         # ScribbleColorizer class (main API)
├── solver.py            # build_affinity_matrix(), solve_colorization()
└── utils.py             # yuv conversions, scribble extraction
```

**Unified API:** `ScribbleColorizer.colorize(gray_image, scribble_overlay)` — `gray_image` is `(H, W, 3)` BGR, `scribble_overlay` is `(H, W, 4)` BGRA with alpha=255 on scribble pixels. Returns `(colorized_bgr, info_dict)`. See §6.

**Key config keys (`scribble:` in config.yaml):** `sigma` (default 0.1 — Gaussian affinity scale), `n_neighbors` (default 8 — 8-connected neighborhood).

**Libraries:** `numpy`, `scipy.sparse`, `scipy.sparse.linalg`, `opencv-python`

---

## 5.2 Example-based (Welsh/Mueller 2002)

**Paper:** "Transferring Color to Greyscale Images" - Welsh, Ashikhmin, Mueller (Stony Brook, SIGGRAPH 2002)  
**Core idea:** Match each pixel in the grayscale target to the most similar pixel in the color reference image using luminance + texture statistics. Transfer chrominance from reference to target.

**Algorithm steps:**
1. Convert both target (grayscale) and reference (color) to CIE Lab
2. For each pixel `p` in target: compute feature vector `f(p) = (L(p), std(L_neighborhood(p)))`
3. For each pixel `q` in reference: compute feature vector `f(q) = (L(q), std(L_neighborhood(q)))`
4. Build KD-tree on reference feature vectors (`sklearn.neighbors.KDTree`)
5. For each target pixel, find k=5 nearest neighbors in reference; pick best by luminance similarity
6. Transfer `a, b` values from best-matched reference pixel to target pixel
7. Combine `L_target + a_transferred + b_transferred` -> convert back to BGR

**File structure required:**
```
src/example_based/
├── __init__.py          # exports ExampleColorizer
├── colorizer.py         # ExampleColorizer class (main API)
├── matching.py          # build_feature_vectors(), kd_tree_match()
└── utils.py             # lab conversions, neighborhood statistics
```

**Two modes (both supported in the demo):**
- **Mode 1 — Automatic (global matching):** KD-tree searches entire reference image. Current implementation. *(Priority: Phase 1)*
- **Mode 2 — User-guided (swatch matching):** User marks corresponding regions on target and reference; pixels inside each target region only match against their paired reference region. *(Priority: Phase 2 — demo polish)*

**Unified API:** `ExampleColorizer.colorize(gray_image, reference_image, swatches=None)` — both image inputs are `(H, W, 3)` BGR. `swatches` is an optional list of `(target_mask, ref_mask)` pairs (each `(H, W)` bool arrays); if `None`, falls back to Mode 1. Returns `(colorized_bgr, info_dict)`. See §6.

**Key config keys (`example_based:` in config.yaml):** `neighborhood_size` (default 5 — NxN window for std), `k_neighbors` (default 5 — KD-tree candidates), `downsample` (default 0.5 — resize reference for speed).

**Libraries:** `numpy`, `opencv-python`, `scikit-learn` (KDTree), `scikit-image`

---

## 5.3 Deep Learning (Zhang 2016)

**Paper:** "Colorful Image Colorization" — Zhang, Isola, Efros (ECCV 2016)  
**Core idea:** Frame colorization as classification over 313 quantized ab bins. Class-rebalanced cross-entropy upweights rare/saturated colors to produce vivid results.

### Model Variants

| # | Category | Model | Source | Mode |
|---|----------|-------|--------|------|
| 1 | CNN | Zhang16 Pretrained | Our reimplementation + official ECCV weights | Inference only |
| 2 | CNN | Zhang16 Fine-tuned | Our reimplementation + fine-tuned on COCO 2017 | Fine-tune |
| 3 | Interactive CNN | Zhang17 auto | Official `colorizers` package, zero hints | Inference only |
| 4 | GAN | DeOldify | Official pretrained (Self-Attention GAN) | Inference only |
| 5 | Diffusion | ControlNet + SD 2.1 | HuggingFace `neurallove/controlnet-sd21-colorization-diffusers` | Inference only |

Only Zhang16 (#1, #2) is our code. Models #3–#5 wrap third-party pretrained weights with the unified `colorize()` API.

### Zhang16Net Architecture

- **Input:** L channel `(B, 1, H, W)`, normalized to `[-1, 1]`
- **Encoder:** 8 conv blocks — blocks 1–3 stride-2 (H→H/8), blocks 4–7 dilated at H/8
- **Decoder:** block 8 upsamples H/8→H/4, then 1×1 conv → `(B, 313, H/4, W/4)` logits
- **Inference:** softmax → upsample → annealed-mean decoding (T=0.38) → ab values
- **Quantization:** 313 in-gamut ab bins (`src/deep_learning/pts_in_hull.npy`)
- **Loss:** class-rebalanced cross-entropy (Zhang 2016 Eq. 2–4)

### Training Hyperparameters (Zhang16 Fine-tuning)

| Parameter | Value |
|-----------|-------|
| Base weights | Zhang16 ECCV pretrained |
| Dataset | COCO 2017 train (118K images) |
| Epochs | 50 |
| Batch size | 8 (AMP enabled for 6 GB VRAM) |
| Learning rate | 0.0002 (Adam) |
| Scheduler | StepLR (step=20, gamma=0.5) |
| Input size | 256×256 |

### Execution Order

Current state: **code-complete.** Pre-training fixes already applied (`requirements.txt`, `config.yaml`, `num_classes=313`).

1. `pytest tests/ -v` — all tests green (T0.1)
2. `tools/download_coco.py --split both` — COCO 2017 (~18 GB) (T1.1)
3. `tools/download_pretrained.py --model zhang16` — official ECCV weights (T1.2)
4. `tools/train_deep.py --config configs/config.yaml` — fine-tune (~4–12h GPU) (T2.1)
5. `tools/evaluate_deep.py --tag pretrained` (T3.1)
6. `tools/evaluate_deep.py --tag finetuned` (T3.2)
7. `tools/compare_methods.py --max-images 50` (T4.1)

### Risks

| Risk | Mitigation |
|------|------------|
| VRAM overflow (6 GB GTX 1660) | AMP enabled; reduce batch size to 4 if OOM |
| ControlNet needs >6 GB | Run on RTX 2080 Ti, or use CPU offloading |
| DeOldify/ControlNet dependency conflicts | Install in separate conda env; document in workflow |
| Fine-tuning diverges | Start from pretrained, conservative LR, monitor val loss |

---

# 6. Unified API Contract

All three methods expose the same base interface so the demo app and cross-method comparison work uniformly.

`colorizer.colorize(image, **kwargs)` returns `(result_bgr, info)` where `result_bgr` is `(H, W, 3)` BGR uint8 and `info` is a dict containing at least `method` (str), `time_seconds` (float), and `image_size` (H, W).

Cross-method comparison on `main` will call this interface — do NOT break it.

---

# 7. Docker - Dev Environment

**Goal:** Everyone runs the same Python environment regardless of OS (Windows/Mac).  
**Scope:** CPU-only — development, unit tests, notebooks, Gradio demo.  
**GPU training:** Done natively (not in Docker) — CUDA in Docker on Windows/Mac is too complex.

## Files (already created)

`docker/Dockerfile`, `docker/docker-compose.yml`, `.dockerignore`

**Dockerfile:** Python 3.10-slim, installs OpenCV system libs, CPU-only PyTorch, project requirements, and Gradio ≥4.0.

**docker-compose.yml services:**
- `dev`: mounts project as volume, exposes port 8888 (Jupyter)
- `demo`: runs `app/demo.py`, exposes port 7860 (Gradio)

**Usage:** `docker compose run --rm dev` for interactive shell · `docker compose up demo` to start the Gradio app · `docker compose run --rm dev pytest tests/ -v` to run tests.

---

# 8. Gradio Demo App

**Location:** `app/demo.py`  
**Framework:** Gradio Blocks  
**Launch:** `python app/demo.py` -> `http://localhost:7860`

## Structure

```
app/
├── demo.py              # Main Gradio app
├── __init__.py
└── assets/
    └── samples/         # 5-10 sample grayscale images for quick demo
```

## Tab layout

**Tab 1 - Deep Learning:**
- Input: image upload (grayscale)
- Control: dropdown (`Zhang16 Pretrained`, `Zhang16 Fine-tuned`, `Zhang17 Auto`, `DeOldify`)
- Output: colorized image + inference time

**Tab 2 - Scribble-based:**
- Input: `gr.ImageEditor` - user draws colored strokes directly on grayscale image
- Output: colorized image
- Instruction: "Use the brush to paint colors on areas you want colored"

**Tab 3 - Example-based:**
- Input 1: grayscale target image
- Input 2: color reference image
- Output: colorized image
- **Mode 1 (Phase 1):** Auto global matching — single "Colorize" button, no extra input
- **Mode 2 (Phase 2):** User-guided swatch matching — user draws region masks on both images via `gr.ImageEditor`, system restricts KD-tree search per region pair

---

# 9. LaTeX Report

**Location:** `reports/` (created on `develop` after all merges)  
**Format:** IEEE conference (`IEEEtran` class), English  
**Compile:** `pdflatex main && bibtex main && pdflatex main && pdflatex main`

## Structure

```
reports/
├── main.tex
├── references.bib
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_scribble.tex         <- scribble team writes this
│   ├── 04_example.tex          <- example team writes this
│   ├── 05_deep_learning.tex    <- DL team writes this
│   ├── 06_experiments.tex
│   ├── 07_results.tex
│   └── 08_conclusion.tex
└── figures/
```

## Section ownership

| Section | Author | Due |
|---------|--------|-----|
| Introduction | Lead | Week 3 |
| Related Work | Lead | Week 3 |
| Scribble Method | Scribble engineer | End of Week 2 |
| Example Method | Example engineer B | End of Week 2 |
| Deep Learning Method | DL Engineer B | End of Week 2 |
| Experiments | Lead | Week 3 |
| Results | Lead (from actual metrics) | Week 3 |
| Conclusion | Lead | Week 3 |

---

# 10. Slides

**Location:** `slides/` (PowerPoint or PDF), ~15 min presentation

## Outline (15-20 slides)
1. Problem: Why colorization? (1 slide)
2. Three methods overview — diagram (1 slide)
3. Scribble-based: algorithm + result (2 slides)
4. Example-based: algorithm + result (2 slides)
5. Deep Learning: architecture + 4 DL categories (3 slides)
6. Experiments: dataset, benchmark, metrics (2 slides)
7. Results: quantitative table + qualitative comparison grids (3 slides)
8. Demo: live or GIF recording (1 slide)
9. Conclusion + future work (1 slide)

---

# 11. Shared Benchmark & Unified Evaluation Dataflow

All three methods are evaluated on the **same 500 images** (`data/raw/coco2017/benchmark/`, seed=42).

## 11.1 Input Preparation per Method

Benchmark images are grayscale. Each method needs different auxiliary inputs for automated evaluation:

| Method | Auxiliary Input | Generation Strategy |
|--------|----------------|---------------------|
| Deep Learning | None | Feed grayscale image directly |
| Example-based | Color reference image | Fixed pool of 100 diverse COCO val images (excl. benchmark); for each test image pick the reference with smallest L1 distance on 64-bin luminance histogram |
| Scribble-based | BGRA scribble overlay | Oracle scribbles: randomly sample 1% of pixels (seed=42) from ground truth color image, mark positions with true BGR values and alpha=255 |

Run `tools/prepare_benchmark.py` once before evaluation to build the reference pool and oracle scribbles.

## 11.2 Color Space & I/O Standard

| Stage | Space | dtype |
|-------|-------|-------|
| Input to `colorize()` | BGR (OpenCV) | uint8 |
| Output from `colorize()` | BGR | uint8 |
| Metric computation (PSNR / SSIM / LPIPS) | RGB | uint8 |

All evaluation scripts convert BGR→RGB before computing metrics.

## 11.3 Evaluation Commands

1. `tools/evaluate_scribble.py --tag oracle` → `results/scribble/metrics/`
2. `tools/evaluate_example.py --tag pool-match` → `results/example_based/metrics/`
3. `tools/evaluate_deep.py --tag pretrained` and `--tag finetuned` → `results/deep_learning/metrics/`
4. `tools/compare_methods.py --max-images 50` → `results/comparison/` (after all methods evaluated)

## 11.4 Results Directory Structure

```
results/
├── scribble/
│   ├── metrics/
│   │   ├── aggregate_metrics.json
│   │   └── per_image_metrics.csv
│   └── figures/
├── example_based/
│   ├── metrics/
│   │   ├── aggregate_metrics.json
│   │   └── per_image_metrics.csv
│   └── figures/
├── deep_learning/
│   ├── metrics/              # per-model JSON + aggregate CSV
│   ├── comparison/           # 5-model DL comparison grids
│   └── figures/
└── comparison/
    ├── cross_method_summary.json
    ├── per_image_metrics.csv
    └── grids/
```

## 11.5 Metrics & Cross-Method Comparison

| Metric | Better | Range |
|--------|--------|-------|
| PSNR | Higher | [0, ~50] dB |
| SSIM | Higher | [0, 1] |
| LPIPS | Lower | [0, 1] |
| Time/img | Lower | seconds |

**Fill after evaluation:**

| Method | PSNR | SSIM | LPIPS | Time/img |
|--------|------|------|-------|----------|
| Scribble (Levin 2004) | - | - | - | - |
| Example (Welsh 2002) | - | - | - | - |
| DL: Zhang16 Pretrained | - | - | - | - |
| DL: Zhang16 Fine-tuned | - | - | - | - |
| DL: Zhang17 Auto | - | - | - | - |
| DL: DeOldify | - | - | - | - |
| DL: ControlNet | - | - | - | - |

---

# 12. Immediate Actions (Do This First — Day 1)

1. **Fix `requirements.txt`** — encoding bug is already fixed in this branch; push to develop
2. **Fix `configs/config.yaml`** — remove scribble/example sections, fix `num_classes: 233` -> `313`
3. **Run `pytest tests/ -v`** — all tests must be green before execution phase
4. **Create `docker/Dockerfile`** — Lead sets up CPU dev environment (§7)
5. **Scaffold `feature/scribble`** — create empty class stubs (§5.1 structure)
6. **Scaffold `feature/example`** — create empty class stubs (§5.2 structure)

---

# 13. Acceptance Criteria (Definition of Done)

## Code
- [ ] `pytest tests/ -v` passes for all 3 methods
- [ ] `configs/config.yaml` has clean section per method
- [ ] Notebooks 01-04 run end-to-end without errors
- [ ] `app/demo.py` starts and all 3 tabs produce output

## Results
- [ ] Scribble + Example evaluated on COCO benchmark (PSNR/SSIM/LPIPS)
- [ ] All 5 DL model variants evaluated on benchmark
- [ ] Cross-method comparison table complete (`results/comparison/`)
- [ ] Qualitative grids generated for report

## Report
- [ ] `reports/main.tex` compiles to PDF without errors
- [ ] All 8 sections present
- [ ] All citations resolve
- [ ] Figures and tables render correctly

## Demo + Docker
- [ ] `docker compose up demo` starts Gradio without errors
- [ ] All 3 tabs produce colorized output
- [ ] Sample images included for quick demo

## Delivery
- [ ] All feature branches merged to `develop`
- [ ] `develop` merged to `main` and tagged `v1.0`
- [ ] README on `main` explains setup and usage
- [ ] Slides complete (15-20 slides)
