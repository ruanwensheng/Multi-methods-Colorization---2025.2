# Master Plan: Multi-Methods Colorization (CV 2025.2)

> **Status:** Active  
> **Owner:** Project Lead  
> **Last updated:** 2026-05-15  
> **Deadline:** ~2026-06-05 (3 weeks)

---

## 1. Project Overview

Build and compare three image colorization methods on a shared benchmark:

| Method | Branch | Algorithm | Team |
|--------|--------|-----------|------|
| Deep Learning | `feature/deep` | Zhang 2016 CNN + 4 DL categories | 2 people |
| Scribble-based | `feature/scribble` | Levin 2004 "Colorization using Optimization" | 1 person |
| Example-based | `feature/example` | Welsh/Mueller 2002 luminance transfer | 2 people |

**Final deliverables:** Source code (GitHub) · LaTeX report (IEEE) · Gradio demo · Slides

---

## 2. Team Responsibilities

| Role | Branch | Core tasks |
|------|--------|------------|
| DL Engineer A | `feature/deep` | Execute pipeline: download COCO, train Zhang16, eval 5 models |
| DL Engineer B | `feature/deep` | Fix Phase 0 issues, run tests, write report sections 3–5 |
| Scribble Engineer | `feature/scribble` | Implement Levin 2004, tests, notebook, report section |
| Example Engineer A | `feature/example` | Implement Welsh 2002 core algorithm + KD-tree matching |
| Example Engineer B | `feature/example` | Tests, evaluation, notebook, report section |
| **Project Lead** | `develop` / `main` | Docker, Gradio demo, report integration, slides, merge coordination |

---

## 3. Branch Strategy

```
main             <- final release (tag v1.0 when done)
develop          <- integration hub (all merges land here first)
feature/deep     <- Deep Learning (code-complete, execution phase)
feature/scribble <- Scribble-based (needs implementation)
feature/example  <- Example-based (needs implementation)
```

### Git Workflow per person

```bash
# Each morning: sync from develop
git fetch origin
git merge origin/develop

# Work on your feature
git add src/<your_method>/
git commit -m "feat: ..."
git push origin feature/<your-branch>

# When ready to merge: open PR -> feature/* -> develop
# Lead reviews and merges
```

### Merge order
1. `feature/deep` -> `develop` (first, since code is done)
2. `feature/scribble` -> `develop`
3. `feature/example` -> `develop`
4. `develop` -> `main` (final release)

---

## 4. Master Timeline

### Week 1 (Days 1-7): Foundation + Implementation Start

| Day | Deep (2 people) | Scribble (1 person) | Example (2 people) | Lead |
|-----|-----------------|---------------------|--------------------|------|
| 1-2 | Fix Phase 0: tests, config, notebook 04 | Set up branch, read Levin 2004 | Set up branch, read Welsh paper | Fix requirements.txt, create Docker |
| 3-4 | Download COCO 2017 (~18 GB) + weights | Implement core solver | Implement feature extraction + KD-tree | Scaffold `app/demo.py` |
| 5-7 | Start Zhang16 fine-tuning (long-running) | Add ScribbleColorizer class + tests | Add ExampleColorizer class + tests | Docker polish, start LaTeX structure |

### Week 2 (Days 8-14): Complete + Evaluate

| Day | Deep | Scribble | Example | Lead |
|-----|------|----------|---------|------|
| 8-10 | Evaluate all 5 models on benchmark | Evaluate on benchmark images | Evaluate on benchmark images | Merge feature/deep -> develop |
| 11-12 | Generate comparison figures, write report sections | Write notebook 01 + report section | Write notebook 02 + report section | Merge scribble + example -> develop |
| 13-14 | Review + polish | Review + polish | Review + polish | Wire Gradio demo with all 3 methods |

### Week 3 (Days 15-21): Integration + Report + Polish

| Day | All | Lead |
|-----|-----|------|
| 15-17 | Review cross-method comparison results | Write report: Introduction, Related Work, Experiments, Results |
| 17-19 | Final bug fixes, test all notebooks | Write Discussion + Conclusion, create Slides |
| 19-21 | Final review pass | Merge develop -> main, tag v1.0, deploy Docker |

---

## 5. Technical Specs per Method

### 5.1 Scribble-based (Levin 2004)

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

**Unified API:**
```python
class ScribbleColorizer:
    def colorize(self, gray_image, scribble_overlay) -> tuple[np.ndarray, dict]:
        """
        gray_image:       (H, W) or (H, W, 3) BGR grayscale image
        scribble_overlay: (H, W, 4) BGRA - alpha=0 for unmarked pixels,
                          alpha=255 for scribble pixels with their BGR color
        Returns: (colorized_bgr: np.ndarray HxWx3, info: dict)
        """
```

**Key parameters (add to config.yaml under `scribble:`):**
```yaml
scribble:
  sigma: 0.1          # affinity Gaussian sigma (intensity difference scale)
  n_neighbors: 8      # pixel neighborhood size (4 or 8-connected)
```

**Libraries:** `numpy`, `scipy.sparse`, `scipy.sparse.linalg`, `opencv-python`

---

### 5.2 Example-based (Welsh/Mueller 2002)

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

**Unified API:**
```python
class ExampleColorizer:
    def colorize(self, gray_image, reference_image) -> tuple[np.ndarray, dict]:
        """
        gray_image:       (H, W) or (H, W, 3) BGR grayscale image
        reference_image:  (H, W, 3) BGR color reference image
        Returns: (colorized_bgr: np.ndarray HxWx3, info: dict)
        """
```

**Key parameters (update `example_based:` in config.yaml):**
```yaml
example_based:
  neighborhood_size: 5    # NxN window for local std computation
  k_neighbors: 5          # KD-tree nearest neighbors to sample from
  downsample: 0.5         # resize reference for speed (1.0 = no downscale)
```

**Libraries:** `numpy`, `opencv-python`, `scikit-learn` (KDTree), `scikit-image`

---

### 5.3 Deep Learning (Zhang 2016) - Execution Phase

Current state: **code-complete, not yet executed.** Known issues to fix before running:

| Issue | File | Fix |
|-------|------|-----|
| `scribble:` / `example_based:` sections | `configs/config.yaml` | Remove those sections |
| `num_classes: 233` | `configs/config.yaml` | Change to `313` |
| `fastapi`, `uvicorn` in requirements | `requirements.txt` | Already removed (this commit) |
| Encoding bug (space between every char) | `requirements.txt` | Already fixed (this commit) |
| Notebook 04 is cross-method | `notebooks/04_method_comparison.ipynb` | Rewrite as 5-model DL comparison |

Execution order after fixes:
```bash
pytest tests/ -v                                          # T0.1 - must be all green
python tools/download_coco.py --split both               # T1.1 (~18 GB)
python tools/download_pretrained.py --model zhang16      # T1.2
python tools/train_deep.py --config configs/config.yaml  # T2.1 (~4-12h GPU)
python tools/evaluate_deep.py --tag pretrained           # T3.1
python tools/evaluate_deep.py --tag finetuned            # T3.2
python tools/compare_methods.py --max-images 50          # T4.1
```

---

## 6. Unified API Contract

All three methods expose the same base interface so the demo app and cross-method comparison work uniformly:

```python
result_bgr, info = colorizer.colorize(image, **kwargs)

# result_bgr: np.ndarray, shape (H, W, 3), dtype uint8, BGR
# info: dict with at least:
#   {"method": str, "time_seconds": float, "image_size": (H, W)}
```

Cross-method comparison on `main` will call this interface — do NOT break it.

---

## 7. Docker - Dev Environment

**Goal:** Everyone runs the same Python environment regardless of OS (Windows/Mac).  
**Scope:** CPU-only — development, unit tests, notebooks, Gradio demo.  
**GPU training:** Done natively (not in Docker) — CUDA in Docker on Windows/Mac is too complex.

### Files to create (Lead, Week 1 Day 1-2)

```
docker/
├── Dockerfile
└── docker-compose.yml
.dockerignore
```

**Dockerfile spec (CPU-only, Python 3.10):**
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gradio>=4.0

CMD ["bash"]
```

**docker-compose.yml services:**
- `dev`: mounts project as volume, exposes port 8888 (Jupyter)
- `demo`: runs `python app/demo.py`, exposes port 7860 (Gradio)

**Usage:**
```bash
docker compose run --rm dev              # interactive dev shell
docker compose up demo                   # start Gradio demo
docker compose run --rm dev pytest tests/ -v    # run tests
docker compose run --rm dev jupyter lab --ip 0.0.0.0 --no-browser  # notebooks
```

---

## 8. Gradio Demo App

**Location:** `app/demo.py`  
**Framework:** Gradio Blocks  
**Launch:** `python app/demo.py` -> `http://localhost:7860`

### Structure

```
app/
├── demo.py              # Main Gradio app
├── __init__.py
└── assets/
    └── samples/         # 5-10 sample grayscale images for quick demo
```

### Tab layout

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

---

## 9. LaTeX Report

**Location:** `reports/` (created on `develop` after all merges)  
**Format:** IEEE conference (`IEEEtran` class), English  
**Compile:** `pdflatex main && bibtex main && pdflatex main && pdflatex main`

### Structure

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

### Section ownership

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

## 10. Slides

**Location:** `slides/` (PowerPoint or PDF), ~15 min presentation

### Outline (15-20 slides)
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

## 11. Shared Benchmark

All three methods are evaluated on the **same 500 images** (`data/raw/coco2017/benchmark/`, seed=42).

**Metrics for every method:**
- PSNR (higher is better)
- SSIM (higher is better)
- LPIPS (lower is better)
- Inference time per image (seconds)

**Cross-method comparison table:**

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

## 12. Immediate Actions (Do This First — Day 1)

1. **Fix `requirements.txt`** — encoding bug is already fixed in this branch; push to develop
2. **Fix `configs/config.yaml`** — remove scribble/example sections, fix `num_classes: 233` -> `313`
3. **Run `pytest tests/ -v`** — all tests must be green before execution phase
4. **Create `docker/Dockerfile`** — Lead sets up CPU dev environment (§7)
5. **Scaffold `feature/scribble`** — create empty class stubs (§5.1 structure)
6. **Scaffold `feature/example`** — create empty class stubs (§5.2 structure)

---

## 13. Acceptance Criteria (Definition of Done)

### Code
- [ ] `pytest tests/ -v` passes for all 3 methods
- [ ] `configs/config.yaml` has clean section per method
- [ ] Notebooks 01-04 run end-to-end without errors
- [ ] `app/demo.py` starts and all 3 tabs produce output

### Results
- [ ] Scribble + Example evaluated on COCO benchmark (PSNR/SSIM/LPIPS)
- [ ] All 5 DL model variants evaluated on benchmark
- [ ] Cross-method comparison table complete (`results/comparison/`)
- [ ] Qualitative grids generated for report

### Report
- [ ] `reports/main.tex` compiles to PDF without errors
- [ ] All 8 sections present
- [ ] All citations resolve
- [ ] Figures and tables render correctly

### Demo + Docker
- [ ] `docker compose up demo` starts Gradio without errors
- [ ] All 3 tabs produce colorized output
- [ ] Sample images included for quick demo

### Delivery
- [ ] All feature branches merged to `develop`
- [ ] `develop` merged to `main` and tagged `v1.0`
- [ ] README on `main` explains setup and usage
- [ ] Slides complete (15-20 slides)
