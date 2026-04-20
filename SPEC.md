# SPEC: Deep Learning Colorization (`feature/deep`)

> **Status:** Living document + strict acceptance criteria
> **Branch:** `feature/deep`
> **Owner:** Minh Kwang
> **Last updated:** 2026-04-13

---

## 1. Problem & Motivation

**Problem:** Given a grayscale (single-channel L) image, predict plausible chrominance (ab channels) to produce a full-color image.

**Why deep learning?** Traditional colorization methods (scribble-based, example-based) require manual input or reference images. Deep learning approaches learn color distributions from large datasets and can colorize automatically — no user interaction required.

**Why compare 4 categories?** The deep learning colorization landscape has evolved through distinct paradigms — from classification CNNs (2016) to interactive systems (2017) to GANs (2019-2020) to diffusion models (2022-2023). Comparing them on the same benchmark reveals each paradigm's strengths, weaknesses, and trade-offs (speed, quality, saturation, hallucination).

**Course context:** This is one of three methods (scribble / example-based / deep learning) being developed in parallel on separate branches. Cross-method comparison happens after all three branches merge into `main`.

---

## 2. Objective

Deliver a **complete deep learning colorization pipeline** that:

1. Reimplements Zhang et al. 2016 "Colorful Image Colorization" from scratch
2. Evaluates **5 model variants** across **4 DL categories** on a shared COCO 2017 benchmark
3. Produces quantitative metrics (PSNR, SSIM, LPIPS), qualitative visualizations, and a written report
4. Is fully reproducible via CLI tools and documented workflows

### The 5 Model Variants

| # | Category | Model | Source | Training on This Branch |
|---|----------|-------|--------|------------------------|
| 1 | **CNN** | Zhang16 Pretrained | Our reimplementation + official ECCV weights | Inference only |
| 2 | **CNN** | Zhang16 Fine-tuned | Our reimplementation + fine-tuned on COCO 2017 | Fine-tune from pretrained |
| 3 | **Interactive CNN** | Zhang17 (SIGGRAPH) | Official `colorizers` package, auto mode (zero hints) | Inference only |
| 4 | **GAN** | DeOldify | Official pretrained (NoGAN / Self-Attention GAN) | Inference only |
| 5 | **Diffusion** | ControlNet + SD 2.1 | HuggingFace `neurallove/controlnet-sd21-colorization-diffusers` | Inference only |

**Key distinction:** Only Zhang16 (#1 and #2) is our code. Models #3-#5 use third-party pretrained weights — we wrap them with a unified inference API for fair comparison.

---

## 3. Theoretical Grounding

Each category has a **theory paper** (the idea) and a **practical model** (what we run):

| Category | Theory Paper | Practical Model |
|----------|-------------|-----------------|
| **CNN** | Zhang 2016 — "Colorful Image Colorization" (arXiv:1603.08511) | Our reimplementation (Zhang16Net, ~31.6M params) |
| **Interactive CNN** | Zhang 2017 — "Real-Time User-Guided Image Colorization" (arXiv:1705.02999) | Official pretrained, automatic mode |
| **GAN** | ChromaGAN (Vitoria, WACV 2020) | DeOldify (Jason Antic) — industry standard for photo restoration |
| **Diffusion** | Palette (Saharia 2022, arXiv:2111.05826) | ControlNet (Zhang & Agrawala 2023, arXiv:2302.05543) + SD 2.1 |

Reference PDFs are stored in `documents/`.

### Zhang16 Architecture (Our Reimplementation)

- **Input:** L channel (B, 1, H, W), normalized to [-1, 1]
- **Encoder:** 8 convolutional blocks — Blocks 1-3 use stride-2 downsampling (H→H/8), Blocks 4-7 use dilated convolutions at H/8 resolution
- **Decoder:** Block 8 upsamples H/8→H/4 via transposed convolution, then 1x1 conv to num_classes
- **Output:** (B, 313, H/4, W/4) logits over quantized ab bins
- **Inference:** Softmax → upsample to full resolution → annealed-mean decoding (T=0.38) → ab values
- **Color space:** CIE Lab — L is luminance (structure), ab is chrominance (color)
- **Quantization:** 313 in-gamut ab bins from `pts_in_hull.npy` (from original paper)
- **Loss:** Class-rebalanced cross-entropy (Zhang 2016 Eq. 2-4) — upweights rare/saturated colors

---

## 4. Project Structure

```
feature/deep branch
├── CLAUDE.md                       # Agent instructions (WAT framework)
├── SPEC.md                         # This file
├── configs/
│   └── config.yaml                 # Deep learning configuration only
├── data/
│   └── raw/coco2017/               # COCO 2017 images (downloaded via tool)
│       ├── train2017/              # 118K training images
│       ├── val2017/                # 5K validation images
│       └── benchmark/              # 500 shared test images (subset of val)
├── documents/                      # Reference papers (PDFs) for DL method
├── models/
│   ├── pretrained/                 # Official pretrained weights (downloaded)
│   │   ├── zhang16_eccv.pth
│   │   └── zhang16_siggraph.pth
│   └── deep_learning/             # Our trained/fine-tuned checkpoints
│       ├── best_model.pth
│       └── last_model.pth
├── notebooks/
│   ├── 03_deep_learning_pipeline.ipynb   # Full pipeline walkthrough
│   └── 04_method_comparison.ipynb        # 4-category DL comparison
├── reports/
│   └── deep_learning/             # LaTeX report for DL method
│       ├── main.tex               # Root document
│       ├── sections/
│       │   ├── introduction.tex
│       │   ├── related_work.tex
│       │   ├── method.tex
│       │   ├── experiments.tex
│       │   ├── results.tex
│       │   ├── discussion.tex
│       │   └── conclusion.tex
│       ├── figures/               # Figures referenced by LaTeX (copied/symlinked from results)
│       ├── references.bib         # BibTeX citations
│       └── main.pdf               # Compiled output
├── results/
│   └── deep_learning/
│       ├── metrics/               # Per-model evaluation CSVs and JSONs
│       ├── comparison/            # Cross-model comparison outputs
│       └── figures/               # Visualizations and comparison grids
├── src/
│   ├── __init__.py
│   └── deep_learning/             # Core module
│       ├── __init__.py
│       ├── model.py               # Zhang16Net + Zhang16Regression architectures
│       ├── colorizer.py           # DeepColorizer inference wrapper
│       ├── dataset.py             # ColorizationDataset + dataloader factory
│       ├── loss.py                # ClassRebalancedCELoss + HuberColorLoss
│       ├── quantize.py            # ABQuantizer (313 ab bins)
│       ├── train.py               # Trainer class (training loop)
│       ├── utils.py               # Color conversions, metrics, visualization
│       ├── pretrained.py          # Zhang17, DeOldify, ControlNet wrappers
│       └── pts_in_hull.npy        # Pre-computed ab bin centers
├── tests/                         # Unit tests for all modules
│   ├── conftest.py                # Shared fixtures
│   ├── test_model.py
│   ├── test_quantize.py
│   ├── test_loss.py
│   ├── test_dataset.py
│   ├── test_colorizer.py
│   ├── test_utils.py
│   ├── test_pretrained.py
│   └── test_train.py
├── tools/                         # CLI entry points
│   ├── download_coco.py           # Download COCO 2017 dataset
│   ├── download_pretrained.py     # Download official pretrained weights
│   ├── train_deep.py              # Train/fine-tune Zhang16
│   ├── evaluate_deep.py           # Evaluate single model
│   └── compare_methods.py         # Compare all 5 model variants
├── workflows/
│   └── deep_learning_pipeline.md  # Step-by-step SOP
├── requirements.txt               # Python dependencies (DL-focused)
└── pytest.ini                     # Test configuration
```

### What does NOT belong on this branch
- Scribble-based method code/config
- Example-based method code/config
- Demo API server (FastAPI) — postponed to post-merge
- Cross-method (scribble vs example vs DL) comparison — happens on `main`

---

## 5. Pipeline Commands

All commands run from project root with conda environment `AI` activated.

### Step 1: Download COCO 2017

```bash
python tools/download_coco.py --split both --benchmark-size 500
```
- Downloads train (118K, ~18GB) and val (5K, ~1GB)
- Creates 500-image benchmark subset from val (seed=42)
- Skips already-downloaded files

### Step 2: Download Pretrained Weights

```bash
python tools/download_pretrained.py --model zhang16
```
- Downloads Zhang16 ECCV and SIGGRAPH weights to `models/pretrained/`

### Step 3: Fine-tune Zhang16

```bash
python tools/train_deep.py --config configs/config.yaml
```
- Loads pretrained ECCV weights, fine-tunes on COCO 2017
- Logs to MLflow experiment `deep-colorization`
- Saves best checkpoint to `models/deep_learning/best_model.pth`

Override options:
```bash
python tools/train_deep.py --epochs 10 --batch-size 4 --lr 0.0001
```

### Step 4: Evaluate Each Model

```bash
# Zhang16 Pretrained (official weights, no fine-tuning)
python tools/evaluate_deep.py --model-path models/pretrained/zhang16_eccv.pth --tag pretrained

# Zhang16 Fine-tuned
python tools/evaluate_deep.py --model-path models/deep_learning/best_model.pth --tag finetuned
```
- Computes PSNR, SSIM, LPIPS on benchmark
- Saves per-image CSV and aggregate JSON to `results/deep_learning/metrics/`

### Step 5: Compare All 5 Models

```bash
python tools/compare_methods.py --max-images 50
```
- Runs all 5 variants on benchmark subset
- Generates side-by-side comparison grids
- Outputs summary metrics table and per-image CSV

### Step 6: View Experiment Tracking

```bash
mlflow ui
```
- Opens dashboard at http://localhost:5000

---

## 6. Experiments

### 6.1 Training Experiment: Zhang16 Fine-tuning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base weights | Zhang16 ECCV pretrained | Start from a good initialization |
| Dataset | COCO 2017 train (118K) | Large-scale diverse dataset |
| Epochs | 50 | Sufficient for fine-tuning (not from scratch) |
| Batch size | 8 | Fits 6GB VRAM with AMP |
| Learning rate | 0.0002 | Adam default, conservative for fine-tuning |
| Optimizer | Adam (beta1=0.9, beta2=0.999) | Standard for vision tasks |
| Scheduler | StepLR (step=20, gamma=0.5) | Halve LR at epoch 20 and 40 |
| Loss | Class-rebalanced cross-entropy | Encourages vivid colors (Zhang 2016 Eq. 2-4) |
| AMP | Enabled | Required for 6GB VRAM |
| Input size | 256x256 | Balance between quality and memory |
| Validation | COCO 2017 val (5K) | Monitor overfitting |

### 6.2 Evaluation Experiment: 5-Model Comparison

All 5 models evaluated on the **same 500-image COCO 2017 benchmark** subset:

| Model | What We Measure |
|-------|----------------|
| Zhang16 Pretrained | Baseline — official weights, our inference code |
| Zhang16 Fine-tuned | Does fine-tuning on COCO improve over pretrained? |
| Zhang17 (auto) | How does the interactive model compare in automatic mode? |
| DeOldify | GAN approach — more saturated/artistic? |
| ControlNet | Diffusion approach — highest quality but slowest? |

**Quantitative metrics:** PSNR, SSIM, LPIPS (per-image and aggregate mean/std)
**Qualitative outputs:** Side-by-side comparison grids (grayscale → each model → ground truth)

---

## 7. Evaluation & Metrics

### Metrics Definition

| Metric | What It Measures | Better | Range |
|--------|-----------------|--------|-------|
| **PSNR** | Pixel-level fidelity (peak signal-to-noise ratio) | Higher | [0, ~50] dB |
| **SSIM** | Structural similarity (luminance, contrast, structure) | Higher | [0, 1] |
| **LPIPS** | Perceptual similarity (learned feature distance) | Lower | [0, 1] |

### Expected Results Format

```
results/deep_learning/
├── metrics/
│   ├── zhang16_pretrained_metrics.json    # {"psnr_mean": ..., "ssim_mean": ..., "lpips_mean": ...}
│   ├── zhang16_finetuned_metrics.json
│   ├── zhang17_metrics.json
│   ├── deoldify_metrics.json
│   ├── controlnet_metrics.json
│   └── per_image_metrics.csv             # All models, all images, all metrics
├── comparison/
│   ├── comparison_summary.json           # Aggregate table
│   └── comparison_grids/                 # Visual comparison images
└── figures/
    ├── metrics_bar_chart.png             # PSNR/SSIM/LPIPS comparison
    ├── qualitative_grid.png             # Best/worst cases per model
    └── training_curves.png              # Loss/metric curves for fine-tuning
```

---

## 8. Report (LaTeX)

The report is written in **LaTeX** and lives in `reports/deep_learning/`. This covers ONLY the deep learning method — it will later be integrated into the full project report on `main`.

### Format
- **Engine:** pdflatex (or xelatex if Unicode fonts needed)
- **Style:** IEEE conference format (`IEEEtran` class) — standard for CV papers
- **Citations:** BibTeX (`references.bib`)
- **Compile:** `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

### Structure

```
reports/deep_learning/
├── main.tex                # Root document — includes all sections
├── sections/
│   ├── introduction.tex    # Problem statement, why DL for colorization
│   ├── related_work.tex    # 4 paradigms: CNN → Interactive → GAN → Diffusion
│   ├── method.tex          # Zhang16 architecture, loss, quantization, training
│   ├── experiments.tex     # Dataset (COCO 2017), metrics, benchmark, hardware
│   ├── results.tex         # Quantitative table, qualitative grids, analysis
│   ├── discussion.tex      # Strengths/weaknesses, speed vs quality trade-offs
│   └── conclusion.tex      # Best approach for what, recommendations
├── figures/                # Figures for the report
│   ├── architecture.png    # Zhang16 network diagram
│   ├── ab_quantization.png # 313 ab bins visualization
│   ├── training_curves.png # Loss/metric over epochs
│   ├── metrics_comparison.png  # PSNR/SSIM/LPIPS bar chart
│   └── qualitative_grid.png   # Side-by-side model outputs
├── references.bib          # BibTeX entries for all cited papers
└── main.pdf                # Compiled output
```

### Section Details

| Section | Content | Key Elements |
|---------|---------|-------------|
| **Introduction** | Colorization problem, motivation for DL approaches | Problem definition, figure showing grayscale→color |
| **Related Work** | Evolution: CNN (2016) → Interactive (2017) → GAN (2020) → Diffusion (2022) | Citation of all 6 papers, taxonomy table |
| **Method** | Zhang16 architecture deep-dive | Network diagram, Eq. for loss (Eq. 2-4), ab quantization, annealed-mean |
| **Experiments** | Setup: COCO 2017, 5 models, 3 metrics, hardware specs | Table of hyperparameters, benchmark description |
| **Results** | Quantitative: metrics table. Qualitative: visual comparison grids | PSNR/SSIM/LPIPS table, best/worst case figures |
| **Discussion** | Trade-off analysis across paradigms | Speed comparison, color saturation analysis, failure cases |
| **Conclusion** | Summary of findings, when to use which approach | Practical recommendations |

### BibTeX Entries Required

```bibtex
@inproceedings{zhang2016colorful,        % CNN - Zhang 2016
@inproceedings{zhang2017realtime,         % Interactive CNN - Zhang 2017
@inproceedings{vitoria2020chromagan,      % GAN theory - ChromaGAN
@misc{antic2019deoldify,                  % GAN practical - DeOldify
@inproceedings{saharia2022palette,        % Diffusion theory - Palette
@inproceedings{zhang2023controlnet,       % Diffusion practical - ControlNet
@inproceedings{lin2014coco,               % Dataset - COCO
@inproceedings{zhang2018lpips,            % Metric - LPIPS
```

### Why LaTeX?
- Equations render properly (loss functions, annealed-mean formula)
- BibTeX handles citations consistently
- IEEE format matches course/publication standards
- Figures and tables are publication-quality
- Modular `\input{}` structure makes it easy to integrate into the full report on `main` later

---

## 9. Code Style & Conventions

| Convention | Rule |
|------------|------|
| **Language** | Python 3.10+ |
| **Framework** | PyTorch 2.x |
| **Color space** | CIE Lab everywhere (L input, ab output) |
| **Normalization** | L: [-1, 1], ab: [-110, 110] raw or [-1, 1] normalized |
| **Image format** | OpenCV BGR for I/O, Lab internally, RGB for metrics |
| **Config** | Single YAML file (`configs/config.yaml`), loaded via `load_config()` |
| **CLI tools** | `argparse` in `tools/`, import from `src.deep_learning` |
| **Inference API** | All models expose `colorize(gray_image) -> (result_bgr, info_dict)` |
| **Experiment tracking** | MLflow for all runs (training, evaluation, comparison) |
| **Device handling** | Auto-detect via `get_device(cfg)`, support cuda and cpu |
| **Reproducibility** | Fixed seeds where applicable, benchmark subset seed=42 |

---

## 10. Testing Strategy

### Unit Tests (existing, `tests/`)

| Test File | What It Covers |
|-----------|---------------|
| `test_model.py` | Forward pass shapes, predict_ab output, parameter counts |
| `test_quantize.py` | Encode/decode, bin validity, class weights |
| `test_loss.py` | Loss computation, gradient flow, size mismatch handling |
| `test_dataset.py` | Data loading, normalization ranges, augmentation |
| `test_colorizer.py` | End-to-end inference, input format handling, determinism |
| `test_utils.py` | Color conversions roundtrip, metrics, config loading |
| `test_pretrained.py` | API consistency, model categories, factory function |
| `test_train.py` | Training step, checkpoint save/load roundtrip |

### Run Tests

```bash
pytest tests/ -v --timeout=60
```

### What Tests Do NOT Cover (and shouldn't on this branch)

- Integration with scribble/example methods
- Demo API endpoints
- Full training convergence (too slow for CI)

---

## 11. Boundaries

### Always Do
- Use conda environment `AI`
- Run all experiments on the shared 500-image COCO 2017 benchmark
- Log all experiments to MLflow
- Use the unified `colorize()` API for all model comparisons
- Keep `config.yaml` as the single source of truth for hyperparameters
- Save checkpoints to `models/deep_learning/`
- Save results to `results/deep_learning/`

### Ask First Before
- Changing the Zhang16Net architecture (it should match the paper)
- Modifying the 313 ab quantization bins
- Adding new comparison models beyond the 5 defined
- Running ControlNet (requires ~10GB VRAM, may need RTX 2080 Ti)
- Re-downloading COCO 2017 (18GB+ bandwidth)
- Creating or overwriting workflow files
- Committing large files (model weights, dataset) to git

### Never Do
- Add scribble or example-based method code to this branch
- Build a demo API on this branch (postponed to post-merge on `main`)
- Commit `main.pdf` without recompiling from source (always compile from `.tex`)
- Delete or overwrite the `documents/` reference papers
- Store API keys or secrets outside `.env`
- Commit model weights or dataset files to git
- Use DDColor or other transformer-based models (not in our 4 categories)
- Skip MLflow logging for any experiment run

---

## 12. Acceptance Criteria (Definition of Done)

This branch is ready to merge into `main` when ALL of the following are true:

- [ ] **Data:** COCO 2017 downloaded with 500-image benchmark subset
- [ ] **Pretrained weights:** Zhang16 ECCV weights downloaded
- [ ] **Fine-tuning:** Zhang16 fine-tuned on COCO 2017, best checkpoint saved
- [ ] **Evaluation — Zhang16 Pretrained:** PSNR, SSIM, LPIPS computed on benchmark
- [ ] **Evaluation — Zhang16 Fine-tuned:** PSNR, SSIM, LPIPS computed on benchmark
- [ ] **Evaluation — Zhang17:** PSNR, SSIM, LPIPS computed on benchmark
- [ ] **Evaluation — DeOldify:** PSNR, SSIM, LPIPS computed on benchmark
- [ ] **Evaluation — ControlNet:** PSNR, SSIM, LPIPS computed on benchmark
- [ ] **Comparison:** Summary table + visual grids for all 5 models
- [ ] **MLflow:** All experiments logged and viewable
- [ ] **Notebook 03:** Pipeline runs end-to-end
- [ ] **Notebook 04:** Shows 5-model comparison with metrics and visualizations
- [ ] **Tests:** All unit tests pass (`pytest tests/ -v`)
- [ ] **Report:** LaTeX report in `reports/deep_learning/` compiles to PDF with all sections, tables, figures, and citations
- [ ] **Config:** `config.yaml` contains only deep learning settings (no scribble/example)
- [ ] **Requirements:** Only DL-relevant dependencies (no FastAPI/uvicorn)
- [ ] **Clean git:** No uncommitted large files, `.gitignore` covers data/models/mlruns

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| VRAM overflow (6GB GTX 1660) | Training crashes | AMP enabled, reduce batch_size to 4 or 2 |
| ControlNet needs >6GB | Cannot run on GTX 1660 | Use RTX 2080 Ti, or run with CPU offloading |
| DeOldify/ControlNet deps conflict | Import errors | Install in separate conda env if needed, document in workflow |
| COCO download fails | Blocked progress | Re-run script (idempotent), use university network |
| Fine-tuning diverges | Bad results | Start from pretrained, use conservative LR, monitor val loss |
| Comparison models unavailable | Incomplete comparison | Mark as "N/A" in results, document why |

---

## Appendix: Dependencies

**Core (always needed):**
- torch, torchvision (CUDA 12.1)
- lpips, torchinfo
- opencv-python, scikit-image, pillow
- numpy, scipy, scikit-learn, pandas
- matplotlib
- mlflow
- jupyter, jupyterlab, ipywidgets
- tqdm, PyYAML, python-dotenv, requests, colorama

**Comparison models (install as needed):**
- `colorizers` — for Zhang17 (from richzhang/colorization)
- `deoldify`, `fastai` — for DeOldify
- `diffusers`, `transformers`, `accelerate` — for ControlNet

**Not needed on this branch:**
- ~~fastapi, uvicorn~~ (demo API — postponed)
