# Agent Instructions — `feature/deep` branch

> This file is loaded at the start of every session. Keep it concise and actionable.

## Project Identity

**Multi-Methods Image Colorization** — Computer vision course project comparing three colorization approaches (scribble-based, example-based, deep learning) on separate branches.

**This branch** implements the **deep learning method**: 4 DL categories, 5 model variants, evaluated on a shared COCO 2017 benchmark.

## Quick Orientation

| What | Where |
|------|-------|
| Full specification (models, pipeline, acceptance criteria) | `SPEC.md` |
| Task breakdown and current progress | `tasks/todo.md` |
| Implementation plan with dependencies | `tasks/plan.md` |
| Step-by-step pipeline SOP | `workflows/deep_learning_pipeline.md` |
| All hyperparameters and paths | `configs/config.yaml` |
| Reference papers (PDFs) | `documents/` |

**Start here:** Read `tasks/todo.md` to see what's done and what's next.

## Key Directories

```
src/deep_learning/     # Core module: model, training, evaluation, inference
tools/                 # CLI entry points: download, train, evaluate, compare
tests/                 # Unit tests (pytest) for all modules
configs/               # config.yaml — single source of truth for hyperparams
notebooks/             # 03 = DL pipeline, 04 = 5-model comparison
workflows/             # Pipeline documentation (SOPs)
reports/deep_learning/ # LaTeX report (IEEEtran format)
data/raw/coco2017/     # Dataset (downloaded via tools/download_coco.py)
models/                # pretrained/ and deep_learning/ checkpoints
results/deep_learning/ # metrics/, comparison/, figures/
```

## Environment

- **Conda env:** `AI` — always activate before running anything
- **GPU:** GTX 1660 Super (6GB) min / RTX 2080 Ti (11GB) for ControlNet
- **Python:** 3.10+, PyTorch 2.x with CUDA 12.1
- **Tracking:** MLflow (experiment: `deep-colorization`)
- **Tests:** `pytest tests/ -v --timeout=60`

## The 5 Model Variants

| # | Category | Model | Our Code? |
|---|----------|-------|-----------|
| 1 | CNN | Zhang16 Pretrained (ECCV weights) | Yes — reimplementation |
| 2 | CNN | Zhang16 Fine-tuned (COCO 2017) | Yes — fine-tuned |
| 3 | Interactive CNN | Zhang17 SIGGRAPH (auto mode) | No — official package |
| 4 | GAN | DeOldify (NoGAN) | No — pretrained wrapper |
| 5 | Diffusion | ControlNet + SD 2.1 | No — HuggingFace pipeline |

All models expose a unified API: `colorize(gray_image) -> (result_bgr, info_dict)`

## Operating Principles

This project uses the **WAT framework** (Workflows, Agents, Tools):
- **Workflows** (`workflows/`): Markdown SOPs — the instructions
- **Agent** (you): Intelligent coordination — read workflows, run tools, handle failures
- **Tools** (`tools/`): Deterministic Python scripts — the execution

**Key rules:**
1. Check `tools/` for existing scripts before building anything new
2. When something fails: read the error, fix the script, retest, update the workflow
3. Don't create or overwrite workflow files without asking
4. All experiments must be logged to MLflow
5. Keep `config.yaml` as the single source of truth

## Conventions

| Convention | Rule |
|------------|------|
| Color space | CIE Lab everywhere (L input, ab output) |
| Normalization | L: [-1, 1], ab: [-110, 110] raw |
| Image I/O | OpenCV BGR for I/O, Lab internally, RGB for metrics |
| Config | Single YAML, loaded via `load_config()` |
| CLI tools | `argparse` in `tools/`, import from `src.deep_learning` |
| Device | Auto-detect via `get_device(cfg)` |
| Reproducibility | Fixed seeds, benchmark subset seed=42 |

## Boundaries

### Always Do
- Use conda environment `AI`
- Run experiments on the shared 500-image COCO 2017 benchmark
- Log all experiments to MLflow
- Use the unified `colorize()` API for model comparisons
- Save checkpoints to `models/deep_learning/`, results to `results/deep_learning/`

### Ask First Before
- Changing the Zhang16Net architecture
- Modifying the ab quantization bins
- Adding comparison models beyond the 5 defined
- Running ControlNet (requires ~10GB VRAM)
- Re-downloading COCO 2017 (18GB+ bandwidth)
- Creating or overwriting workflow files

### Never Do
- Add scribble or example-based code to this branch
- Build a demo API on this branch (postponed to `main`)
- Commit model weights or dataset files to git
- Store secrets outside `.env`
- Use DDColor or other models outside the 4 categories
- Skip MLflow logging for any experiment run
