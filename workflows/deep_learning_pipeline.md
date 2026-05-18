# Deep Learning Colorization Pipeline

## Objective
Train and evaluate a CNN-based automatic colorization model (Zhang et al. 2016) 
on COCO 2017, and compare with:
- GAN: DeOldify (theory: ChromaGAN, Vitoria 2020)
- Diffusion: ControlNet + SD (theory: Palette, Saharia 2022)

## Prerequisites
- Python 3.10+ with CUDA PyTorch installed
- GPU with 6+ GB VRAM (GTX 1660 Super minimum)
- ~20GB disk space for COCO 2017 dataset

## Pipeline Steps

### Step 1: Download / Extract Data
```bash
python tools/download_coco.py --source local-zip --split all --benchmark-size 1000
```
- Extracts COCO 2017 train (118K), val (5K), test (40,670), annotations (6 JSONs)
- Creates the shared **benchmark subset (1,000 images from test2017, seed=42)** — held out from
  training and best-ckpt selection (see SPEC.md §6.3 for the leakage policy)
- Saves to `data/raw/coco2017/`
- Alternate sources: `--source kaggle` (kagglehub) or `--source http` (cocodataset.org with Range resume)

### Step 2: Download Pre-trained Weights
```bash
python tools/download_pretrained.py --model zhang16
```
- Downloads Zhang 2016 ECCV and SIGGRAPH weights
- Saves to `models/pretrained/`

### Step 3: Train / Fine-tune Model
```bash
python tools/train_deep.py --config configs/config.yaml
```
- Trains Zhang16Net on COCO 2017
- Logs to MLflow experiment "deep-colorization"
- Saves best checkpoint to `models/deep_learning/best_model.pth`

**Override options:**
```bash
python tools/train_deep.py --epochs 10 --batch-size 4 --lr 0.0001
```

### Step 4: Evaluate
```bash
python tools/evaluate_deep.py --model-path models/deep_learning/best_model.pth
```
- Runs on the held-out **1,000-image stratified test2017 benchmark**
  (`data/raw/coco2017/benchmark/` — built by Step 1)
- Computes PSNR, SSIM, LPIPS per image
- Reports each metric as **mean + 95% bootstrap CI** (10,000 percentile resamples, seed=42)
- Saves per-image CSV + aggregate JSON (with `_mean`, `_std`, `_ci_lo`, `_ci_hi` fields)
  to `results/deep_learning/metrics/`
- See `docs/benchmark_methodology.md` for the full design

### Step 5: Compare Methods
```bash
python tools/compare_methods.py --max-images 1000
```
- Runs all available models on the benchmark
- Side-by-side comparison grids + per-model metrics table with inline 95% CIs
- Saves to `results/deep_learning/comparison/`

### Step 6: View Results
```bash
mlflow ui
```
- Opens MLflow dashboard at http://localhost:5000
- View experiments, runs, metrics, and artifacts

## Inputs
- `configs/config.yaml` - All configuration
- `data/raw/coco2017/` - Training and test images
- `models/pretrained/` - Pre-trained weights for fine-tuning

## Outputs
- `models/deep_learning/best_model.pth` - Best trained checkpoint
- `results/deep_learning/metrics/` - Evaluation results (CSV, JSON)
- `results/deep_learning/comparison/` - Cross-method comparison
- `mlruns/` - MLflow experiment data

## Edge Cases
- **Out of VRAM**: Reduce `batch_size` to 4 or 2, enable `use_amp: true`
- **No GPU**: Set `device: cpu` in config (training will be very slow)
- **Download fails**: Re-run download script (skips already-downloaded files)
- **Training diverges**: Reduce learning rate, check data preprocessing
