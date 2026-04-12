# Multi-Methods Image Colorization

Automatic image colorization using deep learning. This project compares **4 deep learning paradigms** across **5 model variants** on a shared COCO 2017 benchmark, measuring PSNR, SSIM, and LPIPS.

> **Branch:** `feature/deep` — Deep learning method implementation
> **Course:** Computer Vision (2025)
> **Author:** Minh Kwang

## Problem

Given a grayscale image, predict plausible colors to produce a full-color image. Deep learning approaches learn color distributions from large datasets and colorize automatically — no user interaction required.

## Models Compared

| # | Category | Model | Description |
|---|----------|-------|-------------|
| 1 | **CNN** | Zhang16 Pretrained | Our PyTorch reimplementation of ["Colorful Image Colorization"](https://arxiv.org/abs/1603.08511) (ECCV 2016) with official weights |
| 2 | **CNN** | Zhang16 Fine-tuned | Same architecture, fine-tuned on COCO 2017 |
| 3 | **Interactive CNN** | Zhang17 SIGGRAPH | ["Real-Time User-Guided Colorization"](https://arxiv.org/abs/1705.02999) in automatic mode (zero hints) |
| 4 | **GAN** | DeOldify | Self-Attention GAN / NoGAN — industry standard for photo restoration |
| 5 | **Diffusion** | ControlNet + SD 2.1 | [ControlNet](https://arxiv.org/abs/2302.05543) conditioned colorization with Stable Diffusion |

**Theory papers** behind each practical model: Zhang 2016 (CNN), Zhang 2017 (Interactive), ChromaGAN / Vitoria 2020 (GAN), Palette / Saharia 2022 (Diffusion).

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.1 (6GB+ VRAM, 11GB for ControlNet)
- Conda

### Setup

```bash
# Create and activate conda environment
conda activate AI

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Download COCO 2017 dataset (~19GB)
python tools/download_coco.py --split both --benchmark-size 500

# 2. Download pretrained weights
python tools/download_pretrained.py --model zhang16

# 3. Fine-tune Zhang16 on COCO 2017
python tools/train_deep.py --config configs/config.yaml

# 4. Evaluate individual models
python tools/evaluate_deep.py --model-path models/pretrained/zhang16_eccv.pth --tag pretrained
python tools/evaluate_deep.py --model-path models/deep_learning/best_model.pth --tag finetuned

# 5. Compare all 5 models
python tools/compare_methods.py --max-images 50

# 6. View experiment tracking
mlflow ui  # Opens at http://localhost:5000
```

### Run Tests

```bash
pytest tests/ -v --timeout=60
```

## Project Structure

```
configs/config.yaml         # All hyperparameters (single source of truth)
src/deep_learning/          # Core module
    model.py                #   Zhang16Net architecture (~31.6M params)
    colorizer.py            #   Unified inference API
    dataset.py              #   COCO 2017 dataloader
    loss.py                 #   Class-rebalanced cross-entropy
    quantize.py             #   313 ab color bin quantization
    train.py                #   Training loop with MLflow + AMP
    utils.py                #   Color conversions, metrics, visualization
    pretrained.py           #   Zhang17, DeOldify, ControlNet wrappers
tools/                      # CLI entry points
    download_coco.py        #   Download COCO 2017
    download_pretrained.py  #   Download official weights
    train_deep.py           #   Train/fine-tune Zhang16
    evaluate_deep.py        #   Evaluate single model (PSNR/SSIM/LPIPS)
    compare_methods.py      #   Compare all 5 models
tests/                      # Unit tests (pytest)
notebooks/
    03_deep_learning_pipeline.ipynb   # Full pipeline walkthrough
    04_method_comparison.ipynb        # 5-model DL comparison
reports/deep_learning/      # LaTeX report (IEEEtran format)
```

## Evaluation Metrics

| Metric | Measures | Better |
|--------|----------|--------|
| **PSNR** | Pixel-level fidelity | Higher |
| **SSIM** | Structural similarity | Higher |
| **LPIPS** | Perceptual similarity (learned) | Lower |

All models evaluated on the same **500-image COCO 2017 benchmark** subset (seed=42).

## Architecture: Zhang16Net

Our reimplementation of Zhang et al. 2016:

- **Input:** Grayscale L channel (1, H, W)
- **Encoder:** 8 conv blocks — stride-2 downsampling (blocks 1-3) + dilated convolutions (blocks 4-7)
- **Decoder:** Transposed convolution + 1x1 conv to predict class probabilities
- **Output:** 313 quantized ab color bins, decoded via annealed-mean (T=0.38)
- **Loss:** Class-rebalanced cross-entropy — upweights rare/saturated colors
- **Color space:** CIE Lab (L = luminance input, ab = chrominance prediction)

## References

- Zhang et al., "Colorful Image Colorization", ECCV 2016 ([arXiv:1603.08511](https://arxiv.org/abs/1603.08511))
- Zhang et al., "Real-Time User-Guided Image Colorization with Learned Deep Priors", SIGGRAPH 2017 ([arXiv:1705.02999](https://arxiv.org/abs/1705.02999))
- Vitoria et al., "ChromaGAN: Adversarial Picture Colorization", WACV 2020
- Antic, "DeOldify", 2019 ([GitHub](https://github.com/jantic/DeOldify))
- Saharia et al., "Palette: Image-to-Image Diffusion Models", CVPR 2022 ([arXiv:2111.05826](https://arxiv.org/abs/2111.05826))
- Zhang & Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models", ICCV 2023 ([arXiv:2302.05543](https://arxiv.org/abs/2302.05543))

## Part of a Larger Project

This branch (`feature/deep`) implements one of three colorization methods:

| Branch | Method | Status |
|--------|--------|--------|
| `main` | Cross-method comparison | After all branches merge |
| `feature/scribble` | Scribble-based (Levin et al. 2004) | Separate branch |
| `feature/example` | Example-based color transfer | Separate branch |
| **`feature/deep`** | **Deep learning (this branch)** | **In progress** |
