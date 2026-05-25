"""
Compare multiple colorization methods on the shared benchmark.

Usage:
    python tools/compare_methods.py
    python tools/compare_methods.py --methods zhang16,deoldify --max-images 50

Runs all available methods on the benchmark test set and produces
comparison metrics and side-by-side visualizations.
"""

import os
import sys
import argparse
import json
import csv

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pin caches to project drive BEFORE torch/HF imports — keeps C: from filling.
from src.deep_learning.bootstrap import setup_caches
setup_caches()

from src.deep_learning.utils import load_config, compute_metrics
from src.deep_learning.colorizer import DeepColorizer
from src.deep_learning.pretrained import get_comparison_models
from src.deep_learning.stats import bootstrap_ci


def load_all_models(cfg, model_path=None, device="auto"):
    """Load all available colorization models.

    Returns:
        dict mapping model_name -> colorizer instance.
    """
    models = {}

    # Our Zhang16 model
    if model_path and os.path.exists(model_path):
        try:
            colorizer = DeepColorizer(model_path=model_path, cfg=cfg, device=device)
            models["Zhang16 (Ours)"] = colorizer
            print(f"  Loaded: Zhang16 (Ours) from {model_path}")
        except Exception as e:
            print(f"  Failed to load Zhang16: {e}")
    else:
        # Use untrained model as placeholder
        colorizer = DeepColorizer(cfg=cfg, device=device)
        models["Zhang16 (Ours)"] = colorizer
        print("  Loaded: Zhang16 (Ours) - untrained")

    # Pre-trained comparison models
    comparison = get_comparison_models(cfg)
    for name, model in comparison.items():
        models[name] = model
        print(f"  Loaded: {name}")

    return models


def compare_on_image(models, img_bgr):
    """Run all models on a single image and collect results.

    Returns:
        dict mapping model_name -> (pred_rgb, metrics, elapsed).
    """
    gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    results = {}
    for name, model in models.items():
        try:
            if hasattr(model, "colorize"):
                result_bgr, info = model.colorize(gray)
            else:
                continue

            pred_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
                pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))

            metrics = compute_metrics(pred_rgb, gt_rgb)
            results[name] = (pred_rgb, metrics, info.get("elapsed_sec", 0))
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    return results, gray, gt_rgb


def save_comparison_grid(gray, gt_rgb, results, save_path, img_name=""):
    """Save a side-by-side comparison visualization."""
    n_methods = len(results) + 2  # input + ground truth + methods
    fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 3.5))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Input", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=9)
    axes[1].axis("off")

    for i, (name, (pred, metrics, _)) in enumerate(results.items()):
        axes[i + 2].imshow(pred)
        psnr = metrics.get("psnr", 0)
        ssim = metrics.get("ssim", 0)
        axes[i + 2].set_title(f"{name}\nPSNR:{psnr:.1f} SSIM:{ssim:.3f}", fontsize=8)
        axes[i + 2].axis("off")

    if img_name:
        fig.suptitle(img_name, fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare colorization methods")
    parser.add_argument("--model-path", default="models/deep_learning/best_model.pth",
                        help="Path to our trained model")
    parser.add_argument("--test-dir", default=None, help="Test image directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-images", type=int, default=50, help="Max images to compare")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--device", default="auto", help="Device")
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_dir = args.test_dir or os.path.join(cfg["paths"]["data_coco"], "benchmark")
    output_dir = args.output_dir or os.path.join(cfg["paths"]["results_deep"], "comparison")

    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "grids")
    os.makedirs(vis_dir, exist_ok=True)

    # Load models
    print("Loading models...")
    models = load_all_models(cfg, args.model_path, args.device)
    print(f"  {len(models)} model(s) loaded\n")

    # Get test images
    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([
        f for f in os.listdir(test_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])[:args.max_images]

    # Run comparison
    all_results = {name: [] for name in models}

    for img_name in tqdm(images, desc="Comparing"):
        img_path = os.path.join(test_dir, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        results, gray, gt_rgb = compare_on_image(models, img_bgr)

        for name, (pred, metrics, elapsed) in results.items():
            all_results[name].append({
                "image": img_name,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics.get("lpips", None),
                "elapsed_sec": elapsed,
            })

        # Save comparison grid for first 10 images
        if len(all_results[list(models.keys())[0]]) <= 10:
            save_comparison_grid(
                gray, gt_rgb, results,
                os.path.join(vis_dir, f"compare_{img_name}"),
                img_name=img_name
            )

    # Aggregate with bootstrap 95% CIs (10k resamples, seed=42)
    print(f"\n{'='*95}")
    print(f"{'Method':<25} {'PSNR [95% CI]':>22} {'SSIM [95% CI]':>22} {'Time/img':>12}")
    print(f"{'='*95}")

    summary = {}
    for name, results_list in all_results.items():
        if not results_list:
            continue
        psnr_ci = bootstrap_ci([r["psnr"] for r in results_list], ci=0.95, n_boot=10000, seed=42)
        ssim_ci = bootstrap_ci([r["ssim"] for r in results_list], ci=0.95, n_boot=10000, seed=42)
        time_mean = float(np.mean([r["elapsed_sec"] for r in results_list]))

        psnr_str = f"{psnr_ci['mean']:.2f} [{psnr_ci['ci_lo']:.2f}, {psnr_ci['ci_hi']:.2f}]"
        ssim_str = f"{ssim_ci['mean']:.4f} [{ssim_ci['ci_lo']:.4f}, {ssim_ci['ci_hi']:.4f}]"
        print(f"{name:<25} {psnr_str:>22} {ssim_str:>22} {time_mean:>10.3f}s")

        entry = {
            "num_images":  len(results_list),
            "psnr_mean":   psnr_ci["mean"],
            "psnr_std":    psnr_ci["std"],
            "psnr_ci_lo":  psnr_ci["ci_lo"],
            "psnr_ci_hi":  psnr_ci["ci_hi"],
            "ssim_mean":   ssim_ci["mean"],
            "ssim_std":    ssim_ci["std"],
            "ssim_ci_lo":  ssim_ci["ci_lo"],
            "ssim_ci_hi":  ssim_ci["ci_hi"],
            "avg_time_sec": time_mean,
        }

        # LPIPS if any image has it
        lpips_vals = [r["lpips"] for r in results_list if r.get("lpips") is not None]
        if lpips_vals:
            lpips_ci = bootstrap_ci(lpips_vals, ci=0.95, n_boot=10000, seed=42)
            entry.update({
                "lpips_mean":  lpips_ci["mean"],
                "lpips_std":   lpips_ci["std"],
                "lpips_ci_lo": lpips_ci["ci_lo"],
                "lpips_ci_hi": lpips_ci["ci_hi"],
            })

        entry["ci_level"] = 0.95
        entry["n_bootstrap"] = 10000
        summary[name] = entry

    print(f"{'='*95}")

    # Save summary
    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-image CSV
    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "psnr", "ssim", "lpips", "elapsed_sec"])
        for name, results_list in all_results.items():
            for r in results_list:
                writer.writerow([r["image"], name, r["psnr"], r["ssim"], r.get("lpips", ""), r["elapsed_sec"]])

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
