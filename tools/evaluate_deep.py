"""
Evaluate deep learning colorization model on test set.

Usage:
    python tools/evaluate_deep.py --model-path models/deep_learning/best_model.pth
    python tools/evaluate_deep.py --test-dir data/raw/coco2017/benchmark

Computes PSNR, SSIM, LPIPS and saves results.
"""

import os
import sys
import argparse
import json
import csv
import time

import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deep_learning.utils import load_config, compute_metrics, visualize_result, rgb_to_lab, lab_to_rgb
from src.deep_learning.colorizer import DeepColorizer
from src.deep_learning.stats import bootstrap_ci, format_metric


def evaluate(colorizer, test_dir, output_dir, max_images=None):
    """Run evaluation on a directory of test images.

    Returns:
        List of per-image result dicts.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([
        f for f in os.listdir(test_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    if max_images:
        images = images[:max_images]

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    results = []
    for img_name in tqdm(images, desc="Evaluating"):
        img_path = os.path.join(test_dir, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Ground truth RGB
        gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Create grayscale input
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Colorize
        result_bgr, info = colorizer.colorize(gray)
        pred_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # Resize to match if needed
        if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
            pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))

        # Compute metrics
        metrics = compute_metrics(pred_rgb, gt_rgb)
        metrics["image"] = img_name
        metrics["elapsed_sec"] = info["elapsed_sec"]
        results.append(metrics)

        # Save visualization for first 20 images
        if len(results) <= 20:
            visualize_result(
                gray, pred_rgb, gt_rgb,
                save_path=os.path.join(vis_dir, f"eval_{img_name}"),
                title=f"PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.3f}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate colorization model")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-dir", default=None, help="Directory of test images")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--device", default="auto", help="Device to use")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Defaults from config
    test_dir = args.test_dir or os.path.join(cfg["paths"]["data_coco"], "benchmark")
    output_dir = args.output_dir or os.path.join(cfg["paths"]["results_deep"], "metrics")

    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        print("Run 'python tools/download_coco.py' first.")
        sys.exit(1)

    # Load model
    print(f"Loading model from {args.model_path}...")
    colorizer = DeepColorizer(
        model_path=args.model_path,
        cfg=cfg,
        device=args.device,
        temperature=cfg["deep_learning"].get("temperature", 0.38),
    )

    # Evaluate
    print(f"Evaluating on {test_dir}...")
    results = evaluate(colorizer, test_dir, output_dir, args.max_images)

    if not results:
        print("No results generated.")
        return

    # Save per-image results CSV
    csv_path = os.path.join(output_dir, "test_metrics.csv")
    fieldnames = ["image", "psnr", "ssim", "elapsed_sec"]
    if "lpips" in results[0]:
        fieldnames.append("lpips")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Compute aggregate metrics with bootstrap 95% CIs (10k resamples, seed=42)
    aggregate = {"num_images": len(results)}
    for metric in ("psnr", "ssim", "lpips"):
        vals = [r[metric] for r in results if metric in r and r[metric] is not None]
        if not vals:
            continue
        ci = bootstrap_ci(vals, ci=0.95, n_boot=10000, seed=42)
        aggregate[f"{metric}_mean"]  = ci["mean"]
        aggregate[f"{metric}_std"]   = ci["std"]
        aggregate[f"{metric}_ci_lo"] = ci["ci_lo"]
        aggregate[f"{metric}_ci_hi"] = ci["ci_hi"]
    aggregate["ci_level"]   = 0.95
    aggregate["n_bootstrap"] = 10000
    aggregate["avg_time_sec"] = float(np.mean([r["elapsed_sec"] for r in results]))

    # Save aggregate
    json_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Print summary with CIs
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({aggregate['num_images']} images, "
          f"95% bootstrap CI, n_boot=10,000)")
    print(f"{'='*60}")
    for metric, prec, fmt in (("psnr", 2, "PSNR"), ("ssim", 4, "SSIM"), ("lpips", 4, "LPIPS")):
        if f"{metric}_mean" in aggregate:
            print(f"  {fmt:<6} {aggregate[f'{metric}_mean']:.{prec}f}  "
                  f"[95% CI {aggregate[f'{metric}_ci_lo']:.{prec}f}, "
                  f"{aggregate[f'{metric}_ci_hi']:.{prec}f}]  "
                  f"(std {aggregate[f'{metric}_std']:.{prec}f})")
    print(f"  Avg time: {aggregate['avg_time_sec']:.3f} s/image")
    print(f"\nResults saved to: {output_dir}")

    # MLflow logging
    try:
        import mlflow
        mlflow.set_experiment("deep-colorization-eval")
        with mlflow.start_run():
            mlflow.log_params({"model_path": args.model_path, "test_dir": test_dir})
            mlflow.log_metrics(aggregate)
            mlflow.log_artifact(csv_path)
            mlflow.log_artifact(json_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
