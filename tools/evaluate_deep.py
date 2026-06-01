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

# Pin caches to project drive BEFORE torch/HF imports — keeps C: from filling.
from src.deep_learning.bootstrap import setup_caches
setup_caches()

from src.deep_learning.utils import load_config, compute_metrics, visualize_result, rgb_to_lab, lab_to_rgb
from src.deep_learning.colorizer import DeepColorizer
from src.deep_learning.stats import bootstrap_ci, format_metric


# Comparison-model registry. Keys are CLI-friendly names; values are the
# (lazy) factories so test imports don't pay deoldify/diffusers import cost.
_COMPARISON_FACTORIES = {
    "zhang2017": ("src.deep_learning.pretrained", "Zhang2017Colorizer"),
    "deoldify":  ("src.deep_learning.pretrained", "DeOldifyColorizer"),
}


def build_colorizer(method=None, model_path=None, cfg=None, device="auto", temperature=0.38):
    """Build whichever colorizer the caller asked for.

    Either --method (a comparison-model key) OR --model-path (our Zhang16
    checkpoint) selects the colorizer. The rest of evaluate_deep is model-
    agnostic — it just calls .colorize() and aggregates metrics — so swapping
    the constructor is enough to reuse the pipeline for T3.3/3.4/3.5.
    """
    if method:
        method_lower = method.lower()
        if method_lower not in _COMPARISON_FACTORIES:
            raise ValueError(
                f"Unknown method '{method}'. Known: {list(_COMPARISON_FACTORIES)}"
            )
        mod_name, cls_name = _COMPARISON_FACTORIES[method_lower]
        import importlib
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls()
    if model_path is None:
        raise ValueError("Either --method or --model-path is required")
    return DeepColorizer(
        model_path=model_path, cfg=cfg, device=device, temperature=temperature
    )


def resolve_output_dir(cfg, tag=None, output_dir_override=None):
    """Resolve where eval results land.

    Priority:
      1. Explicit --output-dir override (caller knows best).
      2. results_deep/metrics/<tag>/ — separates T3.1 (pretrained) from T3.2
         (finetuned) so the second run doesn't clobber the first.
      3. results_deep/metrics/        — legacy default; preserved for back-compat.
    """
    if output_dir_override:
        return output_dir_override
    base = os.path.join(cfg["paths"]["results_deep"], "metrics")
    return os.path.join(base, tag) if tag else base


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
    parser.add_argument("--model-path", default=None,
                        help="Path to our Zhang16 checkpoint. Mutually exclusive with --method.")
    parser.add_argument("--method", default=None, choices=list(_COMPARISON_FACTORIES),
                        help="Use a comparison model (Zhang17/DeOldify) instead "
                             "of our Zhang16. Mutually exclusive with --model-path.")
    parser.add_argument("--test-dir", default=None, help="Directory of test images")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--tag", default=None,
                        help="Subdir name for outputs (e.g. 'pretrained' or 'finetuned'). "
                             "Routes results to results_deep/metrics/<tag>/. "
                             "Ignored if --output-dir is set.")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--device", default="auto", help="Device to use")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Defaults from config
    test_dir = args.test_dir or os.path.join(cfg["paths"]["data_coco"], "benchmark")
    output_dir = resolve_output_dir(cfg, tag=args.tag, output_dir_override=args.output_dir)

    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        print("Run 'python tools/download_coco.py' first.")
        sys.exit(1)

    if not args.method and not args.model_path:
        print("ERROR: must pass either --model-path or --method")
        sys.exit(2)
    if args.method and args.model_path:
        print("ERROR: --model-path and --method are mutually exclusive")
        sys.exit(2)

    # Load model (our Zhang16 or a comparison model)
    if args.method:
        print(f"Building comparison model: {args.method}")
    else:
        print(f"Loading model from {args.model_path}...")
    colorizer = build_colorizer(
        method=args.method,
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

    # MLflow logging — tag becomes the run_name so T3.1/T3.2 are searchable
    try:
        import mlflow
        mlflow.set_experiment("deep-colorization-eval")
        with mlflow.start_run(run_name=args.tag or "eval"):
            mlflow.log_params({
                "model_path": args.model_path or "",
                "method": args.method or "",
                "test_dir": test_dir,
                "tag": args.tag or "",
            })
            mlflow.log_metrics(aggregate)
            mlflow.log_artifact(csv_path)
            mlflow.log_artifact(json_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
