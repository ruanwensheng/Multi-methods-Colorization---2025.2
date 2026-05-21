"""
Evaluate ExampleColorizer on the shared COCO benchmark.

Requires prepare_benchmark.py to have been run first (builds benchmark_to_ref.json).

Usage:
    python tools/evaluate_example.py --tag pool-match
    python tools/evaluate_example.py --tag pool-match --max-images 50
"""

import os
import sys
import argparse
import json
import csv

import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deep_learning.utils import load_config, compute_metrics
from src.example_based.colorizer import ExampleColorizer


def evaluate(colorizer, benchmark_dir, ref_mapping, output_dir, max_images=None):
    """Run evaluation on benchmark images.

    Args:
        colorizer:     ExampleColorizer instance
        benchmark_dir: directory containing ground-truth color benchmark images
        ref_mapping:   dict {image_filename: reference_image_path}
        output_dir:    directory to write results
        max_images:    optional limit on number of images to evaluate
    Returns:
        list of per-image result dicts
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([
        f for f in os.listdir(benchmark_dir)
        if os.path.splitext(f)[1].lower() in extensions
        and not os.path.isdir(os.path.join(benchmark_dir, f))
    ])

    if max_images:
        images = images[:max_images]

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    results = []
    skipped = 0

    for img_name in tqdm(images, desc="Evaluating"):
        img_path = os.path.join(benchmark_dir, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            skipped += 1
            continue

        ref_path = ref_mapping.get(img_name)
        if ref_path is None or not os.path.exists(ref_path):
            skipped += 1
            continue

        ref_bgr = cv2.imread(ref_path)
        if ref_bgr is None:
            skipped += 1
            continue

        gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        result_bgr, info = colorizer.colorize(gray, ref_bgr)
        pred_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
            pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))

        metrics = compute_metrics(pred_rgb, gt_rgb)
        metrics["image"] = img_name
        metrics["time_seconds"] = info["time_seconds"]
        results.append(metrics)

        # Save visualisation for first 20 images
        if len(results) <= 20:
            n_panels = 3
            fig_h, fig_w = gt_rgb.shape[0], gt_rgb.shape[1]
            canvas = np.zeros((fig_h, fig_w * n_panels, 3), dtype=np.uint8)
            canvas[:, :fig_w] = cv2.cvtColor(
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB
            )
            canvas[:, fig_w:2 * fig_w] = pred_rgb
            canvas[:, 2 * fig_w:] = gt_rgb
            vis_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(vis_dir, f"eval_{img_name}"), vis_bgr)

    if skipped:
        print(f"Skipped {skipped} images (missing file or reference).")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ExampleColorizer on COCO benchmark")
    parser.add_argument("--tag", default="default", help="Run tag (used as output subdirectory)")
    parser.add_argument("--benchmark-dir", default=None, help="Benchmark image directory")
    parser.add_argument("--ref-mapping", default=None, help="Path to benchmark_to_ref.json")
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    benchmark_dir = args.benchmark_dir or os.path.join(cfg["paths"]["data_coco"], "benchmark")
    ref_mapping_path = args.ref_mapping or os.path.join(
        cfg["paths"]["data_coco"], "reference_pool", "benchmark_to_ref.json"
    )
    base_output = args.output_dir or os.path.join(cfg["paths"]["results_example"], "metrics")
    output_dir = os.path.join(base_output, args.tag)

    if not os.path.isdir(benchmark_dir):
        print(f"ERROR: Benchmark directory not found: {benchmark_dir}")
        print("Run 'python tools/download_coco.py' first.")
        sys.exit(1)

    if not os.path.exists(ref_mapping_path):
        print(f"ERROR: Reference mapping not found: {ref_mapping_path}")
        print("Run 'python tools/prepare_benchmark.py' first.")
        sys.exit(1)

    with open(ref_mapping_path) as f:
        ref_mapping = json.load(f)

    print(f"Loaded reference mapping: {len(ref_mapping)} entries")

    colorizer = ExampleColorizer(cfg=cfg)

    print(f"Evaluating on {benchmark_dir} ...")
    results = evaluate(colorizer, benchmark_dir, ref_mapping, output_dir, args.max_images)

    if not results:
        print("No results generated.")
        return

    # Per-image CSV
    fieldnames = ["image", "psnr", "ssim", "time_seconds"]

    csv_path = os.path.join(output_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Aggregate JSON
    aggregate = {
        "tag": args.tag,
        "method": "example_based",
        "num_images": len(results),
        "psnr_mean": float(np.mean([r["psnr"] for r in results])),
        "psnr_std": float(np.std([r["psnr"] for r in results])),
        "ssim_mean": float(np.mean([r["ssim"] for r in results])),
        "ssim_std": float(np.std([r["ssim"] for r in results])),
        "avg_time_sec": float(np.mean([r["time_seconds"] for r in results])),
    }


    json_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results — {args.tag} ({aggregate['num_images']} images)")
    print(f"{'=' * 50}")
    print(f"PSNR:     {aggregate['psnr_mean']:.2f} ± {aggregate['psnr_std']:.2f}")
    print(f"SSIM:     {aggregate['ssim_mean']:.4f} ± {aggregate['ssim_std']:.4f}")
    print(f"Avg time: {aggregate['avg_time_sec']:.3f} s/image")
    print(f"\nResults saved to: {output_dir}")

    # MLflow logging (optional)
    try:
        import mlflow
        mlflow.set_experiment("example-colorization-eval")
        with mlflow.start_run(run_name=args.tag):
            mlflow.log_params({"tag": args.tag, "benchmark_dir": benchmark_dir})
            mlflow.log_metrics({k: v for k, v in aggregate.items() if isinstance(v, float)})
            mlflow.log_artifact(csv_path)
            mlflow.log_artifact(json_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
