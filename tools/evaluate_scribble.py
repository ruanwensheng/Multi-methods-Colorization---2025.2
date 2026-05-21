"""
Evaluate ScribbleColorizer on the shared COCO benchmark.

Requires prepare_benchmark.py to have been run first — it generates the oracle
scribble PNGs at data/raw/coco2017/benchmark/scribbles/<image_name>.png.

Each oracle scribble is a BGRA image where alpha=255 marks scribbled pixels
(true colour) and alpha=0 marks un-scribbled pixels (transparent).

Usage:
    python tools/evaluate_scribble.py --tag oracle-1pct
    python tools/evaluate_scribble.py --tag oracle-1pct --max-images 50
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
from src.scribble.colorizer import ScribbleColorizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scribble_as_bgr(scribble_path: str, gray_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGRA oracle scribble overlay to the BGR scribble image format
    expected by ScribbleColorizer.

    ScribbleColorizer identifies scribbled pixels as those where R != G or
    G != B.  Un-scribbled pixels must equal the gray image (R == G == B).

    Args:
        scribble_path: path to the BGRA PNG written by prepare_benchmark.py.
        gray_bgr:      (H, W, 3) BGR grayscale image (all channels identical).

    Returns:
        (H, W, 3) BGR uint8 scribble image.
    """
    bgra = cv2.imread(scribble_path, cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise FileNotFoundError(f"Cannot read scribble file: {scribble_path}")

    if bgra.shape[2] != 4:
        raise ValueError(f"Expected 4-channel BGRA, got shape {bgra.shape}: {scribble_path}")

    # Resize if needed (scribble PNG should match benchmark image, but be safe)
    H, W = gray_bgr.shape[:2]
    if bgra.shape[:2] != (H, W):
        bgra = cv2.resize(bgra, (W, H), interpolation=cv2.INTER_NEAREST)

    alpha = bgra[:, :, 3]           # 255 → scribbled, 0 → background
    scribble_mask = alpha == 255    # (H, W) bool

    # Start from gray, overwrite scribbled pixels with true colour
    scribble_bgr = gray_bgr.copy()
    scribble_bgr[scribble_mask] = bgra[scribble_mask, :3]

    return scribble_bgr, int(scribble_mask.sum())


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    colorizer: ScribbleColorizer,
    benchmark_dir: str,
    scribble_dir: str,
    output_dir: str,
    max_images: int = None,
) -> list:
    """Run evaluation on benchmark images using pre-built oracle scribbles.

    Args:
        colorizer:     ScribbleColorizer instance.
        benchmark_dir: Directory containing ground-truth color benchmark images.
        scribble_dir:  Directory containing BGRA oracle scribble PNGs
                       (produced by prepare_benchmark.py).
        output_dir:    Directory to write results.
        max_images:    Optional limit on number of images to evaluate.

    Returns:
        List of per-image result dicts.
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

        # Oracle scribble path: same stem, always .png
        stem = os.path.splitext(img_name)[0]
        scribble_path = os.path.join(scribble_dir, stem + ".png")
        if not os.path.exists(scribble_path):
            skipped += 1
            continue

        gt_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray_1ch = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray_1ch, cv2.COLOR_GRAY2BGR)

        try:
            scribble_bgr, n_scribbled = _load_scribble_as_bgr(scribble_path, gray_bgr)
        except Exception as exc:
            print(f"  WARNING: skipping {img_name} — {exc}")
            skipped += 1
            continue

        result_bgr, info = colorizer.colorize(gray_bgr, scribble_bgr)
        pred_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
            pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))

        metrics = compute_metrics(pred_rgb, gt_rgb)
        metrics["image"]        = img_name
        metrics["time_seconds"] = info["time_seconds"]
        metrics["n_scribbled"]  = n_scribbled
        results.append(metrics)

        # Save visualisation for first 20 images (4 panels)
        if len(results) <= 20:
            fig_h, fig_w = gt_rgb.shape[0], gt_rgb.shape[1]
            canvas = np.zeros((fig_h, fig_w * 4, 3), dtype=np.uint8)
            canvas[:, :fig_w]          = cv2.cvtColor(
                cv2.cvtColor(gray_1ch, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB
            )
            canvas[:, fig_w:2*fig_w]   = cv2.cvtColor(scribble_bgr, cv2.COLOR_BGR2RGB)
            canvas[:, 2*fig_w:3*fig_w] = pred_rgb
            canvas[:, 3*fig_w:]        = gt_rgb
            vis_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(vis_dir, f"eval_{img_name}"), vis_bgr)

    if skipped:
        print(f"Skipped {skipped} images (missing file or unreadable scribble).")

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ScribbleColorizer on COCO benchmark"
    )
    parser.add_argument("--tag",           default="default",
                        help="Run tag (used as output subdirectory)")
    parser.add_argument("--benchmark-dir", default=None,
                        help="Benchmark image directory")
    parser.add_argument("--scribble-dir",  default=None,
                        help="Oracle scribble PNG directory "
                             "(default: <benchmark-dir>/scribbles)")
    parser.add_argument("--output-dir",    default=None,
                        help="Output directory override")
    parser.add_argument("--max-images",    type=int, default=None,
                        help="Max images to evaluate")
    parser.add_argument("--config",        default="configs/config.yaml",
                        help="Config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    benchmark_dir = args.benchmark_dir or os.path.join(
        cfg["paths"]["data_coco"], "benchmark"
    )
    scribble_dir = args.scribble_dir or os.path.join(benchmark_dir, "scribbles")
    base_output  = args.output_dir or os.path.join(
        cfg["paths"]["results_scribble"], "metrics"
    )
    output_dir = os.path.join(base_output, args.tag)

    if not os.path.isdir(benchmark_dir):
        print(f"ERROR: Benchmark directory not found: {benchmark_dir}")
        print("Run 'python tools/download_coco.py' first.")
        sys.exit(1)

    if not os.path.isdir(scribble_dir):
        print(f"ERROR: Scribble directory not found: {scribble_dir}")
        print("Run 'python tools/prepare_benchmark.py' first.")
        sys.exit(1)

    colorizer = ScribbleColorizer(cfg=cfg)

    print(f"Evaluating on {benchmark_dir} ...")
    print(f"Scribbles from  {scribble_dir}")
    results = evaluate(colorizer, benchmark_dir, scribble_dir, output_dir, args.max_images)

    if not results:
        print("No results generated.")
        return

    # Per-image CSV
    fieldnames = ["image", "n_scribbled", "psnr", "ssim", "time_seconds"]

    csv_path = os.path.join(output_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Aggregate JSON
    aggregate = {
        "tag":             args.tag,
        "method":          "scribble_based",
        "num_images":      len(results),
        "psnr_mean":       float(np.mean([r["psnr"] for r in results])),
        "psnr_std":        float(np.std([r["psnr"]  for r in results])),
        "ssim_mean":       float(np.mean([r["ssim"] for r in results])),
        "ssim_std":        float(np.std([r["ssim"]  for r in results])),
        "avg_time_sec":    float(np.mean([r["time_seconds"] for r in results])),
        "avg_n_scribbled": float(np.mean([r["n_scribbled"]  for r in results])),
    }


    json_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Evaluation Results — {args.tag} ({aggregate['num_images']} images)")
    print(f"{'=' * 50}")
    print(f"PSNR:          {aggregate['psnr_mean']:.2f} ± {aggregate['psnr_std']:.2f}")
    print(f"SSIM:          {aggregate['ssim_mean']:.4f} ± {aggregate['ssim_std']:.4f}")
    print(f"Avg time:      {aggregate['avg_time_sec']:.3f} s/image")
    print(f"Avg scribbles: {aggregate['avg_n_scribbled']:.0f} px")
    print(f"\nResults saved to: {output_dir}")

    # MLflow logging (optional)
    try:
        import mlflow
        mlflow.set_experiment("scribble-colorization-eval")
        with mlflow.start_run(run_name=args.tag):
            mlflow.log_params({
                "tag":           args.tag,
                "benchmark_dir": benchmark_dir,
                "scribble_dir":  scribble_dir,
            })
            mlflow.log_metrics({k: v for k, v in aggregate.items()
                                 if isinstance(v, float)})
            mlflow.log_artifact(csv_path)
            mlflow.log_artifact(json_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()