"""
Build the reference pool and oracle scribbles needed before evaluation.

Must be run once after download_coco.py has populated data/raw/coco2017/.

What this script does
---------------------
1. Reads the 500-image benchmark set already created by download_coco.py
   (data/raw/coco2017/benchmark/).
2. Selects 100 diverse reference images from val2017 that are NOT in the
   benchmark set and writes their paths to
   data/raw/coco2017/reference_pool/pool_paths.txt.
3. For every benchmark image computes a 64-bin luminance histogram and picks
   the reference pool image with the smallest L1 distance; saves the mapping
   as data/raw/coco2017/reference_pool/benchmark_to_ref.json.
4. For every benchmark image generates oracle scribbles: randomly sample 1 %
   of pixels (seed=42) from the ground-truth color image, paint those positions
   with the true BGR values and alpha=255, and save a BGRA PNG to
   data/raw/coco2017/benchmark/scribbles/{image_name}.png.

Usage
-----
    python tools/prepare_benchmark.py [--data-dir data/raw/coco2017]
"""

import os
import sys
import json
import argparse
import random

import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deep_learning.utils import load_config


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def luminance_histogram(bgr_image, bins=64):
    """64-bin histogram of the luminance (Y) channel, normalised to sum=1."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float64)
    total = hist.sum()
    return hist / total if total > 0 else hist


def l1_distance(h1, h2):
    return np.sum(np.abs(h1 - h2))


# ---------------------------------------------------------------------------
# Reference pool
# ---------------------------------------------------------------------------

def build_reference_pool(val_dir, benchmark_names, pool_dir, pool_size=100, seed=42):
    """Select `pool_size` val images not in the benchmark set."""
    os.makedirs(pool_dir, exist_ok=True)
    pool_paths_file = os.path.join(pool_dir, "pool_paths.txt")

    if os.path.exists(pool_paths_file):
        with open(pool_paths_file) as f:
            paths = [l.strip() for l in f if l.strip()]
        if len(paths) >= pool_size:
            print(f"Reference pool already exists ({len(paths)} images), skipping.")
            return paths

    benchmark_set = set(benchmark_names)
    candidates = sorted([
        f for f in os.listdir(val_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and f not in benchmark_set
    ])

    if len(candidates) < pool_size:
        print(f"WARNING: only {len(candidates)} non-benchmark val images available.")
        pool_size = len(candidates)

    random.seed(seed)
    selected = random.sample(candidates, pool_size)
    pool_paths = [os.path.join(val_dir, f) for f in selected]

    with open(pool_paths_file, "w") as f:
        for p in pool_paths:
            f.write(p + "\n")

    print(f"Reference pool: {len(pool_paths)} images → {pool_paths_file}")
    return pool_paths


# ---------------------------------------------------------------------------
# Histogram matching
# ---------------------------------------------------------------------------

def build_benchmark_to_ref(benchmark_dir, pool_paths, pool_dir):
    """Match each benchmark image to its closest reference by luminance histogram."""
    mapping_file = os.path.join(pool_dir, "benchmark_to_ref.json")

    benchmark_images = sorted([
        f for f in os.listdir(benchmark_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.isdir(os.path.join(benchmark_dir, f))
    ])

    # Pre-compute pool histograms
    print("Computing reference pool histograms...")
    pool_hists = []
    valid_pool = []
    for p in tqdm(pool_paths, desc="Pool histograms"):
        img = cv2.imread(p)
        if img is None:
            continue
        pool_hists.append(luminance_histogram(img))
        valid_pool.append(p)
    pool_hists = np.array(pool_hists)  # (N_pool, 64)

    # Match each benchmark image
    print("Matching benchmark images to reference pool...")
    mapping = {}
    for bname in tqdm(benchmark_images, desc="Matching"):
        bpath = os.path.join(benchmark_dir, bname)
        img = cv2.imread(bpath)
        if img is None:
            continue
        bhist = luminance_histogram(img)
        dists = np.sum(np.abs(pool_hists - bhist), axis=1)
        best_idx = int(np.argmin(dists))
        mapping[bname] = valid_pool[best_idx]

    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Benchmark→reference mapping: {len(mapping)} entries → {mapping_file}")
    return mapping


# ---------------------------------------------------------------------------
# Oracle scribbles
# ---------------------------------------------------------------------------

def generate_oracle_scribbles(benchmark_dir, scribble_dir, sample_rate=0.01, seed=42):
    """Generate BGRA oracle scribble overlays for the scribble-based method.

    For each benchmark color image, randomly samples ~1 % of pixels (seed=42),
    marks them with their true BGR colour and alpha=255.  The rest is transparent.
    """
    os.makedirs(scribble_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(benchmark_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.isdir(os.path.join(benchmark_dir, f))
    ])

    rng = np.random.default_rng(seed)

    for fname in tqdm(images, desc="Oracle scribbles"):
        out_path = os.path.join(scribble_dir, os.path.splitext(fname)[0] + ".png")
        if os.path.exists(out_path):
            continue

        img = cv2.imread(os.path.join(benchmark_dir, fname))
        if img is None:
            continue

        h, w = img.shape[:2]
        n_pixels = h * w
        n_samples = max(1, int(n_pixels * sample_rate))

        # Sample pixel indices without replacement
        flat_idx = rng.choice(n_pixels, size=n_samples, replace=False)
        rows, cols = np.unravel_index(flat_idx, (h, w))

        # BGRA overlay — transparent by default
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[rows, cols, :3] = img[rows, cols]   # true BGR
        overlay[rows, cols, 3] = 255                 # alpha = scribble

        cv2.imwrite(out_path, overlay)

    print(f"Oracle scribbles saved to {scribble_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark reference pool and oracle scribbles")
    parser.add_argument("--data-dir", default=None, help="Path to data/raw/coco2017 directory")
    parser.add_argument("--pool-size", type=int, default=100, help="Reference pool size")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    coco_dir = args.data_dir or cfg["paths"]["data_coco"]

    val_dir = os.path.join(coco_dir, "val2017")
    benchmark_dir = os.path.join(coco_dir, "benchmark")
    pool_dir = os.path.join(coco_dir, "reference_pool")
    scribble_dir = os.path.join(benchmark_dir, "scribbles")

    for d, name in [(val_dir, "val2017"), (benchmark_dir, "benchmark")]:
        if not os.path.isdir(d):
            print(f"ERROR: {name} directory not found: {d}")
            print("Run 'python tools/download_coco.py' first.")
            sys.exit(1)

    benchmark_names = [
        f for f in os.listdir(benchmark_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not os.path.isdir(os.path.join(benchmark_dir, f))
    ]
    print(f"Benchmark images found: {len(benchmark_names)}")

    # 1. Reference pool
    pool_paths = build_reference_pool(
        val_dir, benchmark_names, pool_dir,
        pool_size=args.pool_size, seed=args.seed
    )

    # 2. Histogram matching
    build_benchmark_to_ref(benchmark_dir, pool_paths, pool_dir)

    # 3. Oracle scribbles
    generate_oracle_scribbles(benchmark_dir, scribble_dir, seed=args.seed)

    print("\nDone. Files created:")
    print(f"  {pool_dir}/pool_paths.txt")
    print(f"  {pool_dir}/benchmark_to_ref.json")
    print(f"  {scribble_dir}/*.png")


if __name__ == "__main__":
    main()
