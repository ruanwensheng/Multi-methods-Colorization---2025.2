"""
Download COCO 2017 dataset for colorization training and evaluation.

Usage:
    python tools/download_coco.py [--split train|val|both] [--benchmark-size 500]

Downloads to data/raw/coco2017/ and creates a shared benchmark subset.
"""

import os
import sys
import argparse
import zipfile
import random
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deep_learning.utils import load_config

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",      # ~18GB
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",          # ~1GB
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",        # ~6GB
}


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def create_benchmark_subset(source_dir, benchmark_dir, size=500, seed=42):
    """Create a fixed random subset for shared benchmarking."""
    if os.path.exists(benchmark_dir) and len(os.listdir(benchmark_dir)) >= size:
        print(f"Benchmark subset already exists ({len(os.listdir(benchmark_dir))} images)")
        return

    os.makedirs(benchmark_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(source_dir) if f.lower().endswith((".jpg", ".png"))])
    if len(images) == 0:
        print(f"No images found in {source_dir}")
        return

    random.seed(seed)
    subset = random.sample(images, min(size, len(images)))

    print(f"Creating benchmark subset: {len(subset)} images...")
    for img in tqdm(subset, desc="Copying"):
        shutil.copy2(os.path.join(source_dir, img), os.path.join(benchmark_dir, img))

    print(f"Benchmark subset created at {benchmark_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 dataset")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both",
                        help="Which split to download")
    parser.add_argument("--benchmark-size", type=int, default=500,
                        help="Number of images for shared benchmark subset")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config file path")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep zip files after extraction")
    args = parser.parse_args()

    cfg = load_config(args.config)
    coco_dir = cfg["paths"]["data_coco"]
    os.makedirs(coco_dir, exist_ok=True)

    # Determine which splits to download
    splits = []
    if args.split in ("train", "both"):
        splits.append("train2017")
    if args.split in ("val", "both"):
        splits.append("val2017")

    for split in splits:
        split_dir = os.path.join(coco_dir, split)

        # Skip if already downloaded
        if os.path.isdir(split_dir) and len(os.listdir(split_dir)) > 100:
            print(f"{split} already exists ({len(os.listdir(split_dir))} images), skipping")
            continue

        zip_path = os.path.join(coco_dir, f"{split}.zip")
        url = COCO_URLS[split]

        # Download
        if not os.path.exists(zip_path):
            print(f"Downloading {split}...")
            print(f"URL: {url}")
            download_file(url, zip_path)
        else:
            print(f"Zip already exists: {zip_path}")

        # Extract
        extract_zip(zip_path, coco_dir)

        # Clean up zip
        if not args.keep_zip and os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Removed {zip_path}")

    # Create benchmark subset from val2017
    val_dir = os.path.join(coco_dir, "val2017")
    benchmark_dir = os.path.join(coco_dir, "benchmark")
    if os.path.isdir(val_dir):
        create_benchmark_subset(val_dir, benchmark_dir, size=args.benchmark_size)

    print("\nDone! Dataset structure:")
    for d in ["train2017", "val2017", "benchmark"]:
        p = os.path.join(coco_dir, d)
        if os.path.isdir(p):
            n = len(os.listdir(p))
            print(f"  {d}/: {n} images")


if __name__ == "__main__":
    main()
