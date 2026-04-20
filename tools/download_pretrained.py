"""
Download pre-trained model weights for colorization.

Usage:
    python tools/download_pretrained.py [--model zhang16|deoldify|ddcolor|all]

Downloads weights to models/pretrained/.
"""

import os
import sys
import argparse

import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.deep_learning.utils import load_config

# Pre-trained weight URLs
PRETRAINED_URLS = {
    "zhang16_eccv": {
        "url": "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
        "filename": "zhang16_eccv.pth",
        "description": "Zhang et al. 2016 ECCV - Colorful Image Colorization",
    },
    "zhang16_siggraph": {
        "url": "https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth",
        "filename": "zhang16_siggraph.pth",
        "description": "Zhang et al. 2017 SIGGRAPH - Interactive Colorization",
    },
}


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def download_model(key, save_dir):
    """Download a specific pre-trained model."""
    info = PRETRAINED_URLS[key]
    dest = os.path.join(save_dir, info["filename"])

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  {info['filename']} already exists ({size_mb:.1f} MB), skipping")
        return True

    print(f"  Downloading: {info['description']}")
    print(f"  URL: {info['url']}")
    try:
        download_file(info["url"], dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Saved: {dest} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained colorization weights")
    parser.add_argument("--model", choices=["zhang16", "all"], default="zhang16",
                        help="Which model weights to download")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    save_dir = cfg["paths"]["models_pretrained"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving weights to: {save_dir}\n")

    if args.model in ("zhang16", "all"):
        print("[Zhang et al. 2016 ECCV]")
        download_model("zhang16_eccv", save_dir)
        print()
        print("[Zhang et al. 2017 SIGGRAPH]")
        download_model("zhang16_siggraph", save_dir)
        print()

    print("Done!")
    print(f"\nAvailable weights in {save_dir}:")
    if os.path.isdir(save_dir):
        for f in sorted(os.listdir(save_dir)):
            size_mb = os.path.getsize(os.path.join(save_dir, f)) / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
