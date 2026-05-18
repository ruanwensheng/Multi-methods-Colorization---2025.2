"""
Download COCO 2017 dataset for colorization training and evaluation.

Primary source: Kaggle dataset `awsaf49/coco-2017-dataset` (via kagglehub, ~27 GB).
Fallback: direct download from images.cocodataset.org (only if --source http is passed).

Idempotency:
    - If `data/raw/coco2017/train2017/` and `val2017/` already contain images,
      no download is attempted.
    - If kagglehub already cached the dataset, no re-download.
    - The 500-image benchmark subset is only rebuilt if missing.

Usage:
    python tools/download_coco.py                       # default: kaggle source
    python tools/download_coco.py --source http         # fallback to coco mirror
    python tools/download_coco.py --benchmark-size 500
    python tools/download_coco.py --link              # symlink instead of copy (saves 27GB)
"""

import os
import sys
import argparse
import random
import shutil
import zipfile
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(path="configs/config.yaml"):
    """Minimal config loader — kept here so this script can run without torch installed."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- helpers ----------

def _image_count(directory):
    """Count image files in a directory (non-recursive)."""
    if not os.path.isdir(directory):
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(
        1 for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    )


def _is_split_ready(split_dir, min_count=100):
    """A split is 'ready' if it already has many images."""
    return _image_count(split_dir) >= min_count


def _find_split_in_tree(root, split_name):
    """Walk `root` looking for a directory named exactly `split_name` (e.g. 'train2017')
    that contains image files. Returns the first match or None.
    """
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == split_name and _image_count(dirpath) > 0:
            return dirpath
    return None


def _materialize_split(src_dir, dst_dir, mode="copy"):
    """Move or symlink the contents of src_dir into dst_dir.

    mode:
        - "copy": copy files (safe, uses 27GB extra)
        - "move": move files (fastest, source becomes empty)
        - "link": create a directory junction / symlink (Windows: junction; saves space)
    """
    os.makedirs(os.path.dirname(dst_dir) or ".", exist_ok=True)

    if mode == "link":
        # Remove empty dst dir if present, then link
        if os.path.exists(dst_dir):
            if os.path.islink(dst_dir) or (os.name == "nt" and os.path.isdir(dst_dir)
                                            and _image_count(dst_dir) == 0):
                try:
                    os.rmdir(dst_dir)
                except OSError:
                    pass
        if os.name == "nt":
            # Use directory junction on Windows (doesn't need admin)
            import subprocess
            subprocess.check_call(["cmd", "/c", "mklink", "/J", dst_dir, src_dir],
                                  stdout=subprocess.DEVNULL)
        else:
            os.symlink(src_dir, dst_dir, target_is_directory=True)
        print(f"  Linked: {dst_dir} -> {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir)
             if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    action = shutil.move if mode == "move" else shutil.copy2
    verb = "Moving" if mode == "move" else "Copying"
    for fname in tqdm(files, desc=f"{verb} {os.path.basename(dst_dir)}"):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            action(src, dst)


def _compute_brightness_saturation_features(source_dir, cache_path):
    """For each image in source_dir, compute (brightness, saturation) from a 32×32 thumbnail.

    Caches the result to `cache_path` (a .npz) so re-runs are instant.

    Returns:
        files (list[str]):    image filenames, sorted
        features (np.ndarray): shape (N, 2) — column 0 = brightness in [0,255],
                              column 1 = mean per-pixel (max-min) saturation in [0,255]
    """
    import numpy as np
    from PIL import Image

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        cached_files = data["files"].tolist()
        cached_feats = data["features"]
        # Sanity: only reuse cache if file list matches the current dir
        current = sorted([f for f in os.listdir(source_dir)
                          if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}])
        if cached_files == current:
            print(f"  Loaded cached features from {cache_path} (N={len(cached_files)})")
            return cached_files, cached_feats
        else:
            print(f"  Cache stale (dir contents changed); recomputing features")

    files = sorted([f for f in os.listdir(source_dir)
                    if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}])
    feats = np.zeros((len(files), 2), dtype=np.float32)

    print(f"  Computing brightness × saturation features for {len(files):,} images "
          f"(one-time, ~1-3 min)...")
    for i, fname in enumerate(tqdm(files, desc="Features")):
        try:
            with Image.open(os.path.join(source_dir, fname)) as img:
                thumb = img.convert("RGB").resize((32, 32), Image.BILINEAR)
                arr = np.asarray(thumb, dtype=np.float32)  # (32,32,3)
        except Exception:
            feats[i] = (0.0, 0.0)
            continue
        feats[i, 0] = float(arr.mean())                                       # brightness
        feats[i, 1] = float((arr.max(axis=2) - arr.min(axis=2)).mean())       # saturation

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, files=np.array(files), features=feats)
    print(f"  Cached features to {cache_path}")
    return files, feats


def create_benchmark_subset(source_dir, benchmark_dir, size=1000, seed=42, strategy="stratified"):
    """Build the shared evaluation benchmark.

    Args:
        source_dir: where to sample from (per SPEC §6.3 this is test2017/).
        benchmark_dir: destination directory.
        size: target sample size (SPEC: 1000).
        seed: RNG seed for reproducibility.
        strategy: 'stratified' (default — by brightness × saturation, 4×4=16 cells)
                  or 'random' (pure random sampling, equivalent to numpy.random.choice).

    Stratification rationale: see SPEC §6.3. Quantile-binning each feature into 4 bins
    yields 16 cells of equal population in test2017; proportional allocation gives ~62
    images per cell × 16 = ~1000. This guarantees coverage of the brightness×saturation
    space, which matters for colorization since per-image PSNR/SSIM/LPIPS varies strongly
    with these properties.
    """
    import numpy as np

    if _is_split_ready(benchmark_dir, min_count=size):
        print(f"Benchmark subset already exists ({_image_count(benchmark_dir)} images), skipping")
        return

    os.makedirs(benchmark_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(source_dir)
                     if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print(f"  No images found in {source_dir}, skipping benchmark creation")
        return

    if strategy == "random":
        print(f"Creating benchmark subset (RANDOM, seed={seed}): {size} of {len(images):,} images")
        random.seed(seed)
        subset = random.sample(images, min(size, len(images)))

    elif strategy == "stratified":
        print(f"Creating benchmark subset (STRATIFIED by brightness × saturation, "
              f"seed={seed}): target {size} of {len(images):,}")
        # Feature cache lives one level up from source_dir so it persists across runs
        cache_path = os.path.join(os.path.dirname(source_dir.rstrip("/\\")),
                                  "test2017_features.npz")
        files, feats = _compute_brightness_saturation_features(source_dir, cache_path)

        # Quantile-bin each feature into N_BINS bins → equal-population strata
        N_BINS = 4
        rng = np.random.default_rng(seed)

        bright_edges = np.quantile(feats[:, 0], np.linspace(0, 1, N_BINS + 1))
        sat_edges    = np.quantile(feats[:, 1], np.linspace(0, 1, N_BINS + 1))
        # np.digitize: edges[1:-1] gives N_BINS bins (indices 0..N_BINS-1)
        b_idx = np.digitize(feats[:, 0], bright_edges[1:-1])
        s_idx = np.digitize(feats[:, 1], sat_edges[1:-1])
        cell_idx = b_idx * N_BINS + s_idx                                      # 0..15
        n_cells = N_BINS * N_BINS

        # Proportional allocation per cell
        selected = []
        cell_report = []
        for c in range(n_cells):
            mask = cell_idx == c
            cell_files = np.array(files)[mask]
            if len(cell_files) == 0:
                cell_report.append((c, 0, 0))
                continue
            proportion = len(cell_files) / len(files)
            n_take = int(round(proportion * size))
            n_take = min(n_take, len(cell_files))
            chosen = rng.choice(cell_files, size=n_take, replace=False)
            selected.extend(chosen.tolist())
            cell_report.append((c, len(cell_files), n_take))

        # Truncate or top-up to exactly `size`
        if len(selected) > size:
            selected = rng.choice(selected, size=size, replace=False).tolist()
        elif len(selected) < size:
            remaining = list(set(files) - set(selected))
            extra = rng.choice(remaining, size=size - len(selected), replace=False)
            selected.extend(extra.tolist())

        subset = sorted(selected)

        # Report cell coverage
        print(f"  Cell coverage (cell_id : pop_in_test → taken_in_benchmark):")
        nonempty = [r for r in cell_report if r[1] > 0]
        for c, pop, taken in nonempty:
            print(f"    cell {c:>2d} : {pop:>6,} → {taken:>4d}")
        print(f"  Non-empty cells: {len(nonempty)}/{n_cells}")

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    print(f"\nCopying {len(subset)} images to {benchmark_dir} ...")
    for img in tqdm(subset, desc="Copy"):
        src = os.path.join(source_dir, img)
        dst = os.path.join(benchmark_dir, img)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"Benchmark subset created at {benchmark_dir}")


# ---------- sources ----------

def download_from_kaggle(coco_dir, splits, mode="copy"):
    """Download COCO 2017 from Kaggle via kagglehub.

    Dataset: awsaf49/coco-2017-dataset (~27 GB compressed)
    """
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run:")
        print("    conda run -n AI pip install kagglehub")
        sys.exit(1)

    print("Downloading COCO 2017 from Kaggle (awsaf49/coco-2017-dataset)...")
    print("  This is ~27 GB and may take a while. kagglehub caches across runs.")
    cache_root = kagglehub.dataset_download("awsaf49/coco-2017-dataset")
    print(f"  Kaggle cache root: {cache_root}")

    for split in splits:
        dst = os.path.join(coco_dir, split)
        if _is_split_ready(dst):
            print(f"  {split}: already populated ({_image_count(dst)} images), skipping")
            continue

        src = _find_split_in_tree(cache_root, split)
        if src is None:
            print(f"  WARNING: {split} not found inside {cache_root}")
            continue
        print(f"  {split}: source = {src}")
        _materialize_split(src, dst, mode=mode)


def _download_with_resume(url, dest_path, max_retries=5, timeout=120):
    """Stream-download `url` to `dest_path` with HTTP Range-header resume + retries.

    If `dest_path` already exists, resumes from its current size.
    Retries network errors with exponential backoff.
    """
    import requests
    from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout

    # Probe total size
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    total_size = int(head.headers.get("content-length", 0))
    if total_size == 0:
        # Some mirrors don't return Content-Length on HEAD; fall back to GET
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

    print(f"  Remote size: {total_size / (1024**3):.2f} GB")

    for attempt in range(1, max_retries + 1):
        resume_from = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        if resume_from >= total_size > 0:
            print(f"  Already complete ({resume_from / (1024**3):.2f} GB)")
            return

        headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}
        if resume_from > 0:
            print(f"  Resuming from {resume_from / (1024**3):.2f} GB ({100*resume_from/total_size:.1f}%)")

        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                # 206 = partial content (resume worked); 200 = full content
                if resume_from > 0 and r.status_code == 200:
                    print("  Server ignored Range header; restarting from 0")
                    resume_from = 0
                    open(dest_path, "wb").close()
                r.raise_for_status()

                mode = "ab" if resume_from > 0 else "wb"
                with open(dest_path, mode) as f, tqdm(
                    total=total_size, initial=resume_from,
                    unit="B", unit_scale=True, desc=os.path.basename(dest_path)
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            # If we get here, the body finished without exception
            final_size = os.path.getsize(dest_path)
            if total_size > 0 and final_size < total_size:
                raise RuntimeError(f"Truncated download: {final_size} < {total_size}")
            return
        except (ChunkedEncodingError, ConnectionError, ReadTimeout, RuntimeError) as e:
            wait = min(2 ** attempt, 30)
            print(f"  Attempt {attempt}/{max_retries} failed: {e}. Retry in {wait}s...")
            import time as _time
            _time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def download_from_http(coco_dir, splits):
    """Download COCO 2017 zips directly from images.cocodataset.org with Range resume."""
    urls = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",   # ~18 GB
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",       # ~1 GB
    }
    for split in splits:
        dst = os.path.join(coco_dir, split)
        if _is_split_ready(dst):
            print(f"{split}: already populated ({_image_count(dst)} images), skipping")
            continue
        if split not in urls:
            print(f"  No HTTP URL for {split}, skipping")
            continue

        zip_path = os.path.join(coco_dir, f"{split}.zip")
        print(f"\n[{split}] Downloading from {urls[split]}")
        _download_with_resume(urls[split], zip_path)

        print(f"  Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            members = z.namelist()
            for m in tqdm(members, desc=f"Extract {split}"):
                z.extract(m, coco_dir)
        os.remove(zip_path)
        print(f"  Removed {zip_path}")


# ---------- main ----------

def extract_local_zip(coco_dir, splits, zip_path=None):
    """Extract an already-downloaded Kaggle archive.zip into coco_dir.

    Strips the leading 'coco2017/' path component so files land directly in
    `coco_dir/{train2017,val2017,test2017,annotations}/`.

    Idempotent: files that already exist on disk are skipped.
    """
    if zip_path is None:
        # Default: data/raw/coco2017/archive.zip
        zip_path = os.path.join(coco_dir, "archive.zip")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Archive not found at {zip_path}. "
            f"Download from Kaggle first: kaggle datasets download "
            f"awsaf49/coco-2017-dataset -p {coco_dir}"
        )

    print(f"Extracting {zip_path} ({os.path.getsize(zip_path)/(1024**3):.2f} GB) ...")

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()

        # Filter: keep only the splits we want + annotations.
        wanted_prefixes = tuple(f"coco2017/{s}/" for s in splits)
        # Also keep annotations (small, often useful)
        wanted_prefixes = wanted_prefixes + ("coco2017/annotations/",)

        to_extract = [m for m in members if m.replace("\\", "/").startswith(wanted_prefixes)]
        skipped_total = len(members) - len(to_extract)
        print(f"  Selected {len(to_extract):,} entries (skipping {skipped_total:,} outside {splits} + annotations)")

        extracted = 0
        skipped_existing = 0
        for m in tqdm(to_extract, desc="Extract"):
            # Strip leading 'coco2017/' so files land directly under coco_dir
            rel = m.replace("\\", "/")
            if rel.startswith("coco2017/"):
                rel = rel[len("coco2017/"):]
            if not rel or rel.endswith("/"):
                # Directory entry — make sure it exists
                os.makedirs(os.path.join(coco_dir, rel), exist_ok=True)
                continue

            dst = os.path.join(coco_dir, rel)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                skipped_existing += 1
                continue

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(m) as src_f, open(dst, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)
            extracted += 1

    print(f"  Extracted {extracted:,} files, skipped {skipped_existing:,} already present")


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 dataset")
    parser.add_argument("--source", choices=["kaggle", "http", "local-zip"], default="kaggle",
                        help="Download source. "
                             "'kaggle' = kagglehub.dataset_download (~27 GB). "
                             "'http' = direct from cocodataset.org with Range resume. "
                             "'local-zip' = extract a pre-downloaded archive.zip in data_coco/.")
    parser.add_argument("--split", choices=["train", "val", "test", "both", "all"], default="both",
                        help="Which splits to download. "
                             "'both' = train+val (default, what SPEC uses). "
                             "'all' = train+val+test (test2017 is NOT used by the evaluation pipeline; "
                             "extract only if you want it on disk).")
    parser.add_argument("--benchmark-size", type=int, default=1000,
                        help="Number of images for the shared benchmark subset (sampled from test2017, seed=42)")
    parser.add_argument("--benchmark-strategy", choices=["stratified", "random"], default="stratified",
                        help="Sampling strategy. 'stratified' (default) = quantile-bin test2017 by "
                             "brightness × saturation (4×4 cells) and sample proportionally — guarantees "
                             "coverage of the color-difficulty space. 'random' = pure random.draw.")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--mode", choices=["copy", "move", "link"], default="copy",
                        help="How to materialize Kaggle files into data/raw/coco2017/. "
                             "'link' uses a directory junction on Windows (saves 27 GB).")
    parser.add_argument("--link", action="store_true",
                        help="Shortcut for --mode link")
    args = parser.parse_args()

    if args.link:
        args.mode = "link"

    cfg = load_config(args.config)
    coco_dir = cfg["paths"]["data_coco"]
    os.makedirs(coco_dir, exist_ok=True)

    splits = []
    if args.split in ("train", "both", "all"):
        splits.append("train2017")
    if args.split in ("val", "both", "all"):
        splits.append("val2017")
    if args.split in ("test", "all"):
        splits.append("test2017")

    print(f"Target directory: {os.path.abspath(coco_dir)}")
    print(f"Splits: {splits}")
    print(f"Source: {args.source}, Mode: {args.mode}\n")

    if args.source == "kaggle":
        download_from_kaggle(coco_dir, splits, mode=args.mode)
    elif args.source == "http":
        download_from_http(coco_dir, splits)
    elif args.source == "local-zip":
        extract_local_zip(coco_dir, splits)

    # Benchmark subset built from test2017 (held-out from training & best-ckpt selection).
    # See SPEC.md §6.3 Data Splits & Leakage Policy.
    test_dir = os.path.join(coco_dir, "test2017")
    benchmark_dir = os.path.join(coco_dir, "benchmark")
    if _image_count(test_dir) > 0:
        create_benchmark_subset(test_dir, benchmark_dir,
                                size=args.benchmark_size,
                                strategy=args.benchmark_strategy)
    else:
        print("\nNote: test2017 is empty, skipping benchmark subset creation. "
              "Run with --split all (or --split test) first to extract test2017.")

    # Summary
    print("\n" + "=" * 50)
    print("Dataset structure:")
    print("=" * 50)
    for d in ["train2017", "val2017", "test2017", "benchmark"]:
        p = os.path.join(coco_dir, d)
        n = _image_count(p)
        marker = "OK" if n > 0 else "EMPTY"
        link_tag = " (junction)" if os.path.islink(p) else ""
        print(f"  [{marker:5s}] {d}/: {n} images{link_tag}")
    print("\nDone.")


if __name__ == "__main__":
    main()
