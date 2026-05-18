"""Quick dataset integrity check.

Verifies:
1. Image counts in each split.
2. Sample images open as valid RGB.
3. Disk usage per split.
4. benchmark/ is a strict subset of test2017/ (the leakage-free holdout — see SPEC §6.3)
   and is DISJOINT from val2017/ (which is used by Trainer for best-ckpt selection).
"""
import os
from PIL import Image

ROOT = "data/raw/coco2017"
SPLITS = ["train2017", "val2017", "test2017", "benchmark", "annotations"]

print("=== Sample image / file validity check ===\n")
for name in SPLITS:
    p = os.path.join(ROOT, name)
    if not os.path.isdir(p):
        print(f"{name}: MISSING")
        continue
    files = sorted(os.listdir(p))
    print(f"{name}: {len(files)} files")
    for f in files[:3]:
        fp = os.path.join(p, f)
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(fp)
            print(f"  {f}: size={img.size}, mode={img.mode}")
        else:
            print(f"  {f}: {os.path.getsize(fp)/(1024**2):.1f} MB")
    print()

print("=== Disk size (sum of sizes) ===")
total = 0
for name in SPLITS:
    p = os.path.join(ROOT, name)
    if not os.path.isdir(p):
        continue
    s = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)
            if os.path.isfile(os.path.join(p, f)))
    print(f"  {name:<14} {s/(1024**3):.2f} GB")
    total += s
print(f"  {'TOTAL':<14} {total/(1024**3):.2f} GB")

print("\n=== Leakage check (SPEC §6.3) ===")
bench_dir = os.path.join(ROOT, "benchmark")
val_dir   = os.path.join(ROOT, "val2017")
test_dir  = os.path.join(ROOT, "test2017")

if os.path.isdir(bench_dir) and os.path.isdir(test_dir):
    bench = set(os.listdir(bench_dir))
    test  = set(os.listdir(test_dir))
    val   = set(os.listdir(val_dir)) if os.path.isdir(val_dir) else set()

    bench_in_test = bench & test
    bench_in_val  = bench & val
    print(f"  benchmark ∩ test2017: {len(bench_in_test)}/{len(bench)}  "
          f"({'OK — subset of held-out test set' if bench_in_test == bench else 'FAIL'})")
    print(f"  benchmark ∩ val2017:  {len(bench_in_val)}/{len(bench)}  "
          f"({'OK — no overlap with validation set' if not bench_in_val else 'LEAK — overlap detected!'})")

    if bench != bench_in_test:
        missing = bench - test
        print(f"  WARNING: {len(missing)} benchmark images not in test2017 — first 3: {list(missing)[:3]}")


# --- Diversity check (SPEC §6.3 stratified sampling) ---
print("\n=== Diversity check: benchmark vs full test2017 (brightness × saturation) ===")
import numpy as np
features_cache = os.path.join(ROOT, "test2017_features.npz")
if os.path.exists(features_cache):
    data = np.load(features_cache, allow_pickle=False)
    all_files = data["files"].tolist()
    all_feats = data["features"]
    bench_set = set(os.listdir(os.path.join(ROOT, "benchmark"))) if os.path.isdir(os.path.join(ROOT, "benchmark")) else set()
    bench_mask = np.array([f in bench_set for f in all_files])

    def _summ(label, mask):
        feats = all_feats[mask]
        if len(feats) == 0:
            print(f"  {label}: (empty)")
            return
        b = feats[:, 0]; s = feats[:, 1]
        print(f"  {label} (N={len(feats)}):")
        print(f"    brightness  mean={b.mean():6.1f}  std={b.std():5.1f}  "
              f"[p10={np.quantile(b,0.1):.1f}, p90={np.quantile(b,0.9):.1f}]")
        print(f"    saturation  mean={s.mean():6.1f}  std={s.std():5.1f}  "
              f"[p10={np.quantile(s,0.1):.1f}, p90={np.quantile(s,0.9):.1f}]")

    _summ("Full test2017", np.ones(len(all_files), dtype=bool))
    _summ("Benchmark    ", bench_mask)

    # KS test (cheap, no scipy): just compare quantile ladders
    def _quantile_max_gap(a, b, n=20):
        qs = np.linspace(0, 1, n + 1)[1:-1]
        return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).max())
    gap_b = _quantile_max_gap(all_feats[:, 0], all_feats[bench_mask, 0])
    gap_s = _quantile_max_gap(all_feats[:, 1], all_feats[bench_mask, 1])
    print(f"  Max quantile gap (population vs benchmark):")
    print(f"    brightness: {gap_b:.2f}   (lower = more representative)")
    print(f"    saturation: {gap_s:.2f}")
else:
    print("  (no features cache yet — run download_coco.py first)")
