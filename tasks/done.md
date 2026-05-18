# Work Summary — Example-based Colorization (Welsh 2002)

## 1. Core Implementation (`src/example_based/`)

| File | What it does |
|---|---|
| `utils.py` | `bgr_to_lab`, `lab_to_bgr`, `local_std` (sliding-window std via `uniform_filter`) |
| `matching.py` | `build_feature_vectors` (per-pixel `[L, std]`), `kd_tree_match` (KDTree + L-refinement) |
| `colorizer.py` | `ExampleColorizer` — reads config, dowsamples reference, runs matching, returns `(BGR, info_dict)` |
| `__init__.py` | Exports `ExampleColorizer` |

**Data flow:**
```
gray_image ──► L_target                reference_image ──► downsample 0.5 ──► L_ref, ab_ref
                 │                                                                    │
                 ▼                                                                    ▼
        build_feature_vectors                                             build_feature_vectors
         [L, local_std(L)]                                                [L, local_std(L)]
                 │                                                                    │
                 └──────────────────── kd_tree_match ───────────────────────────────┘
                                     k=5 candidates → pick min|ΔL|
                                              │
                                              ▼
                                    ab_transferred (H×W, 2)
                                              │
                                     L_target + ab_transferred
                                              │
                                        colorized_bgr
```

**Unified API `info_dict` keys:** `method`, `time_seconds`, `image_size`
(matches `DeepColorizer` which already uses the same three keys)

---

## 2. Tests (`tests/test_example.py`)

21 tests across 4 classes:

| Class | Tests |
|---|---|
| `TestExampleColorizerInit` | creates without cfg, default params, cfg overrides |
| `TestExampleColorizerColorize` | shapes, dtype, resolution preservation, valid range, info_dict keys |
| `TestBuildFeatureVectors` | output shape `(H*W, 2)`, L column, std non-negative, uniform→zero std |
| `TestKdTreeMatch` | output shape, ab range, exact match, k=1 equals nearest-neighbor |

---

## 3. Evaluation Tools

### `tools/prepare_benchmark.py`
1. Reads 500-image benchmark set from `benchmark/`
2. Selects 100 diverse reference images from val2017 (excluding benchmark) → `reference_pool/pool_paths.txt`
3. Matches each benchmark image to closest reference by 64-bin luminance histogram L1 distance → `reference_pool/benchmark_to_ref.json`
4. Generates oracle scribbles (1% pixels, seed=42) → `benchmark/scribbles/*.png`

### `tools/evaluate_example.py`
- Reads `benchmark_to_ref.json` to pair each test image with its reference
- Runs `ExampleColorizer.colorize(gray, reference)` on all 500 images
- Computes PSNR, SSIM, LPIPS per image
- Outputs `results/example_based/metrics/<tag>/aggregate_metrics.json` and `per_image_metrics.csv`
- Saves side-by-side visualizations for first 20 images
- CLI: `python tools/evaluate_example.py --tag pool-match`

---

## 4. Config (`configs/config.yaml`)

Added under `paths:`:
```yaml
results_example: "results/example_based"
results_scribble: "results/scribble"
```

Added section:
```yaml
example_based:
  neighborhood_size: 5
  k_neighbors: 5
  downsample: 0.5
```

---

## 5. Docker Fixes (`docker/Dockerfile`)

| Problem | Fix |
|---|---|
| `libgl1-mesa-glx` removed in Debian Trixie | Replaced with `libgl1` |
| `python:3.10-slim` — packages require Python ≥ 3.11 | Bumped base image to `python:3.11-slim` |
| `pytest` not installed in image | Added `pytest==8.3.5` to `requirements.txt` |

---

## 6. How to Run Tests

### Step 1 — Build Docker image
```powershell
docker compose -f docker/docker-compose.yml build
```

### Step 2 — Start container
```powershell
docker compose -f docker/docker-compose.yml run --rm dev bash
```

### Step 3 — Unit tests (no data needed)
```bash
pytest tests/test_example.py -v
```

### Step 4 — Download data (inside container)
```bash
python tools/download_coco.py --split val
# Then exit container and commit configs/benchmark_filelist.txt
```

### Step 5 — Build reference pool
```bash
python tools/prepare_benchmark.py
```

### Step 6 — Run full evaluation
```bash
python tools/evaluate_example.py --tag pool-match
```

Results: `results/example_based/metrics/pool-match/`
