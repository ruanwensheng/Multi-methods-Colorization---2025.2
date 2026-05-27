"""Tests for tools/compare_methods.py.

The script's docstring advertises a `--methods zhang16,deoldify` filter so users
can run a subset of the comparison (e.g., skip the ControlNet slot documented as
a reproducibility gap in T3.5 — its substitute pipeline produces ~6 dB PSNR and
including it would poison the cross-model summary). The filtering helper lives
on the tool module so it can be unit tested without launching torch/HF.
"""

import os
import sys
import importlib.util


def _load_compare_methods_module():
    """Load tools/compare_methods.py as a module without executing main()."""
    here = os.path.dirname(__file__)
    path = os.path.normpath(os.path.join(here, "..", "tools", "compare_methods.py"))
    spec = importlib.util.spec_from_file_location("compare_methods_tool", path)
    module = importlib.util.module_from_spec(spec)
    # Stub torch-heavy imports the module pulls in transitively. We only want
    # the filter helper, not the model loaders.
    spec.loader.exec_module(module)
    return module


def test_filter_models_keeps_only_named_slugs():
    """`filter_models` keeps entries whose name matches any of the slugs."""
    cm = _load_compare_methods_module()
    models = {
        "Zhang16 (Ours)": object(),
        "Zhang 2017 (Interactive CNN)": object(),
        "DeOldify (GAN)": object(),
        "ControlNet (Diffusion)": object(),
    }
    kept = cm.filter_models(models, ["zhang16", "deoldify"])
    assert set(kept) == {"Zhang16 (Ours)", "DeOldify (GAN)"}


def test_filter_models_passthrough_when_methods_is_none():
    """No filter -> return the dict unchanged (preserve insertion order)."""
    cm = _load_compare_methods_module()
    models = {"A": 1, "B": 2}
    assert cm.filter_models(models, None) is models


def test_filter_models_empty_list_yields_empty_dict():
    """`--methods ''` (parsed to []) should drop everything, not pass through."""
    cm = _load_compare_methods_module()
    models = {"Zhang16 (Ours)": object()}
    assert cm.filter_models(models, []) == {}


def test_filter_models_zhang17_alias_matches_zhang_2017():
    """Regression guard: shorthand `zhang17` must resolve to "Zhang 2017 (...)".

    Naive substring matching misses this because of the space inside
    "zhang 2017"; _SLUG_ALIASES exists to handle exactly this case.
    """
    cm = _load_compare_methods_module()
    models = {
        "Zhang16 (Ours)": object(),
        "Zhang 2017 (Interactive CNN)": object(),
        "DeOldify (GAN)": object(),
    }
    kept = cm.filter_models(models, ["zhang17"])
    assert set(kept) == {"Zhang 2017 (Interactive CNN)"}


def test_filter_models_unknown_slug_raises():
    """Typo guards: unknown slugs should fail loud, not silently no-op."""
    cm = _load_compare_methods_module()
    models = {"Zhang16 (Ours)": object(), "DeOldify (GAN)": object()}
    try:
        cm.filter_models(models, ["zhang17"])
    except ValueError as e:
        assert "zhang17" in str(e)
    else:
        raise AssertionError("filter_models should raise on unknown slug")


def test_cli_help_lists_methods_flag():
    """`tools/compare_methods.py --help` should advertise --methods (docstring promise)."""
    import subprocess
    here = os.path.dirname(__file__)
    script = os.path.normpath(os.path.join(here, "..", "tools", "compare_methods.py"))
    result = subprocess.run(
        [sys.executable, script, "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "--methods" in result.stdout


# ---------------------------------------------------------------------------
# Phase 4 (T4.1): aggregate the cross-model comparison from the canonical
# per-model evaluate_deep.py outputs instead of re-running the models. This
# keeps a single source of truth — the full 1,000-image test2017 benchmark —
# so the comparison table matches the Phase-3 numbers exactly (see
# docs/benchmark_methodology.md and tasks/todo.md).
# ---------------------------------------------------------------------------

import json
import csv as _csv


def _write_model_metrics(metrics_root, tag, psnr_mean, n_rows):
    """Create a fake per-model metrics dir mirroring evaluate_deep.py output."""
    d = os.path.join(metrics_root, tag)
    os.makedirs(d, exist_ok=True)
    agg = {
        "num_images": n_rows,
        "psnr_mean": psnr_mean, "psnr_std": 1.0,
        "psnr_ci_lo": psnr_mean - 0.2, "psnr_ci_hi": psnr_mean + 0.2,
        "ssim_mean": 0.9, "ssim_std": 0.05,
        "ssim_ci_lo": 0.88, "ssim_ci_hi": 0.92,
        "lpips_mean": 0.2, "lpips_std": 0.08,
        "lpips_ci_lo": 0.18, "lpips_ci_hi": 0.22,
        "ci_level": 0.95, "n_bootstrap": 10000, "avg_time_sec": 0.18,
    }
    with open(os.path.join(d, "aggregate_metrics.json"), "w") as f:
        json.dump(agg, f)
    with open(os.path.join(d, "test_metrics.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["image", "psnr", "ssim", "elapsed_sec", "lpips"])
        for i in range(n_rows):
            w.writerow([f"img_{i}.jpg", psnr_mean + i, 0.9, 0.18, 0.2])


def test_aggregate_from_metrics_builds_summary_and_per_image_rows(tmp_path):
    """Summary keyed by display name; rows carry a `model` column (notebook 04 contract)."""
    cm = _load_compare_methods_module()
    root = str(tmp_path / "metrics")
    _write_model_metrics(root, "finetuned", psnr_mean=23.29, n_rows=3)
    _write_model_metrics(root, "deoldify", psnr_mean=24.0, n_rows=2)

    summary, rows = cm.aggregate_from_metrics(root)

    assert "Zhang16 Fine-tuned (Ours)" in summary
    assert "DeOldify (GAN)" in summary
    assert summary["Zhang16 Fine-tuned (Ours)"]["psnr_mean"] == 23.29
    assert summary["Zhang16 Fine-tuned (Ours)"]["category"] == "CNN"
    # Per-image rows: 3 + 2, each tagged with the display name under `model`.
    assert len(rows) == 5
    assert all("model" in r for r in rows)
    assert {r["model"] for r in rows} == {"Zhang16 Fine-tuned (Ours)", "DeOldify (GAN)"}


def test_aggregate_records_controlnet_as_documented_gap(tmp_path):
    """ControlNet has no metrics dir; it must appear as a gap, never a fabricated row."""
    cm = _load_compare_methods_module()
    root = str(tmp_path / "metrics")
    _write_model_metrics(root, "finetuned", psnr_mean=23.29, n_rows=1)

    summary, rows = cm.aggregate_from_metrics(root)

    assert "ControlNet (Diffusion)" in summary
    gap = summary["ControlNet (Diffusion)"]
    assert gap["status"] == "gap"
    assert "psnr_mean" not in gap          # no invented number
    assert gap.get("reason")               # documents why
    # The gap contributes no per-image rows.
    assert all(r["model"] != "ControlNet (Diffusion)" for r in rows)


def test_aggregate_skips_models_with_no_metrics_dir(tmp_path):
    """A missing tag is skipped (not crashed): summary lists only what was evaluated."""
    cm = _load_compare_methods_module()
    root = str(tmp_path / "metrics")
    _write_model_metrics(root, "deoldify", psnr_mean=24.0, n_rows=1)

    summary, _ = cm.aggregate_from_metrics(root)

    ok_models = {n for n, e in summary.items() if e.get("status") == "ok"}
    assert ok_models == {"DeOldify (GAN)"}   # pretrained/finetuned/zhang17 absent


def test_write_comparison_artifacts_csv_has_model_column(tmp_path):
    """comparison_metrics.csv must expose `model` so notebook 04's groupby works."""
    cm = _load_compare_methods_module()
    summary = {"DeOldify (GAN)": {"status": "ok", "psnr_mean": 24.0}}
    rows = [{"image": "a.jpg", "model": "DeOldify (GAN)", "psnr": 24.0,
             "ssim": 0.9, "lpips": 0.15, "elapsed_sec": 0.5}]
    out = str(tmp_path / "comparison")

    summary_path, csv_path = cm.write_comparison_artifacts(summary, rows, out)

    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        assert "model" in reader.fieldnames
        loaded = list(reader)
    assert loaded[0]["model"] == "DeOldify (GAN)"
    with open(summary_path) as f:
        assert "DeOldify (GAN)" in json.load(f)


def test_make_bar_chart_writes_png(tmp_path):
    """Bar chart renders for the ok models and saves a non-empty PNG."""
    cm = _load_compare_methods_module()
    summary = {
        "Zhang16 Fine-tuned (Ours)": {
            "status": "ok", "category": "CNN",
            "psnr_mean": 23.29, "psnr_ci_lo": 23.0, "psnr_ci_hi": 23.5,
            "ssim_mean": 0.92, "ssim_ci_lo": 0.91, "ssim_ci_hi": 0.93,
            "lpips_mean": 0.19, "lpips_ci_lo": 0.18, "lpips_ci_hi": 0.20,
        },
        "DeOldify (GAN)": {
            "status": "ok", "category": "GAN",
            "psnr_mean": 24.0, "psnr_ci_lo": 23.7, "psnr_ci_hi": 24.2,
            "ssim_mean": 0.92, "ssim_ci_lo": 0.91, "ssim_ci_hi": 0.92,
            "lpips_mean": 0.15, "lpips_ci_lo": 0.14, "lpips_ci_hi": 0.15,
        },
        "ControlNet (Diffusion)": {"status": "gap", "reason": "x"},
    }
    path = str(tmp_path / "metrics_bar_chart.png")
    cm.make_bar_chart(summary, path)
    assert os.path.exists(path) and os.path.getsize(path) > 0


def test_make_training_curves_writes_png(tmp_path):
    """Training-curve figure renders from a training_log.json."""
    cm = _load_compare_methods_module()
    log = {
        "train_loss": [3.5, 3.4, 3.3, 3.2, 3.1, 3.0],
        "val_loss":   [3.5, 16.8, 3.3, 3.2, 3.1, 3.6],   # epoch-2 outlier
        "val_psnr":   [22.8, 22.6, 23.0, 22.9, 23.2, 23.6],
        "val_ssim":   [0.91, 0.91, 0.92, 0.91, 0.91, 0.92],
    }
    log_path = tmp_path / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f)
    path = str(tmp_path / "training_curves.png")
    cm.make_training_curves(str(log_path), path)
    assert os.path.exists(path) and os.path.getsize(path) > 0
