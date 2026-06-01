"""Tests for tools/compare_methods.py.

The script's docstring advertises a `--methods zhang16,deoldify` filter so users
can run a subset of the comparison (e.g., evaluate only Zhang16 + DeOldify when
iterating). The filtering helper lives on the tool module so it can be unit
tested without launching torch/HF.

Note: the diffusion paradigm is excluded from the benchmark — see the
"Why diffusion is excluded" section of reports/.../experiments.tex.
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


def test_aggregate_does_not_emit_controlnet_diffusion_entry(tmp_path):
    """The diffusion paradigm is excluded from the benchmark; the aggregator
    must not synthesize a ControlNet (Diffusion) row of any kind."""
    cm = _load_compare_methods_module()
    root = str(tmp_path / "metrics")
    _write_model_metrics(root, "finetuned", psnr_mean=23.29, n_rows=1)

    summary, rows = cm.aggregate_from_metrics(root)

    assert "ControlNet (Diffusion)" not in summary
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
    }
    path = str(tmp_path / "metrics_bar_chart.png")
    cm.make_bar_chart(summary, path)
    assert os.path.exists(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Per-image analysis (the "which model is best for which kind of picture" story)
# ---------------------------------------------------------------------------

def _row(image, model, psnr, ssim, lpips):
    return {"image": image, "model": model, "psnr": str(psnr),
            "ssim": str(ssim), "lpips": str(lpips), "elapsed_sec": "0.1"}


def test_analyze_counts_per_metric_winners():
    """Per-metric win counts respect direction (PSNR/SSIM up, LPIPS down)."""
    cm = _load_compare_methods_module()
    rows = [
        # img1: A wins psnr+ssim, C wins lpips (lowest)
        _row("img1.jpg", "A", 25, 0.90, 0.20),
        _row("img1.jpg", "B", 20, 0.85, 0.30),
        _row("img1.jpg", "C", 18, 0.80, 0.10),
        # img2: B wins psnr+ssim, A wins lpips
        _row("img2.jpg", "A", 15, 0.70, 0.15),
        _row("img2.jpg", "B", 22, 0.90, 0.25),
        _row("img2.jpg", "C", 19, 0.82, 0.20),
    ]
    out = cm.analyze_per_image(rows)
    assert out["n_images"] == 2
    assert out["win_counts"]["psnr"] == {"A": 1, "B": 1, "C": 0}
    assert out["win_counts"]["ssim"] == {"A": 1, "B": 1, "C": 0}
    assert out["win_counts"]["lpips"] == {"A": 1, "B": 0, "C": 1}


def test_analyze_strengths_pick_largest_psnr_margin_first():
    """`strengths[model][0]` must be the image where `model`'s PSNR margin over
    the runner-up is largest --- the row's headline 'this is the picture this
    model is uniquely good at'."""
    cm = _load_compare_methods_module()
    rows = [
        # img1: A wins by 5 dB over B
        _row("img1.jpg", "A", 25, 0.9, 0.2),
        _row("img1.jpg", "B", 20, 0.85, 0.25),
        # img2: A wins by 1 dB over B
        _row("img2.jpg", "A", 22, 0.9, 0.2),
        _row("img2.jpg", "B", 21, 0.85, 0.25),
    ]
    out = cm.analyze_per_image(rows)
    top = out["strengths"]["A"][0]
    assert top["image"] == "img1.jpg"
    assert abs(top["margin"] - 5.0) < 1e-9
    assert top["second_best_model"] == "B"


def test_analyze_win_rates_sum_to_one_per_metric():
    cm = _load_compare_methods_module()
    rows = [
        _row("img1.jpg", "A", 25, 0.9, 0.2),
        _row("img1.jpg", "B", 20, 0.85, 0.25),
        _row("img2.jpg", "A", 21, 0.85, 0.3),
        _row("img2.jpg", "B", 22, 0.90, 0.2),
    ]
    out = cm.analyze_per_image(rows)
    for metric in ("psnr", "ssim", "lpips"):
        total = sum(out["win_rate"][metric].values())
        assert abs(total - 1.0) < 1e-9, f"{metric} win_rate sums to {total}"


def test_analyze_skips_non_numeric_metric_cells():
    """Empty / non-numeric metric strings (e.g., missing LPIPS) must not crash."""
    cm = _load_compare_methods_module()
    rows = [
        _row("img1.jpg", "A", 25, 0.9, ""),    # missing lpips
        _row("img1.jpg", "B", 20, 0.85, 0.3),
    ]
    out = cm.analyze_per_image(rows)
    assert out["win_counts"]["psnr"]["A"] == 1
    # lpips: only B has a numeric value -> no comparison possible -> no winner
    assert out["win_counts"]["lpips"] == {"A": 0, "B": 0}


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
