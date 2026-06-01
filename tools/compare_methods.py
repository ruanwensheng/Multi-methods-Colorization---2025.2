"""
Compare multiple colorization methods on the shared benchmark.

Usage:
    python tools/compare_methods.py
    python tools/compare_methods.py --methods zhang16,deoldify --max-images 50

Runs all available methods on the benchmark test set and produces
comparison metrics and side-by-side visualizations.
"""

import os
import sys
import argparse
import json
import csv

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # save-only tool, never displays — keep it headless-safe
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pin caches to project drive BEFORE torch/HF imports — keeps C: from filling.
from src.deep_learning.bootstrap import setup_caches
setup_caches()

from src.deep_learning.utils import load_config, compute_metrics
from src.deep_learning.colorizer import DeepColorizer
from src.deep_learning.pretrained import get_comparison_models
from src.deep_learning.stats import bootstrap_ci


# User-facing slug -> patterns to substring-match against lowercased model names.
# Each user slug expands to one or more candidate patterns; a model matches if
# ANY pattern is a substring of its lowercased name. This handles the common
# shorthand `zhang17` -> "Zhang 2017 (Interactive CNN)" where a literal substring
# would miss because of the space inside "zhang 2017".
_SLUG_ALIASES = {
    "zhang17":   ["zhang 2017", "zhang17"],
    "zhang2017": ["zhang 2017", "zhang2017"],
    "zhang16":   ["zhang16", "zhang 2016"],
    "zhang2016": ["zhang16", "zhang 2016"],
}


def filter_models(models, method_slugs):
    """Filter a {name: colorizer} dict by case-insensitive substring slugs.

    `method_slugs=None` returns the dict unchanged; an empty list returns {};
    unknown slugs raise ValueError so a typo doesn't silently no-op the run.
    Slug expansion via `_SLUG_ALIASES` lets `zhang17` match "Zhang 2017
    (Interactive CNN)" despite the space — useful since the model names
    aren't user-friendly slugs.
    """
    if method_slugs is None:
        return models
    method_slugs = [s.strip().lower() for s in method_slugs if s.strip()]
    if not method_slugs:
        return {}
    kept = {}
    matched = {s: False for s in method_slugs}
    for name, model in models.items():
        name_lc = name.lower()
        for slug in method_slugs:
            patterns = _SLUG_ALIASES.get(slug, [slug])
            if any(p in name_lc for p in patterns):
                kept[name] = model
                matched[slug] = True
                break
    missing = [s for s, hit in matched.items() if not hit]
    if missing:
        raise ValueError(
            f"--methods slug(s) not found in loaded models: {missing}. "
            f"Available: {sorted(models)}"
        )
    return kept


# =============================================================================
# Aggregate the cross-model comparison from canonical per-model metrics
# =============================================================================
#
# Phase 4 does NOT re-run the models. Re-running on a small subset would produce
# a second, weaker set of numbers that disagree with the Phase-3 table (which is
# computed by evaluate_deep.py on the full 1,000-image test2017 benchmark). The
# project's single source of truth is that benchmark (see CLAUDE.md and
# docs/benchmark_methodology.md), so we assemble the comparison straight from
# each model's evaluate_deep.py output.

# (on-disk metrics tag, display name, paradigm category) in report narrative order.
_EVALUATED_MODELS = [
    ("pretrained", "Zhang16 Pretrained",           "CNN"),
    ("finetuned",  "Zhang16 Fine-tuned (Ours)",    "CNN"),
    ("zhang17",    "Zhang 2017 (Interactive CNN)", "Interactive CNN"),
    ("deoldify",   "DeOldify (GAN)",               "GAN"),
]

def aggregate_from_metrics(metrics_root):
    """Assemble the cross-model comparison from per-model evaluate_deep.py outputs.

    Reads each model's ``aggregate_metrics.json`` (mean/CI over the full
    1,000-image benchmark) and ``test_metrics.csv`` (per-image rows). No model is
    re-run, so numbers match the canonical Phase-3 table exactly.

    Args:
        metrics_root: dir containing one subdir per model tag (results/.../metrics).

    Returns:
        (summary, rows):
          summary: dict display_name -> aggregate entry. Each evaluated model
                   carries ``status="ok"`` plus the evaluate_deep.py fields and
                   a ``category``.
          rows:    list of per-image dicts, each tagged with a ``model`` column
                   (the display name) so notebook 04's groupby("model") works.
    """
    summary = {}
    rows = []
    for tag, display, category in _EVALUATED_MODELS:
        agg_path = os.path.join(metrics_root, tag, "aggregate_metrics.json")
        if not os.path.exists(agg_path):
            continue  # model wasn't evaluated — skip rather than crash
        with open(agg_path) as f:
            entry = json.load(f)
        entry["category"] = category
        entry["status"] = "ok"
        summary[display] = entry

        csv_path = os.path.join(metrics_root, tag, "test_metrics.csv")
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                for r in csv.DictReader(f):
                    rows.append({
                        "image":       r.get("image", ""),
                        "model":       display,
                        "psnr":        r.get("psnr", ""),
                        "ssim":        r.get("ssim", ""),
                        "lpips":       r.get("lpips", ""),
                        "elapsed_sec": r.get("elapsed_sec", ""),
                    })

    return summary, rows


def analyze_per_image(rows, top_k_strengths=5):
    """Per-image winners + per-model strengths from the row-level comparison data.

    Answers the question we care about for the report: ``which model is best at
    which kind of picture, and how much better?'' For every benchmark image we
    rank the models by each metric (higher is better for PSNR/SSIM, lower for
    LPIPS) and tally the wins; for each model we then list the top-K images
    where its PSNR margin over the runner-up is largest --- these are the
    ``this is where this model shines'' exemplars used by the strengths figure.

    Non-numeric metric cells (empty strings, ``None``) are simply skipped; an
    image with fewer than two models scoring a given metric contributes no
    winner for that metric.
    """
    # group by image, in arrival order (model order discovered as we go)
    by_image, models_seen = {}, []
    for r in rows:
        img = r.get("image", ""); model = r.get("model", "")
        if not img or not model:
            continue
        if model not in models_seen:
            models_seen.append(model)
        by_image.setdefault(img, {})[model] = r

    metric_dirs = {"psnr": True, "ssim": True, "lpips": False}  # True = higher better
    win_counts = {m: {model: 0 for model in models_seen} for m in metric_dirs}
    n_counted = {m: 0 for m in metric_dirs}
    winners_per_image, psnr_margins = [], []

    for img, modeldata in by_image.items():
        row_winners = {"image": img}
        for metric, higher_better in metric_dirs.items():
            scored = []
            for model, r in modeldata.items():
                try:
                    scored.append((model, float(r.get(metric, ""))))
                except (ValueError, TypeError):
                    continue
            if len(scored) < 2:
                continue
            scored.sort(key=lambda mv: mv[1], reverse=higher_better)
            winner, w_val = scored[0]
            runner, r_val = scored[1]
            win_counts[metric][winner] += 1
            n_counted[metric] += 1
            row_winners[f"{metric}_winner"] = winner
            if metric == "psnr":
                psnr_margins.append({
                    "image": img, "winner": winner, "winner_value": w_val,
                    "second_best_model": runner, "second_best_value": r_val,
                    "margin": w_val - r_val,
                })
        winners_per_image.append(row_winners)

    win_rate = {m: {model: (win_counts[m][model] / n_counted[m] if n_counted[m] else 0.0)
                    for model in models_seen} for m in metric_dirs}

    strengths = {}
    for model in models_seen:
        wins = [x for x in psnr_margins if x["winner"] == model]
        wins.sort(key=lambda x: -x["margin"])
        strengths[model] = wins[:top_k_strengths]

    return {
        "n_images": len(by_image),
        "models": list(models_seen),
        "win_counts": win_counts,
        "win_rate": win_rate,
        "strengths": strengths,
        "winners_per_image": winners_per_image,
    }


def write_comparison_artifacts(summary, rows, output_dir):
    """Write comparison_summary.json + comparison_metrics.csv. Returns their paths."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    fields = ["image", "model", "psnr", "ssim", "lpips", "elapsed_sec"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, csv_path


# =============================================================================
# Figures (T4.2)
# =============================================================================

def make_bar_chart(summary, save_path):
    """Render PSNR/SSIM/LPIPS bar charts with 95% CI error bars for evaluated models.

    Bars are sorted best-first per metric (LPIPS is lower-is-better).
    """
    ok = {n: e for n, e in summary.items() if e.get("status") == "ok"}
    panels = [("psnr", "PSNR (dB) — higher is better", False),
              ("ssim", "SSIM — higher is better", False),
              ("lpips", "LPIPS — lower is better", True)]

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
    for ax, (key, title, lower_better) in zip(axes, panels):
        items = [(n, e) for n, e in ok.items() if f"{key}_mean" in e]
        items.sort(key=lambda kv: kv[1][f"{key}_mean"], reverse=not lower_better)
        names = [n for n, _ in items]
        means = [e[f"{key}_mean"] for _, e in items]
        lo = [e[f"{key}_mean"] - e.get(f"{key}_ci_lo", e[f"{key}_mean"]) for _, e in items]
        hi = [e.get(f"{key}_ci_hi", e[f"{key}_mean"]) - e[f"{key}_mean"] for _, e in items]
        x = range(len(names))
        ax.bar(x, means, yerr=[lo, hi], capsize=4,
               color=plt.cm.viridis(np.linspace(0.15, 0.85, max(len(names), 1))))
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Cross-model comparison on the 1,000-image test2017 benchmark "
                 "(error bars = 95% bootstrap CI)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_training_curves(training_log_path, save_path, loss_ymax=5.0):
    """Plot fine-tuning curves from models/deep_learning/training_log.json.

    Left panel: train/val loss vs epoch. The y-axis is capped at ``loss_ymax`` so
    the curve stays readable — two val-loss spikes (pathological batches, see
    commit 50d798d) would otherwise flatten everything; off-scale points are
    annotated rather than dropped. Right panel: val PSNR vs epoch.
    """
    with open(training_log_path) as f:
        log = json.load(f)
    train = log.get("train_loss", [])
    val = log.get("val_loss", [])
    psnr = log.get("val_psnr", [])
    epochs = list(range(1, len(train) + 1))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))

    axL.plot(epochs, train, "o-", label="train loss")
    # Break the val line at off-scale outliers (NaN) so they don't draw vertical
    # streaks; mark their true value with an annotation instead.
    val_plot = [v if v <= loss_ymax else np.nan for v in val]
    axL.plot(range(1, len(val) + 1), val_plot, "s-", label="val loss", color="tab:orange")
    axL.set_ylim(min(train + [v for v in val if v <= loss_ymax]) - 0.1, loss_ymax)
    for ep, v in zip(range(1, len(val) + 1), val):
        if v > loss_ymax:  # off-scale outlier — annotate at the top of the axis
            axL.annotate(f"{v:.1f}", xy=(ep, loss_ymax), xytext=(ep, loss_ymax - 0.3),
                         ha="center", fontsize=7, color="tab:red",
                         arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    axL.set_xlabel("epoch"); axL.set_ylabel("loss")
    axL.set_title("Training / validation loss (val outliers clipped)")
    axL.legend(); axL.grid(alpha=0.3)

    if psnr:
        axR.plot(range(1, len(psnr) + 1), psnr, "^-", color="tab:green")
        best_ep = int(np.argmax(psnr)) + 1
        axR.annotate(f"best {psnr[best_ep - 1]:.2f} dB (ep {best_ep})",
                     xy=(best_ep, psnr[best_ep - 1]),
                     xytext=(0.5, 0.1), textcoords="axes fraction", fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axR.set_xlabel("epoch"); axR.set_ylabel("val PSNR (dB)")
    axR.set_title("Validation PSNR")
    axR.grid(alpha=0.3)

    fig.suptitle("Zhang16 fine-tuning on COCO 2017", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_evaluated_models(cfg, ft_path, pretrained_path=None, device="auto"):
    """Load the four evaluated models for a qualitative grid.

    Differs from ``load_all_models`` in that it instantiates Zhang16 *twice*
    with explicit labels --- the pretrained ECCV-init model and our fine-tuned
    checkpoint --- so the figure can show, in one frame, what fine-tuning bought
    visually. Third-party comparisons (Zhang17, DeOldify) are added through
    ``get_comparison_models``. Returned dict order = column order in the grid.
    """
    models = {}
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            models["Zhang16 Pretrained"] = DeepColorizer(
                model_path=pretrained_path, cfg=cfg, device=device)
            print(f"  Loaded: Zhang16 Pretrained from {pretrained_path}")
        except Exception as e:
            print(f"  Zhang16 Pretrained load failed: {e}")
    if ft_path and os.path.exists(ft_path):
        try:
            models["Zhang16 Fine-tuned (Ours)"] = DeepColorizer(
                model_path=ft_path, cfg=cfg, device=device)
            print(f"  Loaded: Zhang16 Fine-tuned (Ours) from {ft_path}")
        except Exception as e:
            print(f"  Zhang16 Fine-tuned load failed: {e}")
    for name, m in get_comparison_models(cfg).items():
        models[name] = m
        print(f"  Loaded: {name}")
    return models


def load_all_models(cfg, model_path=None, device="auto"):
    """Load all available colorization models.

    Returns:
        dict mapping model_name -> colorizer instance.
    """
    models = {}

    # Our Zhang16 model
    if model_path and os.path.exists(model_path):
        try:
            colorizer = DeepColorizer(model_path=model_path, cfg=cfg, device=device)
            models["Zhang16 (Ours)"] = colorizer
            print(f"  Loaded: Zhang16 (Ours) from {model_path}")
        except Exception as e:
            print(f"  Failed to load Zhang16: {e}")
    else:
        # Use untrained model as placeholder
        colorizer = DeepColorizer(cfg=cfg, device=device)
        models["Zhang16 (Ours)"] = colorizer
        print("  Loaded: Zhang16 (Ours) - untrained")

    # Pre-trained comparison models
    comparison = get_comparison_models(cfg)
    for name, model in comparison.items():
        models[name] = model
        print(f"  Loaded: {name}")

    return models


def compare_on_image(models, img_bgr):
    """Run all models on a single image and collect results.

    Returns:
        dict mapping model_name -> (pred_rgb, metrics, elapsed).
    """
    gt_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    results = {}
    for name, model in models.items():
        try:
            if hasattr(model, "colorize"):
                result_bgr, info = model.colorize(gray)
            else:
                continue

            pred_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
                pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))

            metrics = compute_metrics(pred_rgb, gt_rgb)
            results[name] = (pred_rgb, metrics, info.get("elapsed_sec", 0))
        except Exception as e:
            print(f"  Warning: {name} failed: {e}")

    return results, gray, gt_rgb


def save_comparison_grid(gray, gt_rgb, results, save_path, img_name=""):
    """Save a side-by-side comparison visualization."""
    n_methods = len(results) + 2  # input + ground truth + methods
    fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 3.5))

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Input", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=9)
    axes[1].axis("off")

    for i, (name, (pred, metrics, _)) in enumerate(results.items()):
        axes[i + 2].imshow(pred)
        psnr = metrics.get("psnr", 0)
        ssim = metrics.get("ssim", 0)
        axes[i + 2].set_title(f"{name}\nPSNR:{psnr:.1f} SSIM:{ssim:.3f}", fontsize=8)
        axes[i + 2].axis("off")

    if img_name:
        fig.suptitle(img_name, fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_qualitative_grid(models, test_dir, image_names, save_path):
    """Render rows=images x cols=[Input, *models, Ground Truth] into one PNG.

    This re-runs the given models on a handful of benchmark images purely to
    produce the visual figure — it computes no aggregate metrics, so the report's
    quantitative table (from ``aggregate_from_metrics``) is unaffected. Models that
    fail to produce a prediction are dropped from the columns rather than aborting.
    """
    per_image = []  # (img_name, gray, gt_rgb, {model_name: pred_rgb})
    for img_name in image_names:
        img_bgr = cv2.imread(os.path.join(test_dir, img_name))
        if img_bgr is None:
            continue
        results, gray, gt_rgb = compare_on_image(models, img_bgr)
        preds = {name: pred for name, (pred, _m, _t) in results.items()}
        per_image.append((img_name, gray, gt_rgb, preds))

    if not per_image:
        raise RuntimeError("no benchmark images could be read for the qualitative grid")

    # Keep only models that produced at least one prediction, in load order.
    model_names = [m for m in models if any(m in p[3] for p in per_image)]
    col_titles = ["Input"] + model_names + ["Ground Truth"]
    n_rows, n_cols = len(per_image), len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows))
    axes = np.atleast_2d(axes)
    for r, (_img_name, gray, gt_rgb, preds) in enumerate(per_image):
        axes[r][0].imshow(gray, cmap="gray")
        for c, m in enumerate(model_names, start=1):
            if m in preds:
                axes[r][c].imshow(preds[m])
            else:
                axes[r][c].text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8)
        axes[r][-1].imshow(gt_rgb)
        for c in range(n_cols):
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=9)

    fig.suptitle("Qualitative comparison on test2017 benchmark images", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [p[0] for p in per_image], model_names


def make_strengths_figure(models, test_dir, strengths, save_path):
    """One row per evaluated model: the benchmark image where its PSNR margin
    over the runner-up is largest, with every model's prediction side by side.

    The figure answers the question audiences keep asking: ``which model is
    best at which kind of picture?'' Rows show the headline (winning model,
    margin in dB, runner-up); columns are the same as in the qualitative grid.
    Models that never win get no row, so the figure scales gracefully.
    """
    picks = []  # (winner_name, strength_info)
    for name in models:
        s_list = strengths.get(name, [])
        if s_list:
            picks.append((name, s_list[0]))
    if not picks:
        return None

    col_titles = ["Input"] + list(models.keys()) + ["Ground Truth"]
    n_rows, n_cols = len(picks), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 2.9 * n_rows))
    axes = np.atleast_2d(axes)

    for r, (winner_name, info) in enumerate(picks):
        img_bgr = cv2.imread(os.path.join(test_dir, info["image"]))
        if img_bgr is None:
            continue
        results, gray, gt_rgb = compare_on_image(models, img_bgr)
        preds = {n: pred for n, (pred, _m, _t) in results.items()}
        axes[r][0].imshow(gray, cmap="gray")
        for c, m in enumerate(models, start=1):
            if m in preds:
                axes[r][c].imshow(preds[m])
            else:
                axes[r][c].text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8)
        axes[r][-1].imshow(gt_rgb)
        for c in range(n_cols):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(col_titles[c], fontsize=9)
        axes[r][0].set_ylabel(
            f"{winner_name}\nwins by\n{info['margin']:+.2f} dB\n(vs {info['second_best_model']})",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=90,
        )

    fig.suptitle("Per-model strengths: for each model, the benchmark image where its "
                 "PSNR margin over the runner-up is largest", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [(p[0], p[1]["image"], p[1]["margin"]) for p in picks]


def _run_from_metrics(cfg, args):
    """T4.1 + T4.2: assemble the comparison and figures from canonical per-model
    metrics (no model re-run). Optionally renders the qualitative grid, which is
    the only step that loads models (visual only)."""
    metrics_dir = args.metrics_dir or os.path.join(cfg["paths"]["results_deep"], "metrics")
    output_dir = args.output_dir or os.path.join(cfg["paths"]["results_deep"], "comparison")
    figures_dir = args.figures_dir or os.path.join(cfg["paths"]["results_deep"], "figures")
    test_dir = args.test_dir or os.path.join(cfg["paths"]["data_coco"], "benchmark")

    print(f"Aggregating canonical per-model metrics from: {metrics_dir}")
    summary, rows = aggregate_from_metrics(metrics_dir)
    ok = [n for n, e in summary.items() if e.get("status") == "ok"]
    gaps = [n for n, e in summary.items() if e.get("status") == "gap"]
    if not ok:
        print(f"ERROR: no per-model aggregate_metrics.json found under {metrics_dir}")
        sys.exit(1)

    summary_path, csv_path = write_comparison_artifacts(summary, rows, output_dir)
    print(f"  evaluated models: {ok}")
    print(f"  documented gaps : {gaps}")
    print(f"  wrote {summary_path}")
    print(f"  wrote {csv_path} ({len(rows)} per-image rows)")

    print(f"{'='*70}\n{'Model':<30}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}\n{'='*70}")
    for name in ok:
        e = summary[name]
        print(f"{name:<30}{e.get('psnr_mean', float('nan')):>10.2f}"
              f"{e.get('ssim_mean', float('nan')):>10.4f}{e.get('lpips_mean', float('nan')):>10.4f}")
    print('='*70)

    # Per-image analysis ("which model wins which image") — always; cheap, pure.
    analysis = analyze_per_image(rows)
    win_path = os.path.join(output_dir, "win_rates.json")
    with open(win_path, "w") as f:
        json.dump({"n_images":  analysis["n_images"],
                   "models":    analysis["models"],
                   "win_counts": analysis["win_counts"],
                   "win_rate":   analysis["win_rate"]}, f, indent=2)
    winners_csv = os.path.join(output_dir, "per_image_winners.csv")
    with open(winners_csv, "w", newline="") as f:
        fields = ["image", "psnr_winner", "ssim_winner", "lpips_winner"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in analysis["winners_per_image"]:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  wrote {win_path}")
    print(f"  wrote {winners_csv}")
    print("  PSNR wins: " + ", ".join(
        f"{m} {100 * analysis['win_rate']['psnr'][m]:.1f}%" for m in analysis["models"]))
    print("  LPIPS wins: " + ", ".join(
        f"{m} {100 * analysis['win_rate']['lpips'][m]:.1f}%" for m in analysis["models"]))

    bar_path = os.path.join(figures_dir, "metrics_bar_chart.png")
    make_bar_chart(summary, bar_path)
    print(f"  wrote {bar_path}")

    if os.path.exists(args.training_log):
        curves_path = os.path.join(figures_dir, "training_curves.png")
        make_training_curves(args.training_log, curves_path)
        print(f"  wrote {curves_path}")
    else:
        print(f"  (skipped training curves — no log at {args.training_log})")

    # Model-loading is the only GPU step; do it once if either visual is requested.
    needs_models = args.qualitative_grid > 0 or args.strengths_figure
    if needs_models:
        print("\nLoading models for figure rendering...")
        models = load_evaluated_models(
            cfg, ft_path=args.model_path,
            pretrained_path=args.pretrained_path, device=args.device,
        )
        if not models:
            print("  no comparison models available in this env — skipping figures")
            models = None
    else:
        models = None

    if models and args.qualitative_grid > 0:
        extensions = {".jpg", ".jpeg", ".png"}
        names = sorted(f for f in os.listdir(test_dir)
                       if os.path.splitext(f)[1].lower() in extensions)[:args.qualitative_grid]
        grid_path = os.path.join(figures_dir, "qualitative_grid.png")
        used_imgs, used_models = make_qualitative_grid(models, test_dir, names, grid_path)
        print(f"  wrote {grid_path} ({len(used_imgs)} imgs x {len(used_models)} models: {used_models})")

    if models and args.strengths_figure:
        strengths_path = os.path.join(figures_dir, "model_strengths.png")
        used = make_strengths_figure(models, test_dir, analysis["strengths"], strengths_path)
        if used:
            print(f"  wrote {strengths_path} — " + "; ".join(
                f"{m}: {img} (+{margin:.2f} dB)" for m, img, margin in used))
        else:
            print("  no PSNR wins among loaded models — strengths figure skipped")

    print(f"\nDone. Comparison in {output_dir}, figures in {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare colorization methods")
    parser.add_argument("--model-path", default="models/deep_learning/best_model.pth",
                        help="Path to our trained model")
    parser.add_argument("--test-dir", default=None, help="Test image directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-images", type=int, default=50, help="Max images to compare")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--device", default="auto", help="Device")
    parser.add_argument(
        "--methods", default=None,
        help="Comma-separated subset of model slugs to include "
             "(e.g. 'zhang16,zhang17,deoldify'). Case-insensitive substring "
             "match against loaded model names. Default: all loaded.",
    )
    parser.add_argument(
        "--from-metrics", action="store_true",
        help="Assemble the comparison + figures from the canonical per-model "
             "evaluate_deep.py outputs (full 1,000-image benchmark) instead of "
             "re-running the models. This is the single-source-of-truth path.",
    )
    parser.add_argument("--metrics-dir", default=None,
                        help="Per-model metrics dir (default: results_deep/metrics).")
    parser.add_argument("--figures-dir", default=None,
                        help="Figure output dir (default: results_deep/figures).")
    parser.add_argument("--training-log", default="models/deep_learning/training_log.json",
                        help="Training log JSON for the training-curves figure.")
    parser.add_argument("--qualitative-grid", type=int, default=0,
                        help="With --from-metrics: re-run models on N benchmark "
                             "images to render qualitative_grid.png (0 = skip).")
    parser.add_argument("--pretrained-path", default="models/pretrained/zhang16_eccv.pth",
                        help="Zhang16 ECCV checkpoint shown as the Pretrained column "
                             "of the qualitative grid (so the audience can see what "
                             "fine-tuning bought). Set to '' to skip.")
    parser.add_argument("--strengths-figure", action="store_true",
                        help="With --from-metrics: render model_strengths.png "
                             "(one row per model, showing the benchmark image "
                             "where its PSNR margin over the runner-up is largest).")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.from_metrics:
        _run_from_metrics(cfg, args)
        return

    test_dir = args.test_dir or os.path.join(cfg["paths"]["data_coco"], "benchmark")
    output_dir = args.output_dir or os.path.join(cfg["paths"]["results_deep"], "comparison")

    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "grids")
    os.makedirs(vis_dir, exist_ok=True)

    # Load models
    print("Loading models...")
    models = load_all_models(cfg, args.model_path, args.device)
    if args.methods is not None:
        slugs = [s for s in args.methods.split(",")]
        models = filter_models(models, slugs)
        print(f"  Filtered to: {list(models)}")
    print(f"  {len(models)} model(s) loaded\n")

    # Get test images
    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([
        f for f in os.listdir(test_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])[:args.max_images]

    # Run comparison
    all_results = {name: [] for name in models}

    for img_name in tqdm(images, desc="Comparing"):
        img_path = os.path.join(test_dir, img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        results, gray, gt_rgb = compare_on_image(models, img_bgr)

        for name, (pred, metrics, elapsed) in results.items():
            all_results[name].append({
                "image": img_name,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics.get("lpips", None),
                "elapsed_sec": elapsed,
            })

        # Save comparison grid for first 10 images
        if len(all_results[list(models.keys())[0]]) <= 10:
            save_comparison_grid(
                gray, gt_rgb, results,
                os.path.join(vis_dir, f"compare_{img_name}"),
                img_name=img_name
            )

    # Aggregate with bootstrap 95% CIs (10k resamples, seed=42)
    print(f"\n{'='*95}")
    print(f"{'Method':<25} {'PSNR [95% CI]':>22} {'SSIM [95% CI]':>22} {'Time/img':>12}")
    print(f"{'='*95}")

    summary = {}
    for name, results_list in all_results.items():
        if not results_list:
            continue
        psnr_ci = bootstrap_ci([r["psnr"] for r in results_list], ci=0.95, n_boot=10000, seed=42)
        ssim_ci = bootstrap_ci([r["ssim"] for r in results_list], ci=0.95, n_boot=10000, seed=42)
        time_mean = float(np.mean([r["elapsed_sec"] for r in results_list]))

        psnr_str = f"{psnr_ci['mean']:.2f} [{psnr_ci['ci_lo']:.2f}, {psnr_ci['ci_hi']:.2f}]"
        ssim_str = f"{ssim_ci['mean']:.4f} [{ssim_ci['ci_lo']:.4f}, {ssim_ci['ci_hi']:.4f}]"
        print(f"{name:<25} {psnr_str:>22} {ssim_str:>22} {time_mean:>10.3f}s")

        entry = {
            "num_images":  len(results_list),
            "psnr_mean":   psnr_ci["mean"],
            "psnr_std":    psnr_ci["std"],
            "psnr_ci_lo":  psnr_ci["ci_lo"],
            "psnr_ci_hi":  psnr_ci["ci_hi"],
            "ssim_mean":   ssim_ci["mean"],
            "ssim_std":    ssim_ci["std"],
            "ssim_ci_lo":  ssim_ci["ci_lo"],
            "ssim_ci_hi":  ssim_ci["ci_hi"],
            "avg_time_sec": time_mean,
        }

        # LPIPS if any image has it
        lpips_vals = [r["lpips"] for r in results_list if r.get("lpips") is not None]
        if lpips_vals:
            lpips_ci = bootstrap_ci(lpips_vals, ci=0.95, n_boot=10000, seed=42)
            entry.update({
                "lpips_mean":  lpips_ci["mean"],
                "lpips_std":   lpips_ci["std"],
                "lpips_ci_lo": lpips_ci["ci_lo"],
                "lpips_ci_hi": lpips_ci["ci_hi"],
            })

        entry["ci_level"] = 0.95
        entry["n_bootstrap"] = 10000
        summary[name] = entry

    print(f"{'='*95}")

    # Save summary
    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-image CSV
    csv_path = os.path.join(output_dir, "comparison_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # `model` (not `method`) so notebook 04's df.groupby("model") works
        # regardless of whether the CSV came from this path or --from-metrics.
        writer.writerow(["image", "model", "psnr", "ssim", "lpips", "elapsed_sec"])
        for name, results_list in all_results.items():
            for r in results_list:
                writer.writerow([r["image"], name, r["psnr"], r["ssim"], r.get("lpips", ""), r["elapsed_sec"]])

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
