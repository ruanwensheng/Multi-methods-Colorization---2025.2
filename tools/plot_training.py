"""Plot training-loss curves for the Zhang16 fine-tune run.

Two data sources are supported:

  1. `models/deep_learning/training_log.json` — per-epoch metrics written by
     `Trainer.fit()` after each epoch completes. Use this once epoch 1 is
     done; for short-history runs it draws four lines (train_loss, val_loss,
     val_psnr, val_ssim) across the epochs trained so far.

  2. `logs/train_*.log` — the raw stdout from `train_deep.py`, captured via
     `tee`. We parse the tqdm progress lines (`Epoch N: ... loss=X.XXXX`) to
     recover per-batch training loss. Use this MID-EPOCH when no epoch has
     completed yet but you want a "is loss going down?" sanity check.

Both modes are non-fatal — if the chosen source doesn't exist or is empty,
we print a hint and exit 0.

Usage:
  python tools/plot_training.py
  python tools/plot_training.py --log logs/train_smoke_v2.log
  python tools/plot_training.py --history models/deep_learning/training_log.json \\
                                --out results/deep_learning/figures/training_curves.png
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # safe for headless / WSL / CI; we save PNGs, not show
import matplotlib.pyplot as plt


# --- Data extraction -------------------------------------------------------

# tqdm line pattern. Matches both newline-separated and \r-overwritten output
# (tee captures \r as a same-line concatenation, hence non-greedy matching).
# Captures: epoch number, batch index, loss float.
_TQDM_LOSS_RE = re.compile(
    r"Epoch\s+(\d+):\s*\d+%\|[^|]*\|\s*(\d+)/\d+\s+\[[^\]]+,\s*loss=([0-9]+\.[0-9]+)\]"
)


def parse_tqdm_log_for_batch_loss(text):
    """Extract (epoch, batch_idx, loss) tuples from raw tqdm output.

    Args:
        text: The full text content of a training log file. May contain
            embedded \\r characters where tqdm overwrote its own line.

    Returns:
        List of (epoch:int, batch_idx:int, loss:float) tuples in the order
        they appear in the log.
    """
    if not text:
        return []
    matches = _TQDM_LOSS_RE.findall(text)
    return [(int(e), int(b), float(loss)) for e, b, loss in matches]


def load_epoch_history(path):
    """Load training_log.json. Returns the dict, or None if missing/empty."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    train_loss = data.get("train_loss") or []
    if not train_loss:
        return None
    return data


# --- Plotting --------------------------------------------------------------

def plot_epoch_history(history, out_path):
    """Render a 2x2 grid of train/val curves to out_path."""
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(epochs, history.get("train_loss", []), label="train", marker="o")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="val", marker="s")
    ax.set_title("Loss (lower is better)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history.get("val_psnr", []), color="C2", marker="o")
    ax.set_title("Validation PSNR (higher is better)")
    ax.set_xlabel("epoch"); ax.set_ylabel("PSNR (dB)"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history.get("val_ssim", []), color="C3", marker="o")
    ax.set_title("Validation SSIM (higher is better)")
    ax.set_xlabel("epoch"); ax.set_ylabel("SSIM"); ax.grid(alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05, 0.95,
        f"epochs trained: {len(epochs)}\n"
        f"best val_loss:  {min(history.get('val_loss', [float('inf')])):.4f}\n"
        f"best val_psnr:  {max(history.get('val_psnr', [0])):.2f} dB\n"
        f"best val_ssim:  {max(history.get('val_ssim', [0])):.4f}",
        family="monospace", va="top", fontsize=11,
    )

    fig.suptitle("Zhang16 fine-tune on COCO 2017", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_batch_loss(samples, out_path):
    """Render per-batch loss from parsed tqdm samples."""
    if not samples:
        return False
    epochs = sorted({e for e, _, _ in samples})
    fig, ax = plt.subplots(figsize=(11, 5))
    for e in epochs:
        xs = [b for ep, b, _ in samples if ep == e]
        ys = [loss for ep, _, loss in samples if ep == e]
        ax.plot(xs, ys, label=f"epoch {e}", alpha=0.7, linewidth=0.8)
    ax.set_title(f"Per-batch training loss ({len(samples)} samples)")
    ax.set_xlabel("batch index"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# --- CLI -------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument(
        "--history",
        default="models/deep_learning/training_log.json",
        help="Path to the per-epoch training_log.json (default: %(default)s)",
    )
    parser.add_argument(
        "--log",
        default=None,
        help="Optional path to a raw stdout log; parses tqdm 'loss=...' lines for per-batch loss",
    )
    parser.add_argument(
        "--out",
        default="results/deep_learning/figures/training_curves.png",
        help="Where to write the PNG (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    # Prefer per-epoch when available
    history = load_epoch_history(args.history)
    if history is not None:
        plot_epoch_history(history, args.out)
        print(f"Wrote per-epoch curves -> {args.out}")
        return 0

    # Fall back to mid-epoch tqdm parse
    if args.log and os.path.exists(args.log):
        with open(args.log, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        samples = parse_tqdm_log_for_batch_loss(text)
        if samples:
            plot_batch_loss(samples, args.out)
            print(f"Wrote per-batch curves -> {args.out}  ({len(samples)} samples)")
            return 0
        print(f"No tqdm 'loss=...' lines found in {args.log}.")
        return 0

    print(
        f"No data to plot yet.\n"
        f"  Looked for --history at: {args.history} (not present or empty)\n"
        f"  Looked for --log     at: {args.log or '(not supplied)'}\n"
        f"Once epoch 1 finishes, training_log.json appears and this script writes the curves."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
