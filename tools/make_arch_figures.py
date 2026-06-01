"""Generate schematic architecture figures for the four colorization paradigms.

The figures are intentionally drawn ourselves (not copied from the source
papers) so that the report has a single visual style across paradigms and we
avoid figure-permission ambiguity. Layouts faithfully follow the original
papers; see references.bib for the source publications.

Outputs (saved to reports/deep_learning/figures/):
    zhang16_arch.png       - feed-forward classification CNN (Zhang 2016)
    zhang17_arch.png       - interactive CNN with hints branch (Zhang 2017)
    deoldify_arch.png      - U-Net + critic + NoGAN training (DeOldify)
    controlnet_arch.png    - frozen SD UNet + trainable ControlNet copy
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "deep_learning" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----- styling ---------------------------------------------------------------
ENC = "#cde8d3"   # encoder / downsampling
CTX = "#ffe6b3"   # dilated / context core
DEC = "#f2c8c8"   # decoder / upsample
HEAD = "#d9c8ef"  # output head / softmax
IO = "#d9e3f0"    # input/output tile
CRITIC = "#f4d4ad"  # adversarial critic
FROZEN = "#e0e0e0"  # frozen backbone
TRAIN = "#bfd8f7"   # trainable copy
ZEROCONV = "#ffd8d8"  # zero-conv adapters
HINT = "#fff2a8"  # user hint branch


def _box(ax, x, y, w, h, text, color, fs=8.5):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0.8, edgecolor="#444", facecolor=color,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs, linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, style="-", color="#444", lw=0.8):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, linestyle=style, color=color,
    )
    ax.add_patch(arr)


def _label(ax, x, y, text, color="#555", fs=8, fw="normal", ha="center"):
    ax.text(x, y, text, color=color, fontsize=fs,
            fontweight=fw, ha=ha, va="center")


# ----- Figure 1: Zhang 2016 (improved version of existing architecture.png) ----
def zhang16():
    fig, ax = plt.subplots(figsize=(11.5, 2.6), dpi=160)
    ax.set_xlim(0, 14); ax.set_ylim(0, 2.6); ax.axis("off")
    ax.set_title("Zhang 2016 — fully-convolutional classification network "
                 "(our reimplementation, 31.6 M params)",
                 fontsize=10, pad=6)

    # blocks: x, w, color, text
    seq = [
        (0.05, 0.95, IO,  "L input\n1×256×256"),
        (1.10, 0.95, ENC, "conv1\n64,  /2"),
        (2.15, 0.95, ENC, "conv2\n128, /4"),
        (3.20, 0.95, ENC, "conv3\n256, /8"),
        (4.25, 0.95, CTX, "conv4\n512, d=1"),
        (5.30, 0.95, CTX, "conv5\n512, d=2"),
        (6.35, 0.95, CTX, "conv6\n512, d=2"),
        (7.40, 0.95, CTX, "conv7\n512, d=1"),
        (8.45, 0.95, DEC, "conv8\nup→256\n1×1 → 233"),
        (9.50, 0.95, HEAD, "softmax\n233 bins\n64×64"),
        (10.55, 0.95, HEAD, "annealed-mean\nT=0.38 → ab"),
        (11.60, 1.40, IO, "ab up 256×256\n+ L → Lab → RGB"),
    ]
    y, h = 0.85, 0.95
    for x, w, col, txt in seq:
        _box(ax, x, y, w, h, txt, col)
    # connectors
    for i in range(len(seq) - 1):
        x0 = seq[i][0] + seq[i][1]
        x1 = seq[i + 1][0]
        _arrow(ax, x0, y + h / 2, x1, y + h / 2)

    # group labels
    _label(ax, 2.7, 0.55, "encoder (stride-2 downsampling)", color="#2e7d32",
           fs=8.2, fw="bold")
    _label(ax, 6.4, 0.55, "dilated context core (no further downsampling)",
           color="#a05a00", fs=8.2, fw="bold")
    _label(ax, 10.8, 2.25, "classify ab over quantized bins  →  decode  →  recombine with luminance",
           color="#666", fs=8.2, ha="right")

    out = FIG_DIR / "zhang16_arch.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----- Figure 2: Zhang 2017 — interactive CNN with hints branch ---------------
def zhang17():
    fig, ax = plt.subplots(figsize=(11.5, 4.4), dpi=160)
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.set_title("Zhang 2017 — interactive CNN: luminance backbone + sparse-hint branch "
                 "(automatic mode = empty hints)",
                 fontsize=10, pad=6)

    # main backbone — U-Net-ish with skip connections
    main = [
        (0.30, 0.9, IO,  "L\n1×256×256"),
        (1.30, 0.9, ENC, "conv1\n64"),
        (2.30, 0.9, ENC, "conv2\n128"),
        (3.30, 0.9, ENC, "conv3\n256"),
        (4.30, 0.9, CTX, "conv4-7\n512, dilated"),
        (5.50, 0.9, DEC, "conv8\nup, 256"),
        (6.50, 0.9, DEC, "conv9\nup, 128"),
        (7.50, 0.9, DEC, "conv10\nup, 128"),
        (8.50, 0.9, HEAD, "1×1 conv\nab regression"),
        (9.65, 1.45, IO, "ab → Lab\n→ RGB"),
    ]
    yb, hb = 2.55, 0.9
    for x, w, col, txt in main:
        _box(ax, x, yb, w, hb, txt, col, fs=8)
    for i in range(len(main) - 1):
        x0 = main[i][0] + main[i][1]
        x1 = main[i + 1][0]
        _arrow(ax, x0, yb + hb / 2, x1, yb + hb / 2)

    _label(ax, 2.8, 2.35, "encoder", color="#2e7d32", fs=8, fw="bold")
    _label(ax, 4.85, 2.35, "dilated bottleneck", color="#a05a00", fs=8, fw="bold")
    _label(ax, 7.0, 2.35, "decoder (upsampling)", color="#a02a2a", fs=8, fw="bold")

    # symmetric skip connections (encoder → decoder)
    skips = [(1.30 + 0.45, 3.30 + 0.45, "skip"),
             (2.30 + 0.45, 5.50 + 0.45, "skip"),
             (3.30 + 0.45, 6.50 + 0.45, "skip")]
    for xa, xb_, lbl in skips:
        arr = FancyArrowPatch((xa, yb + hb), (xb_, yb + hb),
                              arrowstyle="-|>", mutation_scale=8,
                              connectionstyle="arc3,rad=-0.55",
                              linewidth=0.7, color="#8a55c4",
                              linestyle="--")
        ax.add_patch(arr)
    _label(ax, 5.0, 3.80, "U-shape skip connections", color="#8a55c4",
           fs=8, fw="bold")

    # hints branch (below)
    hints = [
        (0.30, 0.9, HINT, "User\nhints\n(a, b, mask)\n3×256×256"),
        (1.30, 0.9, ENC, "hint conv\n128"),
        (2.30, 0.9, ENC, "hint conv\n128"),
        (3.30, 0.9, ENC, "fuse →\nmain net"),
    ]
    yh, hh = 0.50, 0.9
    for x, w, col, txt in hints:
        _box(ax, x, yh, w, hh, txt, col, fs=8)
    for i in range(len(hints) - 1):
        x0 = hints[i][0] + hints[i][1]
        x1 = hints[i + 1][0]
        _arrow(ax, x0, yh + hh / 2, x1, yh + hh / 2)

    # fuse arrow from hint branch up to conv4 of main
    _arrow(ax, 3.30 + 0.9, yh + hh, 4.75, yb, color="#a05a00", lw=1.0)
    _label(ax, 4.4, 1.65, "concatenate with bottleneck", color="#a05a00",
           fs=7.8, ha="center")

    # auto-mode banner
    _label(ax, 0.75, 0.22, "automatic mode: hints = 0 (zero a,b + zero mask)",
           color="#555", fs=8, ha="left")

    out = FIG_DIR / "zhang17_arch.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----- Figure 3: DeOldify — U-Net + critic + NoGAN ----------------------------
def deoldify():
    fig, ax = plt.subplots(figsize=(13.0, 4.8), dpi=160)
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 4.8); ax.axis("off")
    ax.set_title("DeOldify — U-Net generator (ResNet-34 encoder + self-attention) "
                 "trained with the NoGAN recipe",
                 fontsize=10, pad=6)

    # U-Net generator block (wider)
    g = [
        (0.30, 1.30, IO,  "Grayscale\ninput\n1×H×W"),
        (1.75, 1.50, ENC, "ResNet-34\nencoder\n(ImageNet init)"),
        (3.40, 1.40, CTX, "Self-attention\nbottleneck"),
        (4.95, 1.40, DEC, "Decoder\n(upsample\n+ skips)"),
        (6.50, 1.40, HEAD, "Output head\n→ ab"),
        (8.05, 1.40, IO, "Colorized\nRGB"),
    ]
    yg, hg = 2.85, 1.20
    for x, w, col, txt in g:
        _box(ax, x, yg, w, hg, txt, col, fs=8.2)
    for i in range(len(g) - 1):
        x0 = g[i][0] + g[i][1]
        x1 = g[i + 1][0]
        _arrow(ax, x0, yg + hg / 2, x1, yg + hg / 2)

    # U-Net skip arrows (encoder->decoder)
    arr = FancyArrowPatch((1.75 + 1.50 / 2, yg + hg),
                          (4.95 + 1.40 / 2, yg + hg),
                          arrowstyle="-|>", mutation_scale=8,
                          connectionstyle="arc3,rad=-0.55",
                          linewidth=0.7, color="#8a55c4", linestyle="--")
    ax.add_patch(arr)
    _label(ax, 0.30, 4.35, "Generator G  (U-Net w/ skip + self-attention)",
           color="#2e7d32", fs=9.5, fw="bold", ha="left")
    _label(ax, 4.0, 4.55, "skip connections", color="#8a55c4", fs=8.5, fw="bold")

    # Critic (discriminator)
    _box(ax, 10.20, yg, 2.0, hg,
         "Critic\n(spectral-norm\nPatchGAN)", CRITIC, fs=8.2)
    _label(ax, 10.20, 4.35, "Critic D", color="#a05a00", fs=9.5, fw="bold", ha="left")
    # Generator output -> Critic
    _arrow(ax, 8.05 + 1.40, yg + hg / 2, 10.20, yg + hg / 2)
    _label(ax, 9.70, yg + hg / 2 + 0.22, "fake / real", fs=8, color="#555")

    # NoGAN training stages
    _label(ax, 0.30, 2.20, "NoGAN training recipe",
           color="#444", fs=9.5, fw="bold", ha="left")
    stages = [
        (0.30, 2.80, "Stage 1\nPretrain G  alone\nwith perceptual loss\n(no D, no adversarial)"),
        (3.30, 2.80, "Stage 2\nPretrain D  alone on\nG's frozen outputs vs. real"),
        (6.30, 2.80, "Stage 3\nShort GAN fine-tune\n($\\sim$1-3% of epochs);\nstop before collapse"),
        (9.30, 2.80, "Result\nDeep colour + stable training\n+ no mode collapse"),
    ]
    for x, w, txt in stages:
        _box(ax, x, 0.30, w, 1.55, txt, "#f6f1e3", fs=8.2)
    # arrows between stages
    for i in range(len(stages) - 1):
        x0 = stages[i][0] + stages[i][1]
        x1 = stages[i + 1][0]
        _arrow(ax, x0, 0.30 + 1.55 / 2, x1, 0.30 + 1.55 / 2)

    out = FIG_DIR / "deoldify_arch.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ----- Figure 4: ControlNet + SD 2.1 ------------------------------------------
def controlnet():
    fig, ax = plt.subplots(figsize=(13.0, 5.2), dpi=160)
    ax.set_xlim(0, 15.0); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.set_title("ControlNet + Stable Diffusion 2.1 — frozen text-to-image U-Net "
                 "+ trainable encoder copy with zero-conv adapters",
                 fontsize=10, pad=6)

    # Frozen SD U-Net (top row) — wider boxes, more spacing
    fz = [
        (0.20, 1.10, IO,     "noisy latent\n$z_t$"),
        (1.55, 1.30, FROZEN, "SD enc\nblock 1\n(frozen)"),
        (3.10, 1.30, FROZEN, "SD enc\nblock 2\n(frozen)"),
        (4.65, 1.30, FROZEN, "SD mid\n(frozen)"),
        (6.20, 1.30, FROZEN, "SD dec\nblock 2\n(frozen)"),
        (7.75, 1.30, FROZEN, "SD dec\nblock 1\n(frozen)"),
        (9.30, 1.30, HEAD,   "ε-pred\nhead"),
        (10.85, 1.30, IO,    "denoised\n$z_{t-1}$"),
    ]
    yf, hf = 3.45, 1.10
    for x, w, col, txt in fz:
        _box(ax, x, yf, w, hf, txt, col, fs=8)
    for i in range(len(fz) - 1):
        x0 = fz[i][0] + fz[i][1]
        x1 = fz[i + 1][0]
        _arrow(ax, x0, yf + hf / 2, x1, yf + hf / 2)
    _label(ax, 4.5, 4.85, "Frozen pretrained backbone (Stable Diffusion 2.1)",
           color="#666", fs=9.5, fw="bold")

    # Trainable ControlNet copy (middle row) — aligned under the encoder + mid
    cn = [
        (1.55, 1.30, TRAIN, "copy of\nenc block 1\n(trainable)"),
        (3.10, 1.30, TRAIN, "copy of\nenc block 2\n(trainable)"),
        (4.65, 1.30, TRAIN, "copy of\nSD mid\n(trainable)"),
    ]
    yc, hc = 1.70, 1.10
    for x, w, col, txt in cn:
        _box(ax, x, yc, w, hc, txt, col, fs=8)
    for i in range(len(cn) - 1):
        x0 = cn[i][0] + cn[i][1]
        x1 = cn[i + 1][0]
        _arrow(ax, x0, yc + hc / 2, x1, yc + hc / 2)
    _label(ax, 6.50, 1.50, "Trainable ControlNet copy",
           color="#1e4a8c", fs=9.5, fw="bold", ha="left")

    # Zero-conv adapters — feed each trainable block's output into the matching
    # decoder block of the frozen backbone (skip-style).
    pairs = [
        (1.55 + 1.30 / 2, 7.75 + 1.30 / 2),  # cn enc1  -> SD dec1
        (3.10 + 1.30 / 2, 6.20 + 1.30 / 2),  # cn enc2  -> SD dec2
        (4.65 + 1.30 / 2, 4.65 + 1.30 / 2),  # cn mid   -> SD mid
    ]
    for cn_cx, sd_cx in pairs:
        # zero-conv box just above the middle row
        zx = cn_cx - 0.32
        _box(ax, zx, yc + hc + 0.05, 0.64, 0.32, "0-conv", ZEROCONV, fs=7.2)
        # arrow from zero-conv up to the target SD block (bottom edge)
        _arrow(ax, cn_cx, yc + hc + 0.05 + 0.32,
               sd_cx, yf, color="#a02a2a", lw=0.9)
    _label(ax, 7.50, 3.10,
           "zero-conv (init w = 0): backbone unchanged at t=0,\n"
           "so SD's priors are preserved while the copy learns control.",
           color="#a02a2a", fs=8, ha="left")

    # Conditioning input (bottom-left)
    _box(ax, 0.20, 0.20, 1.10, 1.10,
         "Grayscale\nL (cond c)\n→ encoded", "#e7e0ff", fs=8)
    _arrow(ax, 0.20 + 1.10, 0.75, 1.55, yc + hc / 2, color="#444")
    _label(ax, 0.20, 1.50, "condition", color="#444", fs=8, ha="left")

    # Iterative denoising banner
    _label(ax, 12.20, 4.0, "× 20-50 DDIM\nsteps", color="#555",
           fs=9, ha="left")

    out = FIG_DIR / "controlnet_arch.png"
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for name, fn in [("zhang16", zhang16),
                     ("zhang17", zhang17),
                     ("deoldify", deoldify),
                     ("controlnet", controlnet)]:
        path = fn()
        print(f"wrote {path}  ({path.stat().st_size/1024:.1f} KB)")
