"""
Utility functions for deep learning colorization.

Provides color space conversions, metric computation, visualization,
and configuration loading.
"""

import time
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from skimage import color as skcolor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_config(path="configs/config.yaml"):
    """Load YAML config file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rgb_to_lab(rgb_image):
    """Convert RGB image [0,255] uint8 to Lab.

    Args:
        rgb_image: np.ndarray (H, W, 3) uint8 RGB image.

    Returns:
        L: np.ndarray (H, W) float64, range [0, 100].
        ab: np.ndarray (H, W, 2) float64, range ~ [-110, 110].
    """
    rgb_float = rgb_image.astype(np.float64) / 255.0
    lab = skcolor.rgb2lab(rgb_float)
    L = lab[:, :, 0]
    ab = lab[:, :, 1:]
    return L, ab


def lab_to_rgb(L, ab):
    """Convert Lab channels back to RGB image [0,255] uint8.

    Args:
        L: np.ndarray (H, W) float64, range [0, 100].
        ab: np.ndarray (H, W, 2) float64, range ~ [-110, 110].

    Returns:
        np.ndarray (H, W, 3) uint8 RGB image.
    """
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float64)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab
    rgb = skcolor.lab2rgb(lab)  # returns [0, 1] float
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def bgr_to_lab(bgr_image):
    """Convert BGR image (OpenCV format) to Lab channels."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_to_lab(rgb)


def lab_to_bgr(L, ab):
    """Convert Lab channels to BGR image (OpenCV format)."""
    rgb = lab_to_rgb(L, ab)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def normalize_L(L):
    """Normalize L channel from [0, 100] to [-1, 1]."""
    return L / 50.0 - 1.0


def denormalize_L(L_norm):
    """Denormalize L channel from [-1, 1] to [0, 100]."""
    return (L_norm + 1.0) * 50.0


def normalize_ab(ab):
    """Normalize ab channels from [-110, 110] to [-1, 1]."""
    return ab / 110.0


def denormalize_ab(ab_norm):
    """Denormalize ab channels from [-1, 1] to [-110, 110]."""
    return ab_norm * 110.0


def compute_metrics(predicted_rgb, ground_truth_rgb):
    """Compute image quality metrics between predicted and ground truth.

    Args:
        predicted_rgb: np.ndarray (H, W, 3) uint8 RGB image.
        ground_truth_rgb: np.ndarray (H, W, 3) uint8 RGB image.

    Returns:
        dict with keys: "psnr", "ssim".
    """
    metrics = {}

    # PSNR
    metrics["psnr"] = float(peak_signal_noise_ratio(
        ground_truth_rgb, predicted_rgb, data_range=255
    ))

    # SSIM
    metrics["ssim"] = float(structural_similarity(
        ground_truth_rgb, predicted_rgb, channel_axis=2, data_range=255
    ))

    return metrics


def visualize_result(gray, predicted, ground_truth=None, save_path=None, title=None):
    """Visualize colorization result with side-by-side panels.

    Args:
        gray: np.ndarray (H, W) or (H, W, 3) grayscale image.
        predicted: np.ndarray (H, W, 3) RGB predicted colorization.
        ground_truth: Optional np.ndarray (H, W, 3) RGB ground truth.
        save_path: Optional path to save the figure.
        title: Optional figure title.
    """
    n_panels = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Grayscale input
    if gray.ndim == 3:
        axes[0].imshow(gray)
    else:
        axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Grayscale Input")
    axes[0].axis("off")

    # Predicted colorization
    axes[1].imshow(predicted)
    axes[1].set_title("Predicted")
    axes[1].axis("off")

    # Ground truth (if available)
    if ground_truth is not None:
        axes[2].imshow(ground_truth)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def visualize_comparison(results_dict, save_path=None):
    """Visualize comparison across multiple colorization methods.

    Args:
        results_dict: dict mapping method_name -> (gray, predicted_rgb, metrics_dict).
        save_path: Optional path to save the figure.
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))

    # Show grayscale from first method
    gray, _, _ = results_dict[methods[0]]
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Input")
    axes[0].axis("off")

    for i, method in enumerate(methods):
        _, predicted, metrics = results_dict[method]
        axes[i + 1].imshow(predicted)
        metric_str = ""
        if metrics:
            parts = []
            if "psnr" in metrics:
                parts.append(f"PSNR: {metrics['psnr']:.1f}")
            if "ssim" in metrics:
                parts.append(f"SSIM: {metrics['ssim']:.3f}")
            metric_str = "\n".join(parts)
        axes[i + 1].set_title(f"{method}\n{metric_str}", fontsize=9)
        axes[i + 1].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def get_device(cfg=None):
    """Get the appropriate torch device based on config or auto-detection."""
    import torch

    if cfg and cfg.get("device", "auto") != "auto":
        return torch.device(cfg["device"])

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
