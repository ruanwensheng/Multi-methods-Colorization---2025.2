import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from skimage import color as skcolor


def bgr_to_lab(bgr_image):
    """Convert BGR uint8 image to CIE Lab.

    Returns:
        L:  (H, W) float64, range [0, 100]
        ab: (H, W, 2) float64, range ~[-110, 110]
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    lab = skcolor.rgb2lab(rgb.astype(np.float64) / 255.0)
    return lab[:, :, 0], lab[:, :, 1:]


def lab_to_bgr(L, ab):
    """Convert CIE Lab channels to BGR uint8 image."""
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float64)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab
    rgb = np.clip(skcolor.lab2rgb(lab) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def local_std(L_channel, window_size=5):
    """Per-pixel local standard deviation of L in an NxN sliding window.

    Uses Var(x) = E[x^2] - E[x]^2 with uniform_filter for O(1) per-pixel cost.

    Args:
        L_channel:   (H, W) float array
        window_size: int, NxN window
    Returns:
        (H, W) float, local std >= 0
    """
    L = L_channel.astype(np.float64)
    mean = uniform_filter(L, size=window_size)
    mean_sq = uniform_filter(L ** 2, size=window_size)
    return np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
