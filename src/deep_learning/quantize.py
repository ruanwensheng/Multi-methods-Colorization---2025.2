"""
Ab color space quantization for Zhang et al. 2016.

Quantizes the CIE Lab ab color space into in-gamut bins using a 10x10 grid.
Provides encoding (ab -> bin index) and decoding (probability -> ab) operations.

Reference: "Colorful Image Colorization", Zhang et al., ECCV 2016, Section 2.2
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter


def _compute_in_gamut_ab_bins(grid_size=10):
    """Compute in-gamut ab bin centers.

    Creates a grid over the ab space [-110, 110] with the given grid size,
    then filters to keep only bins that map to valid sRGB colors.
    Tests gamut across all L values [0, 100].

    Returns:
        np.ndarray: Shape (N, 2) array of ab bin centers.
    """
    a_range = np.arange(-110, 120, grid_size)
    b_range = np.arange(-110, 120, grid_size)

    ab_grid = np.array(np.meshgrid(a_range, b_range)).T.reshape(-1, 2)

    # D65 illuminant reference white
    X_n, Y_n, Z_n = 0.95047, 1.00000, 1.08883
    delta = 6.0 / 29.0

    # XYZ to linear sRGB matrix
    M = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570],
    ])

    def _lab_to_linear_rgb(L, a, b):
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        def f_inv(t):
            return t ** 3 if t > delta else 3 * delta ** 2 * (t - 4.0 / 29.0)

        X = X_n * f_inv(fx)
        Y = Y_n * f_inv(fy)
        Z = Z_n * f_inv(fz)

        return M @ np.array([X, Y, Z])

    in_gamut = []
    for ab in ab_grid:
        # Check if this ab pair is in sRGB gamut at ANY L value
        for L_val in range(0, 101, 5):
            rgb_lin = _lab_to_linear_rgb(float(L_val), ab[0], ab[1])
            if np.all(rgb_lin >= -0.01) and np.all(rgb_lin <= 1.01):
                in_gamut.append(ab)
                break

    return np.array(in_gamut, dtype=np.float64)


def _load_or_compute_ab_bins(grid_size=10):
    """Load pre-computed ab bins from file, or compute them.

    If pts_in_hull.npy exists in the module directory, loads from there.
    Otherwise computes from gamut analysis.

    Returns:
        np.ndarray: Shape (N, 2) array of ab bin centers.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    pts_path = os.path.join(module_dir, "pts_in_hull.npy")

    if os.path.exists(pts_path):
        return np.load(pts_path)

    ab_bins = _compute_in_gamut_ab_bins(grid_size)
    np.save(pts_path, ab_bins)
    return ab_bins


class ABQuantizer:
    """Handles quantization of ab color values to discrete bins.

    The number of bins depends on the gamut computation method:
    - Default computation: ~226 bins (sRGB gamut check)
    - Official Zhang 2016 pts_in_hull.npy: 313 bins

    The model's num_classes is set to match automatically.

    Args:
        grid_size: Size of each bin in ab space (default 10).
        sigma: Gaussian smoothing sigma for class rebalancing weights.
        lmbda: Mixing weight between empirical and uniform distributions.
    """

    def __init__(self, grid_size=10, sigma=5.0, lmbda=0.5):
        from scipy.spatial import cKDTree

        self.grid_size = grid_size
        self.sigma = sigma
        self.lmbda = lmbda
        self.ab_bins = _load_or_compute_ab_bins(grid_size)
        self.num_bins = len(self.ab_bins)
        self._tree = cKDTree(self.ab_bins)

    def encode(self, ab):
        """Quantize ab values to nearest bin indices using KD-tree.

        Args:
            ab: np.ndarray of shape (..., 2) with ab values.

        Returns:
            np.ndarray of int indices with shape (...,).
        """
        original_shape = ab.shape[:-1]
        ab_flat = ab.reshape(-1, 2)
        _, indices = self._tree.query(ab_flat)
        return indices.reshape(original_shape)

    def encode_soft(self, ab, n_neighbors=5, sigma=5.0):
        """Soft-encode ab values as weighted distribution over nearby bins.

        Args:
            ab: np.ndarray of shape (..., 2) with ab values.
            n_neighbors: Number of nearest bins to assign weight to.
            sigma: Gaussian kernel sigma for soft assignment.

        Returns:
            np.ndarray of shape (..., num_bins) with soft assignments.
        """
        ab_flat = ab.reshape(-1, 2)
        N = ab_flat.shape[0]

        dists = np.sum((ab_flat[:, np.newaxis, :] - self.ab_bins[np.newaxis, :, :]) ** 2, axis=2)

        top_k_idx = np.argpartition(dists, n_neighbors, axis=1)[:, :n_neighbors]
        top_k_dists = np.take_along_axis(dists, top_k_idx, axis=1)

        weights = np.exp(-top_k_dists / (2 * sigma ** 2))
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        soft = np.zeros((N, self.num_bins), dtype=np.float32)
        for i in range(N):
            soft[i, top_k_idx[i]] = weights[i]

        return soft.reshape(ab.shape[:-1] + (self.num_bins,))

    def decode(self, probs, temperature=0.38):
        """Decode probability distribution over bins to ab values using annealed-mean.

        Implements Eq. 5 from Zhang et al. 2016:
            f_T(z) = exp(log(z) / T) / sum(exp(log(z_q) / T))

        Args:
            probs: np.ndarray of shape (..., num_bins) with probabilities.
            temperature: Annealing temperature (0 = mode, 1 = mean).

        Returns:
            np.ndarray of shape (..., 2) with decoded ab values.
        """
        log_probs = np.log(probs + 1e-8)
        annealed = np.exp(log_probs / temperature)
        annealed = annealed / (annealed.sum(axis=-1, keepdims=True) + 1e-8)

        ab = annealed @ self.ab_bins
        return ab

    def compute_class_weights(self, empirical_dist=None):
        """Compute class-rebalancing weights (Zhang 2016, Eq. 4).

        w ~ ((1-lambda) * p_tilde + lambda/Q) ^ -1

        Args:
            empirical_dist: Optional empirical distribution over bins.
                If None, returns uniform weights.

        Returns:
            np.ndarray of shape (num_bins,) with per-class weights.
        """
        Q = self.num_bins

        if empirical_dist is None:
            return np.ones(Q, dtype=np.float32)

        p_tilde = gaussian_filter(empirical_dist.astype(np.float64), sigma=self.sigma)
        p_tilde = p_tilde / (p_tilde.sum() + 1e-8)

        w = ((1 - self.lmbda) * p_tilde + self.lmbda / Q) ** (-1)
        w = w / (np.sum(p_tilde * w) + 1e-8)

        return w.astype(np.float32)


def compute_empirical_distribution(dataset, quantizer, max_samples=1000, cache_path=None):
    """Estimate the empirical distribution of ab values across a dataset.

    Walks up to ``max_samples`` images, accumulates per-bin pixel counts using
    the quantizer's nearest-neighbor encoder, and returns a normalized
    distribution over bins. Results are cached to ``cache_path`` so the
    expensive walk runs at most once across training sessions.

    Without this, ``ClassRebalancedCELoss`` falls back to uniform weights and
    the model has no incentive to predict saturated colors — the classic
    "Zhang16 outputs sepia" failure mode.

    Args:
        dataset: Indexable dataset yielding dicts with key ``"ab"`` (a tensor
            shaped (2, H, W) with values normalized to roughly [-1, 1]).
        quantizer: ABQuantizer instance.
        max_samples: Cap on the number of images visited. Default 1000.
        cache_path: Optional .npy path. If it exists, it's loaded and returned
            without recomputing. If provided and missing, the computed
            distribution is written there.

    Returns:
        np.ndarray of shape (quantizer.num_bins,), summing to ~1.0.
    """
    if cache_path is not None and os.path.exists(cache_path):
        return np.load(cache_path)

    counts = np.zeros(quantizer.num_bins, dtype=np.float64)
    n = min(len(dataset), int(max_samples))
    for i in range(n):
        sample = dataset[i]
        ab_norm = sample["ab"].detach().cpu().numpy()  # (2, H, W) in ~[-1, 1]
        ab = ab_norm.transpose(1, 2, 0) * 110.0        # (H, W, 2) in [-110, 110]
        idx = quantizer.encode(ab).reshape(-1)
        binc = np.bincount(idx, minlength=quantizer.num_bins).astype(np.float64)
        counts += binc

    total = counts.sum()
    dist = counts / total if total > 0 else np.full(quantizer.num_bins, 1.0 / quantizer.num_bins)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, dist)
    return dist
