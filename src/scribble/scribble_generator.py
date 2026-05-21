"""
AutoScribbleGenerator — automatic scribble image creation for ScribbleColorizer.

Generates a "scribble image" from a ground-truth color image by sampling a
fraction of pixels (the scribbles) and leaving the rest as gray.

Usage:
    from src.scribble_based.scribble_generator import AutoScribbleGenerator

    gen = AutoScribbleGenerator(seed=42)

    # Uniform random sampling, 5% of pixels kept as color
    scribble_bgr = gen.generate(color_bgr, gray_bgr, density=0.05, mode='uniform')

    # Edge-guided sampling — biases toward region boundaries
    scribble_bgr = gen.generate(color_bgr, gray_bgr, density=0.05, mode='edge')
"""

import numpy as np
import cv2


class AutoScribbleGenerator:
    """Generate scribble images from a reference color image.

    A scribble image has the same shape as the input.  Scribbled pixels
    carry the actual color from ``color_bgr``; non-scribbled pixels carry
    the gray value (R == G == B) from ``gray_bgr``.  The ScribbleColorizer
    identifies scribbled pixels as those where R ≠ G or G ≠ B.

    Args:
        seed: Random seed for reproducibility (default 0).
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        color_bgr: np.ndarray,
        gray_bgr: np.ndarray,
        density: float = 0.05,
        mode: str = "uniform",
        edge_blur_ksize: int = 5,
        edge_low_thresh: int = 30,
        edge_high_thresh: int = 100,
        edge_dilate_iters: int = 3,
        min_scribbles: int = 10,
    ) -> np.ndarray:
        """Build a scribble image from ``color_bgr``.

        Args:
            color_bgr:        (H, W, 3) BGR uint8 ground-truth color image.
            gray_bgr:         (H, W, 3) BGR uint8 grayscale image
                              (all channels identical).  This is used as the
                              background of the scribble image.
            density:          Fraction of pixels to keep as scribbles, in
                              (0, 1].  E.g. 0.05 keeps ~5% of pixels colored.
            mode:             Sampling strategy:
                              - 'uniform'  — pixels sampled uniformly at random.
                              - 'edge'     — pixels near Canny edges are sampled
                                             with higher probability, giving
                                             better color propagation across
                                             region boundaries.
            edge_blur_ksize:  Kernel size for Gaussian blur before Canny edge
                              detection (only used in 'edge' mode).
            edge_low_thresh:  Lower hysteresis threshold for Canny.
            edge_high_thresh: Upper hysteresis threshold for Canny.
            edge_dilate_iters: Number of dilation iterations applied to the
                              edge map to widen the high-probability zone.
            min_scribbles:    Minimum number of scribbled pixels guaranteed
                              regardless of density (safety floor).

        Returns:
            scribble_bgr: (H, W, 3) BGR uint8 scribble image.
                          Scribbled pixels have color from ``color_bgr``;
                          all other pixels equal ``gray_bgr``.
        """
        if not 0.0 < density <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {density}")
        if mode not in ("uniform", "edge"):
            raise ValueError(f"mode must be 'uniform' or 'edge', got '{mode}'")

        H, W = color_bgr.shape[:2]
        N = H * W
        n_keep = max(min_scribbles, int(round(density * N)))
        n_keep = min(n_keep, N)

        # Build sampling probability map
        if mode == "uniform":
            prob = np.ones(N, dtype=np.float64)
        else:  # 'edge'
            prob = self._edge_probability_map(
                gray_bgr,
                blur_ksize=edge_blur_ksize,
                low_thresh=edge_low_thresh,
                high_thresh=edge_high_thresh,
                dilate_iters=edge_dilate_iters,
            )

        prob = prob / prob.sum()
        chosen = self._rng.choice(N, size=n_keep, replace=False, p=prob)

        # Build output: start from gray, overwrite chosen pixels with color
        scribble = gray_bgr.copy()
        rows, cols = np.unravel_index(chosen, (H, W))
        scribble[rows, cols] = color_bgr[rows, cols]

        # Edge case: if a "colored" pixel happens to be perfectly gray in the
        # color image (R==G==B), the ScribbleColorizer will not recognize it
        # as a constraint.  We can't fix this without knowing the solver's
        # internal representation, so we leave it — the solver will propagate
        # color from neighboring true scribbles.

        return scribble

    def generate_batch(
        self,
        color_bgr: np.ndarray,
        gray_bgr: np.ndarray,
        densities,
        mode: str = "uniform",
        **kwargs,
    ) -> list:
        """Convenience wrapper: generate multiple scribble images at once.

        Args:
            color_bgr: ground-truth color image.
            gray_bgr:  grayscale image.
            densities: iterable of float density values.
            mode:      sampling mode (same for all densities).
            **kwargs:  forwarded to :meth:`generate`.

        Returns:
            List of ``(density, scribble_bgr)`` tuples.
        """
        results = []
        for d in densities:
            scr = self.generate(color_bgr, gray_bgr, density=d, mode=mode, **kwargs)
            results.append((d, scr))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_probability_map(
        gray_bgr: np.ndarray,
        blur_ksize: int,
        low_thresh: int,
        high_thresh: int,
        dilate_iters: int,
        edge_weight: float = 10.0,
    ) -> np.ndarray:
        """Return a flat probability array biased toward edge regions.

        Non-edge pixels receive weight 1.0; edge (and dilated-edge) pixels
        receive weight ``edge_weight``.

        Args:
            gray_bgr:     (H, W, 3) grayscale BGR image.
            blur_ksize:   Gaussian blur kernel size (must be odd).
            low_thresh:   Canny lower threshold.
            high_thresh:  Canny upper threshold.
            dilate_iters: Dilation iterations on edge map.
            edge_weight:  Weight multiplier for edge pixels.

        Returns:
            flat (H*W,) float64 array of unnormalized weights.
        """
        gray_1ch = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)

        # Ensure blur kernel size is odd
        ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        blurred = cv2.GaussianBlur(gray_1ch, (ksize, ksize), 0)

        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        if dilate_iters > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=dilate_iters)

        # Convert binary edge map to weight map
        prob = np.ones(gray_1ch.shape, dtype=np.float64)
        prob[edges > 0] = edge_weight
        return prob.ravel()