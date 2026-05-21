import time
import numpy as np
import cv2
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter

from ..base_colorizer import BaseColorizer


class ScribbleColorizer(BaseColorizer):
    """Scribble-based colorization via intensity-weighted optimization.

    Implements Levin, Lischinski, Weiss (SIGGRAPH 2004):
    "Colorization using Optimization".

    Core idea: neighboring pixels with similar luminance should share
    similar chrominance. The user annotates sparse color scribbles; the
    algorithm propagates them by minimizing a quadratic cost function
    where the affinity between pixels decays as their luminance difference
    grows.

    Works in YUV space (internally converts BGR → YUV → BGR) but the
    affinity is computed in CIE Lab for perceptual uniformity.
    """

    def __init__(self, cfg=None):
        params = (cfg or {}).get("scribble_based", {})
        # NxN window for local mean/variance in affinity (equation 3 in paper)
        self.window_size: int = params.get("window_size", 3)
        # Small epsilon to avoid division-by-zero in low-contrast regions
        self.epsilon: float = params.get("epsilon", 1e-6)

    # ------------------------------------------------------------------
    # Public API (satisfies BaseColorizer contract)
    # ------------------------------------------------------------------

    def colorize(
        self,
        gray_image: np.ndarray,
        scribble_image: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """Propagate color scribbles across a grayscale image.

        Args:
            gray_image:     (H, W, 3) BGR uint8 grayscale target **or**
                            (H, W) uint8 single-channel grayscale.
            scribble_image: (H, W, 3) BGR uint8 image where scribbled
                            pixels have color and un-scribbled pixels match
                            gray_image (i.e. R == G == B for un-scribbled).
                            Pass the same array as gray_image to run with
                            no scribbles (no-op result).

        Returns:
            colorized_bgr: (H, W, 3) BGR uint8
            info_dict:     {'method', 'time_seconds', 'image_size', 'n_scribbled'}
        """
        start = time.time()

        # ---- normalise inputs to BGR uint8 --------------------------
        if gray_image.ndim == 2:
            gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        else:
            # if 3
            gray_bgr = gray_image.copy()
        

        # size of pic
        H, W = gray_bgr.shape[:2]
        N = H * W

        # ---- luminance channel (float, 0-1) in YUV color space
        gray_yuv = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2YUV).astype(np.float64) / 255.0
        # Y is the 1st dimension in YUV color space, so we get the 0 
        Y = gray_yuv[:, :, 0]          # (H, W)

        # ---- detect scribbled pixels --------------------------------
        # A pixel is "scribbled" when its RGB values in scribble_image are
        # not all equal (i.e. it carries actual color, not just gray).
        scr_bgr = scribble_image.astype(np.int32)

        # ma trận true false HWW
        is_scribbled = (
            (scr_bgr[:, :, 0] != scr_bgr[:, :, 1]) |
            (scr_bgr[:, :, 1] != scr_bgr[:, :, 2])
        )                               # (H, W) bool
        n_scribbled = int(is_scribbled.sum())

        # ---- extract scribble UV values (float, 0-1) ----------------
        scr_yuv = cv2.cvtColor(scribble_image, cv2.COLOR_BGR2YUV).astype(np.float64) / 255.0
        U_scr = scr_yuv[:, :, 1]       # (H, W)
        V_scr = scr_yuv[:, :, 2]       # (H, W)

        # ---- build affinity weights ---------------------------------
        # ma trận trọng số có kích cỡ NN, với N là số pixel của ảnh gốc
        # ma trận ghi lại mức độ kết nối giữa mọi pixel, dựa trên ảnh đen trắng gốc (Y)
        # trọng số càng gần 0 thì càng ko có khả năng lan sang đó
        W_mat = self._build_weight_matrix(Y, H, W)   # sparse (N, N)

        # ---- solve for U and V channels independently ---------------
        U_out = self._solve_channel(W_mat, Y, U_scr, is_scribbled, H, W)
        V_out = self._solve_channel(W_mat, Y, V_scr, is_scribbled, H, W)

        # ---- reconstruct BGR uint8 ----------------------------------
        yuv_out = np.stack([Y, U_out, V_out], axis=-1)
        yuv_u8 = np.clip(yuv_out * 255, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(yuv_u8, cv2.COLOR_YUV2BGR)

        elapsed = time.time() - start
        return result_bgr, {
            "method": "scribble_based",
            "time_seconds": elapsed,
            "image_size": (H, W),
            "n_scribbled": n_scribbled,
        }



    def _build_weight_matrix(
        self, Y: np.ndarray, H: int, W: int
    ) -> csr_matrix:
        """Build the N×N sparse affinity matrix W.

        For each pixel r, its 4-connected neighbors s share a weight:
            w_rs ∝ 1 + (1/σ_r²)(Y(r) − µ_r)(Y(s) − µ_r)   [eq. 3]

        The weight matrix is row-normalised so each row sums to 1,
        matching the form required by the cost function J(U).

        Args:
            Y: (H, W) float64 luminance in [0, 1]
            H, W: image dimensions

        Returns:
            csr_matrix of shape (H*W, H*W)
        """
        ws = self.window_size
        N = H * W

        # Local mean and variance via uniform_filter (O(1) per pixel)
        mu = uniform_filter(Y, size=ws)
        mu_sq = uniform_filter(Y ** 2, size=ws)
        sigma2 = np.maximum(mu_sq - mu ** 2, self.epsilon)  # (H, W)

        mat = lil_matrix((N, N), dtype=np.float64)

        # 4-connected neighbourhood offsets: right, down (left/up covered
        # by symmetry when we write both directions).
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dr, dc in directions:
            # Compute valid pixel index ranges
            r0, r1 = max(0, -dr), min(H, H - dr)
            c0, c1 = max(0, -dc), min(W, W - dc)

            # Source pixels and their neighbours
            ys, xs = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
            yn, xn = ys + dr, xs + dc

            idx_s = (ys * W + xs).ravel()
            idx_n = (yn * W + xn).ravel()

            # Correlation-based affinity (eq. 3):
            #   w = 1 + (Y(r)-µ_r)(Y(s)-µ_r) / σ_r²
            w = 1.0 + (
                (Y[ys, xs] - mu[ys, xs]) *
                (Y[yn, xn] - mu[ys, xs]) /
                sigma2[ys, xs]
            )
            w = w.ravel()

            # Accumulate (we will row-normalise afterwards)
            for i, (s, n, wval) in enumerate(zip(idx_s, idx_n, w)):
                mat[s, n] += wval

        mat = mat.tocsr()

        # Row-normalise: each row of W sums to 1
        row_sums = np.asarray(mat.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0          # guard against isolated pixels
        from scipy.sparse import diags
        D_inv = diags(1.0 / row_sums)
        return D_inv @ mat

    def _solve_channel(
        self,
        W_mat: csr_matrix,
        Y: np.ndarray,
        C_scr: np.ndarray,
        is_scribbled: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """Solve the linear system for one chrominance channel.

        The cost function J(C) = Σ_r (C(r) - Σ_{s∈N(r)} w_rs · C(s))²
        subject to C(r) = C_scr(r) for scribbled pixels.

        Rearranging: (I - W) C = 0, with hard constraints at scribbles.
        We enforce constraints by substituting known values into the RHS.

        Args:
            W_mat:        row-normalised affinity matrix (N, N)
            Y:            (H, W) luminance (used only for shape/context here)
            C_scr:        (H, W) scribbled chrominance values
            is_scribbled: (H, W) bool mask of constrained pixels
            H, W:         image dimensions

        Returns:
            (H, W) float64 solved chrominance channel in [0, 1]
        """
        from scipy.sparse import eye as speye

        N = H * W
        scr_flat = is_scribbled.ravel()          # (N,) bool
        c_flat = C_scr.ravel()                   # (N,) float

        # A = I - W  (the Laplacian-like matrix)
        A = speye(N, format="csr") - W_mat

        # Build RHS: b = A · c_known  for free pixels → b_free = -(A_free_x_fixed · c_fixed)
        # Equivalently: for each constrained pixel r, we zero its row and set diagonal to 1.
        # Then: b[r] = c_scr[r]  for constrained, b[r] = 0 for free.
        A = A.tolil()
        b = np.zeros(N, dtype=np.float64)

        constrained_idx = np.where(scr_flat)[0]
        # Direct manipulation of lil_matrix internals is much faster than __setitem__
        for idx in constrained_idx:
            A.data[idx] = [1.0]      # Only diagonal value
            A.rows[idx] = [idx]       # Column index
            b[idx] = c_flat[idx]

        A = A.tocsr()
        C_solved = spsolve(A, b)
        return np.clip(C_solved.reshape(H, W), 0.0, 1.0)