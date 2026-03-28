import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


class ScribbleColorizer:
    """
    Scribble-based colorization theo Levin et al. 2004.
    
    Nguyên lý: pixel lân cận có intensity tương tự → nên có màu tương tự.
    Cost function: J(U) = sum_p [ sum_{q in N(p)} w_pq * (U(p) - U(q))^2 ]
    Minimize J → giải sparse linear system → ra U, V cho toàn ảnh.
    """

    def __init__(self, sigma: float = 0.1, n_neighbors: int = 8):
        """
        Args:
            sigma: độ rộng của weight function (Eq.3 trong paper)
            n_neighbors: dùng 4 hoặc 8 pixel lân cận
        """
        self.sigma = sigma
        self.n_neighbors = n_neighbors
        # 8 hướng lân cận (dx, dy)
        if n_neighbors == 8:
            self.neighbors = [(-1,-1),(-1,0),(-1,1),
                              (0,-1),          (0,1),
                              (1,-1), (1,0),  (1,1)]
        else:  # 4 neighbors
            self.neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

    def _build_weight_matrix(self, Y: np.ndarray) -> lil_matrix:
        """
        Xây dựng weight matrix W từ kênh luminance Y.
        
        w_pq = exp(-(Y(p) - Y(q))^2 / (2 * sigma_p^2))
        sigma_p^2 = variance của Y trong cửa sổ quanh p
        
        Args:
            Y: kênh luminance, shape (H, W), range [0, 1]
        Returns:
            W: sparse matrix shape (H*W, H*W)
        """
        H, W = Y.shape
        N = H * W
        W_mat = lil_matrix((N, N), dtype=np.float64)

        for r in range(H):
            for c in range(W):
                p = r * W + c  # index phẳng của pixel p

                # Tính variance trong cửa sổ 3x3 quanh p (sigma_p^2)
                r0, r1 = max(0, r-1), min(H, r+2)
                c0, c1 = max(0, c-1), min(W, c+2)
                window = Y[r0:r1, c0:c1]
                var_p = np.var(window) + 1e-6  # tránh chia 0

                for dr, dc in self.neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        q = nr * W + nc
                        # Weight function Eq.3: Levin et al.
                        diff = float(Y[r, c]) - float(Y[nr, nc])
                        w = np.exp(-(diff ** 2) / (2 * var_p))
                        W_mat[p, q] = w

        return W_mat

    def _solve_channel(
        self,
        Y: np.ndarray,
        W_mat,
        scribble_mask: np.ndarray,
        scribble_values: np.ndarray
    ) -> np.ndarray:
        """
        Giải linear system cho một kênh màu (U hoặc V).
        
        Với pixel có scribble: C(p) = giá trị đã biết (constraint cứng)
        Với pixel khác: minimize cost function → hệ phương trình tuyến tính
        
        Args:
            Y: luminance (H, W)
            W_mat: weight matrix (N, N)
            scribble_mask: bool mask (H*W,) — True nếu pixel có scribble
            scribble_values: giá trị màu tại scribble pixel (H*W,)
        Returns:
            channel: giá trị màu đã giải cho toàn ảnh (H*W,)
        """
        H, W = Y.shape
        N = H * W

        from scipy.sparse import eye, diags
        from scipy.sparse import csr_matrix

        W_csr = csr_matrix(W_mat)

        # Tổng weight mỗi hàng → diagonal matrix D
        row_sums = np.array(W_csr.sum(axis=1)).flatten()
        D = diags(row_sums)

        # Laplacian: L = D - W
        L = D - W_csr

        # Build hệ phương trình: A * x = b
        # - Pixel có scribble: x(p) = known_value  → hàng p của A là identity
        # - Pixel khác: L(p,:) * x = 0             → minimize cost
        A = lil_matrix((N, N), dtype=np.float64)
        b = np.zeros(N, dtype=np.float64)

        mask_flat = scribble_mask.flatten()
        val_flat  = scribble_values.flatten()

        for p in range(N):
            if mask_flat[p]:
                # Constraint: pixel này có màu biết sẵn
                A[p, p] = 1.0
                b[p] = val_flat[p]
            else:
                # Minimize: sum_q w_pq*(C(p) - C(q)) = 0
                A[p, :] = L[p, :]

        A_csr = csr_matrix(A)
        # Giải sparse linear system
        channel = spsolve(A_csr, b)
        return np.clip(channel, 0, 1).reshape(H, W)

    def colorize(
        self,
        gray_img: np.ndarray,
        scribble_img: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """
        Colorize ảnh grayscale dựa trên scribble của user.
        
        Args:
            gray_img  : ảnh grayscale BGR (H, W, 3) hoặc (H, W)
            scribble_img: ảnh scribble BGR (H, W, 3) — pixel = (0,0,0)
                          nếu không có scribble, ngược lại là màu user vẽ
        Returns:
            result: ảnh màu BGR (H, W, 3)
            info  : dict chứa thời gian chạy và số pixel có scribble
        """
        t_start = time.time()

        # --- Chuẩn hóa input ---
        if gray_img.ndim == 3:
            gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray_img.copy()

        H, W = gray.shape

        # Normalize về [0, 1]
        Y = gray.astype(np.float64) / 255.0

        # --- Chuyển scribble sang YUV ---
        scribble_yuv = cv2.cvtColor(scribble_img, cv2.COLOR_BGR2YUV)
        scribble_yuv = scribble_yuv.astype(np.float64) / 255.0

        # Mask: pixel nào khác (0,0,0) trong ảnh scribble → có màu
        scribble_gray = cv2.cvtColor(scribble_img, cv2.COLOR_BGR2GRAY)
        scribble_mask = scribble_gray > 10  # threshold nhỏ để lọc noise

        n_scribble = int(scribble_mask.sum())
        logger.info(f"Số pixel có scribble: {n_scribble}")

        if n_scribble == 0:
            raise ValueError("Không tìm thấy scribble nào. Hãy vẽ ít nhất 1 stroke màu.")

        # --- Build weight matrix (bước tốn thời gian nhất) ---
        logger.info("Đang build weight matrix...")
        t_w = time.time()
        W_mat = self._build_weight_matrix(Y)
        logger.info(f"Weight matrix: {time.time() - t_w:.2f}s")

        # --- Giải cho kênh U và V ---
        logger.info("Đang giải kênh U...")
        U_solved = self._solve_channel(
            Y, W_mat,
            scribble_mask,
            scribble_yuv[:, :, 1]  # kênh U
        )

        logger.info("Đang giải kênh V...")
        V_solved = self._solve_channel(
            Y, W_mat,
            scribble_mask,
            scribble_yuv[:, :, 2]  # kênh V
        )

        # --- Merge Y + U + V → BGR ---
        result_yuv = np.stack([Y, U_solved, V_solved], axis=2)
        result_yuv = (result_yuv * 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result_yuv, cv2.COLOR_YUV2BGR)

        elapsed = time.time() - t_start
        info = {
            "elapsed_sec": round(elapsed, 3),
            "n_scribble_pixels": n_scribble,
            "image_size": (H, W),
            "sigma": self.sigma,
        }
        logger.info(f"Colorization xong: {elapsed:.2f}s")
        return result_bgr, info