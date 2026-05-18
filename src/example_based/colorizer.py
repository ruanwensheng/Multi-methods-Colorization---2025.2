import time
import numpy as np
import cv2

from .matching import build_feature_vectors, kd_tree_match
from .utils import bgr_to_lab, lab_to_bgr


class ExampleColorizer:
    """Example-based colorization via luminance + texture KD-tree matching.

    Implements Welsh, Ashikhmin, Mueller (SIGGRAPH 2002): each target pixel is
    matched to the most similar reference pixel by (L, local_std(L)) features;
    ab chrominance is transferred from the best match.
    """

    def __init__(self, cfg=None):
        params = (cfg or {}).get("example_based", {})
        self.neighborhood_size = params.get("neighborhood_size", 5)
        self.k_neighbors = params.get("k_neighbors", 5)
        self.downsample = params.get("downsample", 0.5)

    def colorize(self, gray_image, reference_image):
        """Transfer color from reference to grayscale target.

        Args:
            gray_image:       (H, W) or (H, W, 3) BGR uint8 grayscale target
            reference_image:  (H, W, 3) BGR uint8 color reference
        Returns:
            colorized_bgr: (H, W, 3) BGR uint8
            info_dict:     {'method', 'time_seconds', 'image_size'}
        """
        start = time.time()

        if gray_image.ndim == 2:
            bgr_target = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        else:
            bgr_target = gray_image

        orig_h, orig_w = bgr_target.shape[:2]
        L_target, _ = bgr_to_lab(bgr_target)

        # Downsample reference to control KD-tree build/query cost
        if self.downsample != 1.0:
            ds_w = max(1, int(reference_image.shape[1] * self.downsample))
            ds_h = max(1, int(reference_image.shape[0] * self.downsample))
            ref_small = cv2.resize(reference_image, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
        else:
            ref_small = reference_image

        L_ref, ab_ref = bgr_to_lab(ref_small)
        ref_h, ref_w = L_ref.shape

        lab_target = np.zeros((orig_h, orig_w, 3), dtype=np.float64)
        lab_target[:, :, 0] = L_target

        lab_ref = np.zeros((ref_h, ref_w, 3), dtype=np.float64)
        lab_ref[:, :, 0] = L_ref
        lab_ref[:, :, 1:] = ab_ref

        target_feats = build_feature_vectors(lab_target, self.neighborhood_size)
        ref_feats = build_feature_vectors(lab_ref, self.neighborhood_size)
        ref_ab_flat = ab_ref.reshape(-1, 2)

        ab_matched = kd_tree_match(target_feats, ref_feats, ref_ab_flat, self.k_neighbors)
        ab_transferred = ab_matched.reshape(orig_h, orig_w, 2)

        result_bgr = lab_to_bgr(L_target, ab_transferred)
        elapsed = time.time() - start

        return result_bgr, {
            "method": "example_based",
            "time_seconds": elapsed,
            "image_size": (orig_h, orig_w),
        }
