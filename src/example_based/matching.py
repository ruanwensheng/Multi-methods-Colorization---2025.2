import numpy as np
from sklearn.neighbors import KDTree

from .utils import local_std


def build_feature_vectors(lab_image, neighborhood_size=5):
    """Build per-pixel feature vectors [L, local_std(L)] for KD-tree matching.

    Args:
        lab_image:         (H, W, 3) float64 CIE Lab image
        neighborhood_size: NxN window for local std computation
    Returns:
        (H*W, 2) float64 — rows are [L(p), std(L_neighborhood(p))]
    """
    L = lab_image[:, :, 0]
    std_map = local_std(L, window_size=neighborhood_size)
    return np.stack([L.ravel(), std_map.ravel()], axis=1)


def kd_tree_match(target_feats, ref_feats, ref_ab, k=5):
    """Match each target pixel to the best reference pixel via KD-tree.

    Finds k nearest neighbors in (L, std) feature space, then among those
    candidates picks the one with minimum |L_target - L_ref|.

    Args:
        target_feats: (N_target, 2) float64 [L, std]
        ref_feats:    (N_ref, 2) float64 [L, std]
        ref_ab:       (N_ref, 2) float64 ab values of reference pixels
        k:            number of nearest-neighbor candidates to consider
    Returns:
        (N_target, 2) float64 — ab values transferred to each target pixel
    """
    tree = KDTree(ref_feats)
    _, indices = tree.query(target_feats, k=k)  # (N, k)

    L_target = target_feats[:, 0:1]               # (N, 1)
    L_ref_cands = ref_feats[indices, 0]            # (N, k)
    best = np.argmin(np.abs(L_ref_cands - L_target), axis=1)  # (N,)

    best_ref_idx = indices[np.arange(len(target_feats)), best]
    return ref_ab[best_ref_idx]
