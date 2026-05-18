"""Tests for ExampleColorizer and its sub-components."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_bgr():
    """64x64 synthetic color BGR image."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32, :32] = [255, 0, 0]
    img[:32, 32:] = [0, 255, 0]
    img[32:, :32] = [0, 0, 255]
    img[32:, 32:] = [128, 128, 0]
    return img


@pytest.fixture
def sample_gray_2d():
    """64x64 single-channel grayscale."""
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def sample_gray_3ch(sample_bgr):
    """64x64 grayscale as (H, W, 3) BGR (all channels equal)."""
    import cv2
    g = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def colorizer():
    from src.example_based.colorizer import ExampleColorizer
    return ExampleColorizer()


# ---------------------------------------------------------------------------
# ExampleColorizer initialisation
# ---------------------------------------------------------------------------

class TestExampleColorizerInit:
    def test_creates_without_cfg(self):
        from src.example_based.colorizer import ExampleColorizer
        c = ExampleColorizer()
        assert c is not None

    def test_default_params(self):
        from src.example_based.colorizer import ExampleColorizer
        c = ExampleColorizer()
        assert c.neighborhood_size == 5
        assert c.k_neighbors == 5
        assert c.downsample == 0.5

    def test_cfg_overrides_defaults(self):
        from src.example_based.colorizer import ExampleColorizer
        cfg = {"example_based": {"neighborhood_size": 3, "k_neighbors": 3, "downsample": 1.0}}
        c = ExampleColorizer(cfg=cfg)
        assert c.neighborhood_size == 3
        assert c.k_neighbors == 3
        assert c.downsample == 1.0


# ---------------------------------------------------------------------------
# ExampleColorizer.colorize()
# ---------------------------------------------------------------------------

class TestExampleColorizerColorize:
    def test_input_hw3_bgr(self, colorizer, sample_gray_3ch, sample_bgr):
        result, info = colorizer.colorize(sample_gray_3ch, sample_bgr)
        assert result.shape == (64, 64, 3)

    def test_input_hw_gray(self, colorizer, sample_gray_2d, sample_bgr):
        result, info = colorizer.colorize(sample_gray_2d, sample_bgr)
        assert result.shape == (64, 64, 3)

    def test_output_dtype_uint8(self, colorizer, sample_bgr):
        result, _ = colorizer.colorize(sample_bgr, sample_bgr)
        assert result.dtype == np.uint8

    def test_preserves_original_resolution(self, colorizer, sample_bgr):
        for h, w in [(32, 32), (50, 80), (100, 60)]:
            gray = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            ref = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            result, _ = colorizer.colorize(gray, ref)
            assert result.shape == (h, w, 3), f"Expected ({h},{w},3), got {result.shape}"

    def test_output_valid_range(self, colorizer, sample_bgr):
        result, _ = colorizer.colorize(sample_bgr, sample_bgr)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_info_dict_has_required_keys(self, colorizer, sample_bgr):
        _, info = colorizer.colorize(sample_bgr, sample_bgr)
        assert "method" in info
        assert "time_seconds" in info
        assert "image_size" in info

    def test_info_method_value(self, colorizer, sample_bgr):
        _, info = colorizer.colorize(sample_bgr, sample_bgr)
        assert info["method"] == "example_based"

    def test_info_time_seconds_positive(self, colorizer, sample_bgr):
        _, info = colorizer.colorize(sample_bgr, sample_bgr)
        assert info["time_seconds"] > 0

    def test_info_image_size_matches_input(self, colorizer, sample_bgr):
        _, info = colorizer.colorize(sample_bgr, sample_bgr)
        assert info["image_size"] == (64, 64)

    def test_different_size_reference_ok(self, colorizer, sample_bgr):
        """Reference and target can have different spatial dimensions."""
        ref = np.random.randint(0, 256, (128, 96, 3), dtype=np.uint8)
        result, _ = colorizer.colorize(sample_bgr, ref)
        assert result.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# build_feature_vectors
# ---------------------------------------------------------------------------

class TestBuildFeatureVectors:
    def test_output_shape(self):
        from src.example_based.matching import build_feature_vectors
        lab = np.random.rand(32, 48, 3).astype(np.float64)
        lab[:, :, 0] *= 100
        lab[:, :, 1:] = lab[:, :, 1:] * 220 - 110
        feats = build_feature_vectors(lab, neighborhood_size=5)
        assert feats.shape == (32 * 48, 2)

    def test_L_column_matches_channel(self):
        from src.example_based.matching import build_feature_vectors
        lab = np.zeros((10, 10, 3), dtype=np.float64)
        lab[:, :, 0] = 55.0
        feats = build_feature_vectors(lab, neighborhood_size=3)
        np.testing.assert_allclose(feats[:, 0], 55.0)

    def test_std_nonnegative(self):
        from src.example_based.matching import build_feature_vectors
        lab = np.random.rand(20, 20, 3).astype(np.float64)
        feats = build_feature_vectors(lab)
        assert (feats[:, 1] >= 0).all()

    def test_uniform_image_std_near_zero(self):
        from src.example_based.matching import build_feature_vectors
        lab = np.full((16, 16, 3), 50.0, dtype=np.float64)
        feats = build_feature_vectors(lab)
        np.testing.assert_allclose(feats[:, 1], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# kd_tree_match
# ---------------------------------------------------------------------------

class TestKdTreeMatch:
    def _make_feats_ab(self, n):
        feats = np.random.rand(n, 2).astype(np.float64)
        feats[:, 0] *= 100
        ab = (np.random.rand(n, 2) * 220 - 110).astype(np.float64)
        return feats, ab

    def test_output_shape(self):
        from src.example_based.matching import kd_tree_match
        t_feats = np.random.rand(50, 2)
        r_feats, r_ab = self._make_feats_ab(200)
        result = kd_tree_match(t_feats, r_feats, r_ab, k=5)
        assert result.shape == (50, 2)

    def test_returns_valid_ab_range(self):
        from src.example_based.matching import kd_tree_match
        t_feats = np.random.rand(30, 2)
        r_feats, r_ab = self._make_feats_ab(100)
        result = kd_tree_match(t_feats, r_feats, r_ab, k=5)
        assert result.min() >= -130
        assert result.max() <= 130

    def test_exact_match_returns_correct_ab(self):
        """When a target feature exactly matches a reference feature, that ab is returned."""
        from src.example_based.matching import kd_tree_match
        r_feats = np.array([[50.0, 5.0], [80.0, 10.0], [20.0, 2.0]])
        r_ab = np.array([[10.0, -20.0], [30.0, 40.0], [-10.0, 5.0]])
        t_feats = np.array([[50.0, 5.0]])
        result = kd_tree_match(t_feats, r_feats, r_ab, k=1)
        np.testing.assert_allclose(result[0], r_ab[0], atol=1e-6)

    def test_k1_same_as_nearest_neighbor(self):
        """k=1 should simply return the single nearest neighbor's ab."""
        from src.example_based.matching import kd_tree_match
        from sklearn.neighbors import KDTree
        r_feats = np.random.rand(50, 2).astype(np.float64)
        r_ab = np.random.rand(50, 2).astype(np.float64)
        t_feats = np.random.rand(10, 2).astype(np.float64)
        result = kd_tree_match(t_feats, r_feats, r_ab, k=1)
        tree = KDTree(r_feats)
        _, idx = tree.query(t_feats, k=1)
        expected = r_ab[idx[:, 0]]
        np.testing.assert_allclose(result, expected)
