"""Tests for DeepColorizer - end-to-end inference wrapper."""

import numpy as np
import cv2
import pytest


class TestDeepColorizerInit:
    """Test colorizer initialization."""

    def test_creates_without_weights(self):
        from src.deep_learning.colorizer import DeepColorizer
        colorizer = DeepColorizer(device="cpu")
        assert colorizer.model is not None

    def test_default_temperature(self):
        from src.deep_learning.colorizer import DeepColorizer
        colorizer = DeepColorizer(device="cpu")
        assert colorizer.temperature == 0.38

    def test_model_in_eval_mode(self):
        from src.deep_learning.colorizer import DeepColorizer
        colorizer = DeepColorizer(device="cpu")
        assert not colorizer.model.training


class TestDeepColorizerColorize:
    """Test the colorize() method with various inputs."""

    @pytest.fixture
    def colorizer(self):
        from src.deep_learning.colorizer import DeepColorizer
        return DeepColorizer(device="cpu", temperature=0.38)

    def test_grayscale_input(self, colorizer, sample_gray):
        result, info = colorizer.colorize(sample_gray)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_bgr_3channel_input(self, colorizer):
        bgr = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result, info = colorizer.colorize(bgr)
        assert result.shape == (100, 80, 3)

    def test_preserves_original_resolution(self, colorizer):
        for h, w in [(50, 50), (100, 200), (300, 150)]:
            gray = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            result, _ = colorizer.colorize(gray)
            assert result.shape == (h, w, 3)

    def test_info_dict_has_required_keys(self, colorizer, sample_gray):
        _, info = colorizer.colorize(sample_gray)
        assert "elapsed_sec" in info
        assert "model_name" in info
        assert "device" in info
        assert "temperature" in info

    def test_elapsed_time_positive(self, colorizer, sample_gray):
        _, info = colorizer.colorize(sample_gray)
        assert info["elapsed_sec"] > 0

    def test_output_is_valid_bgr(self, colorizer, sample_gray):
        result, _ = colorizer.colorize(sample_gray)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_deterministic_with_same_input(self, colorizer):
        gray = np.full((32, 32), 128, dtype=np.uint8)
        r1, _ = colorizer.colorize(gray)
        r2, _ = colorizer.colorize(gray)
        np.testing.assert_array_equal(r1, r2)


class TestDeepColorizerBatch:
    """Test batch colorization."""

    def test_colorize_batch_returns_list(self, tmp_path):
        from src.deep_learning.colorizer import DeepColorizer

        # Create temp images
        for i in range(3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)

        colorizer = DeepColorizer(device="cpu")
        paths = [str(tmp_path / f"img_{i}.png") for i in range(3)]
        results = colorizer.colorize_batch(paths)
        assert len(results) == 3
        for result_bgr, info in results:
            assert result_bgr.shape[2] == 3
            assert "path" in info
