"""Tests for utility functions - color conversion, metrics, config."""

import os
import numpy as np
import pytest


class TestColorConversions:
    """Test RGB <-> Lab color space conversions."""

    def test_rgb_to_lab_returns_L_and_ab(self, sample_rgb):
        from src.deep_learning.utils import rgb_to_lab
        L, ab = rgb_to_lab(sample_rgb)
        assert L.shape == (64, 64)
        assert ab.shape == (64, 64, 2)

    def test_L_range_0_to_100(self, sample_rgb):
        from src.deep_learning.utils import rgb_to_lab
        L, _ = rgb_to_lab(sample_rgb)
        assert L.min() >= 0.0
        assert L.max() <= 100.0

    def test_ab_range_reasonable(self, sample_rgb):
        from src.deep_learning.utils import rgb_to_lab
        _, ab = rgb_to_lab(sample_rgb)
        assert ab.min() >= -128
        assert ab.max() <= 128

    def test_lab_to_rgb_returns_uint8(self, sample_rgb):
        from src.deep_learning.utils import rgb_to_lab, lab_to_rgb
        L, ab = rgb_to_lab(sample_rgb)
        rgb_back = lab_to_rgb(L, ab)
        assert rgb_back.dtype == np.uint8
        assert rgb_back.shape == sample_rgb.shape

    def test_roundtrip_preserves_image(self, sample_rgb):
        """RGB -> Lab -> RGB should preserve the image within rounding error."""
        from src.deep_learning.utils import rgb_to_lab, lab_to_rgb
        L, ab = rgb_to_lab(sample_rgb)
        rgb_back = lab_to_rgb(L, ab)
        max_diff = np.abs(sample_rgb.astype(int) - rgb_back.astype(int)).max()
        assert max_diff <= 2  # allow small rounding error

    def test_black_image(self):
        from src.deep_learning.utils import rgb_to_lab, lab_to_rgb
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        L, ab = rgb_to_lab(black)
        assert np.allclose(L, 0, atol=1)
        rgb_back = lab_to_rgb(L, ab)
        assert np.all(rgb_back <= 5)

    def test_white_image(self):
        from src.deep_learning.utils import rgb_to_lab, lab_to_rgb
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        L, ab = rgb_to_lab(white)
        assert L.mean() > 95
        rgb_back = lab_to_rgb(L, ab)
        assert rgb_back.mean() > 250


class TestNormalization:
    """Test L and ab normalization/denormalization."""

    def test_normalize_L_range(self):
        from src.deep_learning.utils import normalize_L, denormalize_L
        L = np.array([0.0, 50.0, 100.0])
        L_norm = normalize_L(L)
        np.testing.assert_allclose(L_norm, [-1.0, 0.0, 1.0])

    def test_denormalize_L_roundtrip(self):
        from src.deep_learning.utils import normalize_L, denormalize_L
        L = np.array([25.0, 50.0, 75.0])
        np.testing.assert_allclose(denormalize_L(normalize_L(L)), L)

    def test_normalize_ab_range(self):
        from src.deep_learning.utils import normalize_ab, denormalize_ab
        ab = np.array([-110.0, 0.0, 110.0])
        ab_norm = normalize_ab(ab)
        np.testing.assert_allclose(ab_norm, [-1.0, 0.0, 1.0])

    def test_denormalize_ab_roundtrip(self):
        from src.deep_learning.utils import normalize_ab, denormalize_ab
        ab = np.array([-50.0, 0.0, 50.0])
        np.testing.assert_allclose(denormalize_ab(normalize_ab(ab)), ab)


class TestComputeMetrics:
    """Test image quality metric computation."""

    def test_identical_images_high_psnr(self, sample_rgb):
        from src.deep_learning.utils import compute_metrics
        metrics = compute_metrics(sample_rgb, sample_rgb)
        assert metrics["psnr"] > 40  # near-perfect
        assert metrics["ssim"] > 0.99

    def test_different_images_lower_psnr(self, sample_rgb):
        from src.deep_learning.utils import compute_metrics
        noisy = np.clip(sample_rgb.astype(int) + np.random.randint(-50, 50, sample_rgb.shape), 0, 255).astype(np.uint8)
        metrics = compute_metrics(noisy, sample_rgb)
        assert metrics["psnr"] < 40
        assert metrics["ssim"] < 0.99

    def test_returns_dict_with_required_keys(self, sample_rgb):
        from src.deep_learning.utils import compute_metrics
        metrics = compute_metrics(sample_rgb, sample_rgb)
        assert "psnr" in metrics
        assert "ssim" in metrics

    def test_psnr_is_float(self, sample_rgb):
        from src.deep_learning.utils import compute_metrics
        metrics = compute_metrics(sample_rgb, sample_rgb)
        assert isinstance(metrics["psnr"], float)
        assert isinstance(metrics["ssim"], float)


class TestLoadConfig:
    """Test YAML configuration loading."""

    def test_loads_valid_config(self, cfg):
        assert "project" in cfg
        assert "deep_learning" in cfg
        assert "evaluation" in cfg

    def test_project_name(self, cfg):
        assert cfg["project"]["name"] == "colorization-mlops"

    def test_deep_learning_section(self, cfg):
        dl = cfg["deep_learning"]
        assert "model" in dl
        assert "epochs" in dl
        assert "batch_size" in dl
        assert "learning_rate" in dl
        assert "loss" in dl
        assert "comparison" in dl

    def test_comparison_models_present(self, cfg):
        comp = cfg["deep_learning"]["comparison"]
        assert "zhang2017" in comp
        assert "deoldify" in comp
        assert "controlnet" in comp

    def test_no_scribble_or_example_sections(self, cfg):
        # This branch is deep-learning only; scribble/example live on other branches.
        # See SPEC.md acceptance criteria and CLAUDE.md "Never Do" list.
        assert "scribble" not in cfg
        assert "example_based" not in cfg

    def test_only_expected_top_level_keys(self, cfg):
        expected = {"project", "paths", "image", "deep_learning", "evaluation", "device"}
        assert set(cfg.keys()) == expected

    def test_missing_file_raises(self):
        from src.deep_learning.utils import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestGetDevice:
    """Test device auto-detection."""

    def test_returns_torch_device(self):
        import torch
        from src.deep_learning.utils import get_device
        device = get_device()
        assert isinstance(device, torch.device)

    def test_respects_config_override(self):
        import torch
        from src.deep_learning.utils import get_device
        device = get_device({"device": "cpu"})
        assert device == torch.device("cpu")
