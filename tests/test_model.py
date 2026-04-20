"""Tests for Zhang16Net and Zhang16Regression model architectures."""

import torch
import pytest


class TestZhang16Net:
    """Test the classification-based Zhang 2016 network."""

    @pytest.fixture
    def model(self, quantizer):
        from src.deep_learning.model import Zhang16Net
        return Zhang16Net(num_classes=quantizer.num_bins)

    def test_output_shape_single(self, model, quantizer):
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, quantizer.num_bins, 64, 64)

    def test_output_shape_batch(self, model, quantizer):
        x = torch.randn(4, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, quantizer.num_bins, 32, 32)

    def test_spatial_downsampling_is_8x(self, model, quantizer):
        """Output spatial resolution should be input/4 (block8 upsamples from /8 to /4)."""
        for size in [64, 128, 256]:
            x = torch.randn(1, 1, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape[2] == size // 4
            assert out.shape[3] == size // 4

    def test_predict_ab_full_resolution(self, model, quantizer):
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            ab = model.predict_ab(x, quantizer, temperature=0.38)
        assert ab.shape == (1, 2, 128, 128)

    def test_predict_ab_values_in_range(self, model, quantizer):
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            ab = model.predict_ab(x, quantizer, temperature=0.38)
        assert ab.min() >= -120
        assert ab.max() <= 120

    def test_parameter_count_reasonable(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total > 1_000_000   # at least 1M params
        assert total < 100_000_000  # less than 100M params

    def test_all_params_require_grad(self, model):
        for p in model.parameters():
            assert p.requires_grad


class TestZhang16Regression:
    """Test the regression-based variant."""

    @pytest.fixture
    def model(self):
        from src.deep_learning.model import Zhang16Regression
        return Zhang16Regression()

    def test_output_shape(self, model):
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 64, 64)

    def test_output_in_tanh_range(self, model):
        """Regression output should be in [-1, 1] due to Tanh."""
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_predict_ab_full_resolution(self, model):
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            ab = model.predict_ab(x)
        assert ab.shape == (1, 2, 128, 128)

    def test_predict_ab_denormalized_range(self, model):
        """predict_ab should return values in [-110, 110]."""
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            ab = model.predict_ab(x)
        assert ab.min() >= -110
        assert ab.max() <= 110


class TestBuildModel:
    """Test the model factory function."""

    def test_cross_entropy_builds_zhang16net(self, cfg, quantizer):
        from src.deep_learning.model import build_model, Zhang16Net
        cfg["deep_learning"]["loss"] = "cross_entropy"
        model = build_model(cfg, quantizer=quantizer)
        assert isinstance(model, Zhang16Net)

    def test_huber_builds_regression(self, cfg):
        from src.deep_learning.model import build_model, Zhang16Regression
        cfg["deep_learning"]["loss"] = "huber"
        model = build_model(cfg)
        assert isinstance(model, Zhang16Regression)

    def test_num_classes_matches_quantizer(self, cfg, quantizer):
        from src.deep_learning.model import build_model
        cfg["deep_learning"]["loss"] = "cross_entropy"
        model = build_model(cfg, quantizer=quantizer)
        # The final conv should output quantizer.num_bins channels
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == quantizer.num_bins
