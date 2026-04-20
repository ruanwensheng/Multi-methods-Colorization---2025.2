"""Tests for loss functions - ClassRebalancedCELoss and HuberColorLoss."""

import torch
import numpy as np
import pytest


class TestClassRebalancedCELoss:
    """Test the classification-based loss with class rebalancing."""

    @pytest.fixture
    def loss_fn(self, quantizer):
        from src.deep_learning.loss import ClassRebalancedCELoss
        return ClassRebalancedCELoss(quantizer)

    def test_returns_scalar(self, loss_fn, quantizer):
        logits = torch.randn(2, quantizer.num_bins, 32, 32)
        target = torch.randn(2, 2, 256, 256)
        loss = loss_fn(logits, target)
        assert loss.dim() == 0  # scalar

    def test_loss_positive(self, loss_fn, quantizer):
        logits = torch.randn(2, quantizer.num_bins, 32, 32)
        target = torch.randn(2, 2, 128, 128)
        loss = loss_fn(logits, target)
        assert loss.item() > 0

    def test_loss_decreases_toward_target(self, loss_fn, quantizer):
        """Loss should be lower when predictions are closer to target."""
        target = torch.zeros(1, 2, 64, 64)  # ab = (0, 0)

        # Random predictions (high loss)
        logits_random = torch.randn(1, quantizer.num_bins, 16, 16)

        # Peaked predictions near origin (lower loss)
        logits_peaked = torch.full((1, quantizer.num_bins, 16, 16), -10.0)
        origin_idx = quantizer.encode(np.array([0.0, 0.0]))
        logits_peaked[0, origin_idx, :, :] = 10.0

        loss_random = loss_fn(logits_random, target)
        loss_peaked = loss_fn(logits_peaked, target)
        assert loss_peaked.item() < loss_random.item()

    def test_gradient_flows(self, loss_fn, quantizer):
        logits = torch.randn(1, quantizer.num_bins, 16, 16, requires_grad=True)
        target = torch.randn(1, 2, 64, 64)
        loss = loss_fn(logits, target)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_with_custom_weights(self, quantizer):
        from src.deep_learning.loss import ClassRebalancedCELoss
        weights = np.random.rand(quantizer.num_bins).astype(np.float32)
        loss_fn = ClassRebalancedCELoss(quantizer, class_weights=weights)
        logits = torch.randn(1, quantizer.num_bins, 16, 16)
        target = torch.randn(1, 2, 64, 64)
        loss = loss_fn(logits, target)
        assert loss.item() > 0


class TestHuberColorLoss:
    """Test the Smooth-L1 regression loss."""

    @pytest.fixture
    def loss_fn(self):
        from src.deep_learning.loss import HuberColorLoss
        return HuberColorLoss()

    def test_returns_scalar(self, loss_fn):
        pred = torch.randn(2, 2, 64, 64)
        target = torch.randn(2, 2, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_zero_loss_for_identical(self, loss_fn):
        x = torch.randn(1, 2, 32, 32)
        loss = loss_fn(x, x)
        assert loss.item() < 1e-6

    def test_handles_size_mismatch(self, loss_fn):
        """Should resize target to match prediction."""
        pred = torch.randn(1, 2, 32, 32)
        target = torch.randn(1, 2, 128, 128)
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_gradient_flows(self, loss_fn):
        pred = torch.randn(1, 2, 32, 32, requires_grad=True)
        target = torch.randn(1, 2, 32, 32)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestBuildLoss:
    """Test the loss factory function."""

    def test_builds_ce_loss(self, cfg, quantizer):
        from src.deep_learning.loss import build_loss, ClassRebalancedCELoss
        cfg["deep_learning"]["loss"] = "cross_entropy"
        loss_fn = build_loss(cfg, quantizer=quantizer)
        assert isinstance(loss_fn, ClassRebalancedCELoss)

    def test_builds_huber_loss(self, cfg):
        from src.deep_learning.loss import build_loss, HuberColorLoss
        cfg["deep_learning"]["loss"] = "huber"
        loss_fn = build_loss(cfg)
        assert isinstance(loss_fn, HuberColorLoss)

    def test_ce_requires_quantizer(self, cfg):
        from src.deep_learning.loss import build_loss
        cfg["deep_learning"]["loss"] = "cross_entropy"
        with pytest.raises(ValueError):
            build_loss(cfg, quantizer=None)

    def test_unknown_loss_raises(self, cfg):
        from src.deep_learning.loss import build_loss
        cfg["deep_learning"]["loss"] = "unknown"
        with pytest.raises(ValueError):
            build_loss(cfg)
