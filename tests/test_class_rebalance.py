"""Tests for empirical class distribution + ClassRebalancedCELoss rebalancing.

Before this fix, build_loss() called compute_class_weights() with no empirical
distribution, so it returned uniform weights — silently disabling the entire
rebalancing scheme. These tests pin that the empirical distribution is now
computed (and cached) and that the resulting weights are non-uniform on
natural-image data."""

import os
import numpy as np
import cv2
import pytest


@pytest.fixture
def tiny_image_dir(tmp_path):
    """A handful of color-biased images to give a non-uniform ab distribution."""
    # Mostly muted natural-looking colors (skin-tone / earth-tone bias).
    rng = np.random.default_rng(0)
    for i in range(6):
        # Channels biased toward warm desaturated tones.
        r = rng.integers(140, 230, size=(64, 64), dtype=np.uint16)
        g = rng.integers(110, 180, size=(64, 64), dtype=np.uint16)
        b = rng.integers( 80, 150, size=(64, 64), dtype=np.uint16)
        img = np.stack([b, g, r], axis=-1).astype(np.uint8)  # BGR for cv2
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)
    return str(tmp_path)


class TestComputeEmpiricalDistribution:
    def test_returns_distribution_over_bins(self, tiny_image_dir, quantizer):
        from src.deep_learning.quantize import compute_empirical_distribution
        from src.deep_learning.dataset import ColorizationDataset

        ds = ColorizationDataset(tiny_image_dir, input_size=(64, 64))
        dist = compute_empirical_distribution(ds, quantizer, max_samples=6)

        assert dist.shape == (quantizer.num_bins,)
        assert dist.dtype == np.float64 or dist.dtype == np.float32
        np.testing.assert_allclose(dist.sum(), 1.0, atol=1e-5)

    def test_distribution_is_non_uniform_on_natural_data(self, tiny_image_dir, quantizer):
        from src.deep_learning.quantize import compute_empirical_distribution
        from src.deep_learning.dataset import ColorizationDataset

        ds = ColorizationDataset(tiny_image_dir, input_size=(64, 64))
        dist = compute_empirical_distribution(ds, quantizer, max_samples=6)

        uniform = np.full(quantizer.num_bins, 1.0 / quantizer.num_bins)
        # Real-image ab is dramatically concentrated in a few bins.
        assert np.linalg.norm(dist - uniform) > 0.1, \
            "Empirical distribution must be visibly non-uniform on natural images"

    def test_caches_to_disk(self, tiny_image_dir, quantizer, tmp_path):
        from src.deep_learning.quantize import compute_empirical_distribution
        from src.deep_learning.dataset import ColorizationDataset

        ds = ColorizationDataset(tiny_image_dir, input_size=(64, 64))
        cache = str(tmp_path / "dist.npy")
        d1 = compute_empirical_distribution(ds, quantizer, max_samples=6, cache_path=cache)
        assert os.path.exists(cache)

        # Second call must hit cache (we sentinel by writing a totally different
        # array to the cache; if cache is honored, that array is returned).
        sentinel = np.full(quantizer.num_bins, 1.0 / quantizer.num_bins).astype(np.float64)
        np.save(cache, sentinel)
        d2 = compute_empirical_distribution(ds, quantizer, max_samples=6, cache_path=cache)
        np.testing.assert_allclose(d2, sentinel)


class TestBuildLossWithRebalance:
    def test_ce_loss_uses_rebalanced_weights_when_dataset_supplied(
        self, cfg, quantizer, tiny_image_dir, tmp_path
    ):
        from src.deep_learning.loss import build_loss
        from src.deep_learning.dataset import ColorizationDataset

        cfg["deep_learning"]["loss"] = "cross_entropy"
        cfg["deep_learning"]["class_weight_sample_size"] = 6
        cfg["paths"]["models_deep"] = str(tmp_path)

        ds = ColorizationDataset(tiny_image_dir, input_size=(64, 64))
        loss_fn = build_loss(cfg, quantizer=quantizer, train_dataset=ds)

        weights = loss_fn.class_weights.cpu().numpy()
        assert weights.shape == (quantizer.num_bins,)
        # Uniform weights ≡ 1.0 across the board; rebalanced ones must vary.
        assert weights.std() > 0.01, \
            "Class weights must vary across bins — flat weights mean rebalancing is disabled"

    def test_ce_loss_still_works_without_dataset(self, cfg, quantizer):
        """Backwards compatibility — old code path with no training data must still build."""
        from src.deep_learning.loss import build_loss, ClassRebalancedCELoss
        cfg["deep_learning"]["loss"] = "cross_entropy"
        loss_fn = build_loss(cfg, quantizer=quantizer)
        assert isinstance(loss_fn, ClassRebalancedCELoss)
