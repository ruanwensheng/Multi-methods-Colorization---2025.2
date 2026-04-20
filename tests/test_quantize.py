"""Tests for ABQuantizer - ab color space quantization."""

import numpy as np
import pytest


class TestABQuantizerInit:
    """Test ABQuantizer initialization and bin computation."""

    def test_creates_nonzero_bins(self, quantizer):
        assert quantizer.num_bins > 0

    def test_bins_shape_is_n_by_2(self, quantizer):
        assert quantizer.ab_bins.shape == (quantizer.num_bins, 2)

    def test_bins_are_within_ab_range(self, quantizer):
        assert np.all(quantizer.ab_bins >= -110)
        assert np.all(quantizer.ab_bins <= 110)

    def test_bins_are_on_grid(self, quantizer):
        """All bin centers should be multiples of grid_size."""
        remainders = quantizer.ab_bins % quantizer.grid_size
        assert np.allclose(remainders, 0)

    def test_no_duplicate_bins(self, quantizer):
        unique = np.unique(quantizer.ab_bins, axis=0)
        assert len(unique) == quantizer.num_bins

    def test_includes_origin(self, quantizer):
        """The origin (0, 0) should be in-gamut."""
        has_origin = np.any(np.all(quantizer.ab_bins == 0, axis=1))
        assert has_origin


class TestABQuantizerEncode:
    """Test encoding ab values to bin indices."""

    def test_single_point_returns_scalar(self, quantizer):
        ab = np.array([0.0, 0.0])
        idx = quantizer.encode(ab)
        assert idx.shape == ()

    def test_batch_returns_correct_shape(self, quantizer):
        ab = np.random.randn(10, 2) * 50
        indices = quantizer.encode(ab)
        assert indices.shape == (10,)

    def test_2d_spatial_returns_correct_shape(self, quantizer):
        ab = np.random.randn(8, 8, 2) * 50
        indices = quantizer.encode(ab)
        assert indices.shape == (8, 8)

    def test_indices_in_valid_range(self, quantizer):
        ab = np.random.randn(100, 2) * 50
        indices = quantizer.encode(ab)
        assert np.all(indices >= 0)
        assert np.all(indices < quantizer.num_bins)

    def test_origin_encodes_to_nearest_bin(self, quantizer):
        ab = np.array([0.0, 0.0])
        idx = quantizer.encode(ab)
        # Encoded bin center should be close to origin
        decoded_center = quantizer.ab_bins[idx]
        assert np.linalg.norm(decoded_center) < 15  # within one grid step

    def test_encoding_is_deterministic(self, quantizer):
        ab = np.array([[30.0, -20.0], [50.0, 50.0]])
        idx1 = quantizer.encode(ab)
        idx2 = quantizer.encode(ab)
        np.testing.assert_array_equal(idx1, idx2)


class TestABQuantizerDecode:
    """Test decoding probability distributions to ab values."""

    def test_uniform_probs_near_origin(self, quantizer):
        """Uniform distribution should decode near the center of mass of all bins."""
        probs = np.ones((1, quantizer.num_bins)) / quantizer.num_bins
        ab = quantizer.decode(probs, temperature=1.0)
        assert ab.shape == (1, 2)
        # Center of mass should be roughly near origin
        assert np.all(np.abs(ab) < 50)

    def test_one_hot_decodes_to_bin_center(self, quantizer):
        """A one-hot distribution should decode very close to that bin's center."""
        target_idx = 5
        probs = np.zeros((1, quantizer.num_bins))
        probs[0, target_idx] = 1.0
        ab = quantizer.decode(probs, temperature=1.0)
        np.testing.assert_allclose(ab[0], quantizer.ab_bins[target_idx], atol=0.01)

    def test_low_temperature_sharpens(self, quantizer):
        """Lower temperature should produce a sharper (less uniform) distribution."""
        # Create a distribution with a clear peak
        probs = np.ones((1, quantizer.num_bins)) * 0.001
        probs[0, 10] = 0.9
        probs = probs / probs.sum()
        mode_idx = 10

        ab_warm = quantizer.decode(probs, temperature=1.0)
        ab_cold = quantizer.decode(probs, temperature=0.1)

        mode_center = quantizer.ab_bins[mode_idx]
        dist_warm = np.linalg.norm(ab_warm[0] - mode_center)
        dist_cold = np.linalg.norm(ab_cold[0] - mode_center)
        assert dist_cold <= dist_warm

    def test_output_shape_matches_input_batch(self, quantizer):
        probs = np.random.rand(5, quantizer.num_bins)
        probs = probs / probs.sum(axis=1, keepdims=True)
        ab = quantizer.decode(probs, temperature=0.38)
        assert ab.shape == (5, 2)


class TestABQuantizerClassWeights:
    """Test class rebalancing weight computation."""

    def test_uniform_weights_when_no_distribution(self, quantizer):
        weights = quantizer.compute_class_weights(empirical_dist=None)
        assert weights.shape == (quantizer.num_bins,)
        np.testing.assert_array_equal(weights, np.ones(quantizer.num_bins))

    def test_weights_positive(self, quantizer):
        dist = np.random.rand(quantizer.num_bins)
        dist = dist / dist.sum()
        weights = quantizer.compute_class_weights(empirical_dist=dist)
        assert np.all(weights > 0)

    def test_rare_classes_get_higher_weight(self, quantizer):
        """Rare colors should get higher rebalancing weight."""
        dist = np.ones(quantizer.num_bins) / quantizer.num_bins
        # Make one class very rare
        dist[0] = 0.0001
        dist = dist / dist.sum()
        weights = quantizer.compute_class_weights(empirical_dist=dist)
        # Weight for the rare class should be above average
        assert weights[0] > np.mean(weights)
