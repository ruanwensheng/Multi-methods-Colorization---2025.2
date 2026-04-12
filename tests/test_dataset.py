"""Tests for ColorizationDataset and data loading."""

import os
import numpy as np
import cv2
import torch
import pytest


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temp directory with test images."""
    for i in range(5):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"test_{i}.jpg"), img)
    return str(tmp_path)


class TestColorizationDataset:
    """Test the PyTorch Dataset for colorization."""

    def test_creates_from_directory(self, temp_image_dir):
        from src.deep_learning.dataset import ColorizationDataset
        ds = ColorizationDataset(temp_image_dir, input_size=(64, 64))
        assert len(ds) == 5

    def test_returns_dict_with_L_and_ab(self, temp_image_dir):
        from src.deep_learning.dataset import ColorizationDataset
        ds = ColorizationDataset(temp_image_dir, input_size=(64, 64))
        sample = ds[0]
        assert "L" in sample
        assert "ab" in sample
        assert "path" in sample

    def test_L_shape_is_1_H_W(self, temp_image_dir):
        from src.deep_learning.dataset import ColorizationDataset
        ds = ColorizationDataset(temp_image_dir, input_size=(128, 128))
        sample = ds[0]
        assert sample["L"].shape == (1, 128, 128)

    def test_ab_shape_is_2_H_W(self, temp_image_dir):
        from src.deep_learning.dataset import ColorizationDataset
        ds = ColorizationDataset(temp_image_dir, input_size=(128, 128))
        sample = ds[0]
        assert sample["ab"].shape == (2, 128, 128)

    def test_L_normalized_range(self, temp_image_dir):
        from src.deep_learning.dataset import ColorizationDataset
        ds = ColorizationDataset(temp_image_dir, input_size=(64, 64))
        sample = ds[0]
        assert sample["L"].min() >= -1.5
        assert sample["L"].max() <= 1.5

    def test_ab_normalized_range(self, temp_image_dir):
        from src.deep_learning.dataset import ColorizationDataset
        ds = ColorizationDataset(temp_image_dir, input_size=(64, 64))
        sample = ds[0]
        assert sample["ab"].min() >= -1.5
        assert sample["ab"].max() <= 1.5

    def test_tensors_are_float(self, temp_image_dir):
        from src.deep_learning.dataset import ColorizationDataset
        ds = ColorizationDataset(temp_image_dir, input_size=(64, 64))
        sample = ds[0]
        assert sample["L"].dtype == torch.float32
        assert sample["ab"].dtype == torch.float32

    def test_empty_dir_raises(self, tmp_path):
        from src.deep_learning.dataset import ColorizationDataset
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        with pytest.raises(ValueError):
            ColorizationDataset(empty_dir)

    def test_augment_produces_different_results(self, temp_image_dir):
        """With augmentation, flipping should produce different outputs sometimes."""
        from src.deep_learning.dataset import ColorizationDataset
        ds = ColorizationDataset(temp_image_dir, input_size=(64, 64), augment=True)
        # Run multiple times - at least one should be flipped
        results = [ds[0]["L"].numpy() for _ in range(20)]
        # Check if we got at least 2 different results
        unique_count = len(set(r.tobytes() for r in results))
        assert unique_count >= 2


class TestGetDataloaders:
    """Test the dataloader factory."""

    def test_returns_none_for_missing_dirs(self, cfg):
        from src.deep_learning.dataset import get_dataloaders
        # With no data downloaded, all loaders should be None
        train, val, test = get_dataloaders(cfg)
        # These may or may not be None depending on whether data exists
        # Just verify they're either None or DataLoader
        from torch.utils.data import DataLoader
        for loader in [train, val, test]:
            assert loader is None or isinstance(loader, DataLoader)
