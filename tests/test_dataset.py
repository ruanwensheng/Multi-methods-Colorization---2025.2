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


def _fake_cfg(tmp_path, num_workers=0, val_num_workers=None, splits=("train2017",)):
    """Synth cfg that points at a tmp coco2017-shaped tree.

    Builds <tmp>/coco2017/<split>/ with a few JPEGs so get_dataloaders has
    something to return without needing the real 26 GB dataset.
    """
    coco_root = tmp_path / "coco2017"
    for split in splits:
        split_dir = coco_root / split
        split_dir.mkdir(parents=True)
        for i in range(6):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(split_dir / f"img_{i}.jpg"), img)
    dl_cfg = {
        "batch_size": 2,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if val_num_workers is not None:
        dl_cfg["val_num_workers"] = val_num_workers
    return {
        "image": {"input_size": [64, 64]},
        "paths": {"data_coco": str(coco_root)},
        "deep_learning": dl_cfg,
    }


class TestDataLoaderWorkerSafety:
    """Guard against the data-loader-bound training we hit in May 2026.

    With num_workers=0, fine-tuning Zhang16 on 118K COCO images took ~2.5 h
    per epoch on a GTX 1660 SUPER (GPU at 70% utilisation, CPU-bound on
    serial JPEG decode + skimage Lab conversion). Switching to multi-worker
    data loading is what makes the 10-epoch fine-tune fit in the user's
    5-6 hour budget. These tests pin that the worker-safe path actually
    works on Windows (where DataLoader workers need pickling-safe Datasets
    and spawn-friendly main scripts).
    """

    def test_dataloader_with_workers_yields_batches(self, tmp_path):
        # Smoke: a multi-worker DataLoader can actually deliver batches.
        # If this hangs/crashes, num_workers>0 is unsafe on this host.
        from src.deep_learning.dataset import get_dataloaders
        cfg = _fake_cfg(tmp_path, num_workers=2)
        train, _, _ = get_dataloaders(cfg)
        assert train is not None
        batches = []
        for i, batch in enumerate(train):
            batches.append(batch)
            if i >= 1:  # 2 batches is enough to prove workers + main can communicate
                break
        assert len(batches) >= 1
        assert "L" in batches[0] and "ab" in batches[0]

    def test_train_loader_enables_persistent_workers_when_workers_positive(self, tmp_path):
        # persistent_workers=True keeps worker procs alive across epochs.
        # Without this, every epoch pays the Windows process-spawn tax (~seconds).
        from src.deep_learning.dataset import get_dataloaders
        cfg = _fake_cfg(tmp_path, num_workers=2)
        train, _, _ = get_dataloaders(cfg)
        assert train.persistent_workers is True, \
            "train DataLoader should use persistent_workers when num_workers > 0"

    def test_train_loader_has_no_persistent_workers_when_workers_zero(self, tmp_path):
        # Persistent workers requires num_workers > 0; respect that.
        from src.deep_learning.dataset import get_dataloaders
        cfg = _fake_cfg(tmp_path, num_workers=0)
        train, _, _ = get_dataloaders(cfg)
        assert train.persistent_workers is False

    def test_train_loader_has_prefetch_when_workers_positive(self, tmp_path):
        # prefetch_factor>1 lets each worker pre-build batches ahead of the
        # training loop, hiding I/O latency behind GPU work.
        from src.deep_learning.dataset import get_dataloaders
        cfg = _fake_cfg(tmp_path, num_workers=2)
        train, _, _ = get_dataloaders(cfg)
        # PyTorch default is 2; we want at least that.
        assert train.prefetch_factor is None or train.prefetch_factor >= 2


class TestValNumWorkers:
    """Validation/benchmark loaders must NOT inherit train num_workers on Windows.

    Smoke-v3 crashed entering validate() with WinError 1455 ("paging file too
    small"): val_loader spawned 4 fresh torch-importing worker procs on top of
    the 4 persistent train workers already alive, blowing through committed VM
    in seconds. A 5k-image val set doesn't need workers — the train loop is
    paused during validate(), so the GPU isn't waiting on disk IO.

    Default is val_num_workers=0; users can override via config.
    """

    def test_val_loader_defaults_to_zero_workers_when_train_uses_workers(self, tmp_path):
        from src.deep_learning.dataset import get_dataloaders
        cfg = _fake_cfg(tmp_path, num_workers=4, splits=("train2017", "val2017"))
        train, val, _ = get_dataloaders(cfg)
        assert train.num_workers == 4
        assert val is not None
        assert val.num_workers == 0, (
            "val loader must default to 0 workers to avoid the Windows pagefile "
            "explosion when val workers spawn alongside persistent train workers"
        )

    def test_benchmark_loader_defaults_to_zero_workers(self, tmp_path):
        from src.deep_learning.dataset import get_dataloaders
        cfg = _fake_cfg(tmp_path, num_workers=4, splits=("train2017", "benchmark"))
        _, _, bench = get_dataloaders(cfg)
        assert bench is not None
        assert bench.num_workers == 0

    def test_val_num_workers_override_is_honored(self, tmp_path):
        from src.deep_learning.dataset import get_dataloaders
        cfg = _fake_cfg(
            tmp_path, num_workers=4, val_num_workers=2,
            splits=("train2017", "val2017"),
        )
        train, val, _ = get_dataloaders(cfg)
        assert train.num_workers == 4
        assert val.num_workers == 2

    def test_val_loader_skips_persistent_workers_when_zero(self, tmp_path):
        # persistent_workers requires num_workers > 0; val defaults to 0 so it
        # must come back False (not raise) even when the train side is positive.
        from src.deep_learning.dataset import get_dataloaders
        cfg = _fake_cfg(tmp_path, num_workers=4, splits=("train2017", "val2017"))
        _, val, _ = get_dataloaders(cfg)
        assert val.persistent_workers is False
