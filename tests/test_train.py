"""Tests for Trainer - training loop, checkpoint save/load."""

import os
import torch
import numpy as np
import cv2
import pytest


@pytest.fixture
def tiny_data_dir(tmp_path):
    """Create a tiny dataset for smoke testing."""
    for i in range(8):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i}.jpg"), img)
    return str(tmp_path)


@pytest.fixture
def tiny_cfg(cfg, tiny_data_dir):
    """Config pointing to tiny data for fast tests."""
    cfg["deep_learning"]["epochs"] = 2
    cfg["deep_learning"]["batch_size"] = 2
    cfg["deep_learning"]["loss"] = "huber"
    cfg["deep_learning"]["use_amp"] = False
    cfg["paths"]["data_coco"] = os.path.dirname(tiny_data_dir)
    cfg["paths"]["models_deep"] = os.path.join(tiny_data_dir, "models")
    return cfg


class TestTrainerInit:
    """Test Trainer construction."""

    def test_creates_with_huber_loss(self, tiny_cfg, tiny_data_dir):
        from src.deep_learning.model import Zhang16Regression
        from src.deep_learning.loss import HuberColorLoss
        from src.deep_learning.dataset import ColorizationDataset
        from src.deep_learning.train import Trainer
        from torch.utils.data import DataLoader

        model = Zhang16Regression()
        loss_fn = HuberColorLoss()
        ds = ColorizationDataset(tiny_data_dir, input_size=(64, 64))
        loader = DataLoader(ds, batch_size=2)

        trainer = Trainer(model, loader, loader, loss_fn, tiny_cfg, torch.device("cpu"))
        assert trainer.model is not None
        assert trainer.optimizer is not None


class TestTrainerTrainEpoch:
    """Test single training epoch."""

    def test_returns_positive_loss(self, tiny_cfg, tiny_data_dir):
        from src.deep_learning.model import Zhang16Regression
        from src.deep_learning.loss import HuberColorLoss
        from src.deep_learning.dataset import ColorizationDataset
        from src.deep_learning.train import Trainer
        from torch.utils.data import DataLoader

        model = Zhang16Regression()
        loss_fn = HuberColorLoss()
        ds = ColorizationDataset(tiny_data_dir, input_size=(64, 64))
        loader = DataLoader(ds, batch_size=2)

        trainer = Trainer(model, loader, loader, loss_fn, tiny_cfg, torch.device("cpu"))
        loss = trainer.train_epoch(1)
        assert loss > 0
        assert isinstance(loss, float)


class TestTrainerCheckpoint:
    """Test checkpoint save/load roundtrip."""

    def test_save_and_load(self, tiny_cfg, tiny_data_dir, tmp_path):
        from src.deep_learning.model import Zhang16Regression
        from src.deep_learning.loss import HuberColorLoss
        from src.deep_learning.dataset import ColorizationDataset
        from src.deep_learning.train import Trainer
        from torch.utils.data import DataLoader

        model = Zhang16Regression()
        loss_fn = HuberColorLoss()
        ds = ColorizationDataset(tiny_data_dir, input_size=(64, 64))
        loader = DataLoader(ds, batch_size=2)

        trainer = Trainer(model, loader, loader, loss_fn, tiny_cfg, torch.device("cpu"))
        trainer.best_val_loss = 0.42

        # Save
        ckpt_path = str(tmp_path / "test_ckpt.pth")
        trainer.save_checkpoint(ckpt_path, epoch=5)
        assert os.path.exists(ckpt_path)

        # Load into fresh trainer
        model2 = Zhang16Regression()
        trainer2 = Trainer(model2, loader, loader, loss_fn, tiny_cfg, torch.device("cpu"))
        epoch = trainer2.load_checkpoint(ckpt_path)

        assert epoch == 5
        assert trainer2.best_val_loss == 0.42

        # Model weights should match
        for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
            assert torch.equal(p1, p2)
