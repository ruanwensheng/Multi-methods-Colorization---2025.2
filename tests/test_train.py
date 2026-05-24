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


class TestTrainerCheckpointResume:
    """Test multi-epoch resume: scheduler state + history must survive the round-trip.

    Without these, resuming from `last_model.pth` between epochs would silently
    reset the LR schedule and clobber the per-epoch training_log.json — both
    matter when the user is training 1 epoch at a time across multiple sessions
    (a workflow we use because each epoch on GTX 1660 SUPER takes ~8.5 h).
    """

    def _make_trainer(self, tiny_cfg, tiny_data_dir):
        from src.deep_learning.model import Zhang16Regression
        from src.deep_learning.loss import HuberColorLoss
        from src.deep_learning.dataset import ColorizationDataset
        from src.deep_learning.train import Trainer
        from torch.utils.data import DataLoader

        model = Zhang16Regression()
        loss_fn = HuberColorLoss()
        ds = ColorizationDataset(tiny_data_dir, input_size=(64, 64))
        loader = DataLoader(ds, batch_size=2)
        return Trainer(model, loader, loader, loss_fn, tiny_cfg, torch.device("cpu"))

    def test_checkpoint_contains_scheduler_state(self, tiny_cfg, tiny_data_dir, tmp_path):
        trainer = self._make_trainer(tiny_cfg, tiny_data_dir)
        ckpt_path = str(tmp_path / "ckpt.pth")
        trainer.save_checkpoint(ckpt_path, epoch=3)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # StepLR has a state_dict — we should be saving it.
        assert "scheduler_state_dict" in ckpt, \
            "save_checkpoint must persist scheduler state for clean LR-schedule resume"

    def test_checkpoint_contains_history(self, tiny_cfg, tiny_data_dir, tmp_path):
        trainer = self._make_trainer(tiny_cfg, tiny_data_dir)
        # Pretend we ran 2 epochs already
        trainer.history["train_loss"] = [1.2, 0.9]
        trainer.history["val_loss"] = [1.1, 0.85]
        ckpt_path = str(tmp_path / "ckpt.pth")
        trainer.save_checkpoint(ckpt_path, epoch=2)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "history" in ckpt
        assert ckpt["history"]["train_loss"] == [1.2, 0.9]

    def test_load_restores_scheduler_last_epoch(self, tiny_cfg, tiny_data_dir, tmp_path):
        trainer = self._make_trainer(tiny_cfg, tiny_data_dir)
        # Advance scheduler so last_epoch != 0
        trainer.scheduler.step()
        trainer.scheduler.step()
        trainer.scheduler.step()
        expected_last_epoch = trainer.scheduler.last_epoch
        assert expected_last_epoch == 3

        ckpt_path = str(tmp_path / "ckpt.pth")
        trainer.save_checkpoint(ckpt_path, epoch=3)

        trainer2 = self._make_trainer(tiny_cfg, tiny_data_dir)
        # Fresh trainer's scheduler starts at last_epoch=0
        assert trainer2.scheduler.last_epoch == 0
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.scheduler.last_epoch == expected_last_epoch

    def test_load_restores_history(self, tiny_cfg, tiny_data_dir, tmp_path):
        trainer = self._make_trainer(tiny_cfg, tiny_data_dir)
        trainer.history["train_loss"] = [3.0, 2.5]
        trainer.history["val_loss"] = [2.8, 2.4]
        trainer.history["val_psnr"] = [12.0, 14.5]
        trainer.history["val_ssim"] = [0.5, 0.6]
        ckpt_path = str(tmp_path / "ckpt.pth")
        trainer.save_checkpoint(ckpt_path, epoch=2)

        trainer2 = self._make_trainer(tiny_cfg, tiny_data_dir)
        assert trainer2.history["train_loss"] == []
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.history["train_loss"] == [3.0, 2.5]
        assert trainer2.history["val_psnr"] == [12.0, 14.5]

    def test_load_handles_legacy_checkpoint_without_scheduler_or_history(self, tiny_cfg, tiny_data_dir, tmp_path):
        # The smoke-v2 run currently in flight uses the OLD checkpoint format
        # (no scheduler/history keys). Resuming from it must not crash.
        trainer = self._make_trainer(tiny_cfg, tiny_data_dir)
        legacy_ckpt = {
            "epoch": 1,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "best_val_loss": 0.5,
            "config": tiny_cfg["deep_learning"],
        }
        ckpt_path = str(tmp_path / "legacy.pth")
        torch.save(legacy_ckpt, ckpt_path)

        trainer2 = self._make_trainer(tiny_cfg, tiny_data_dir)
        # Should not raise on missing scheduler_state_dict / history
        epoch = trainer2.load_checkpoint(ckpt_path)
        assert epoch == 1
        assert trainer2.best_val_loss == 0.5
