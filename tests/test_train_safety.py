"""Tests for training-loop safety: NaN guard, gradient clipping, batch logging, mid-epoch ckpt.

These tests pin behavior added after the smoke-v2 crash, where loss went NaN at
iter 1222 of epoch 1 and the loop kept running for 8 more hours before OOM'ing
in validation. The fixes here make a re-run impossible to repeat that mistake.
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import cv2
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    cfg["deep_learning"]["epochs"] = 1
    cfg["deep_learning"]["batch_size"] = 2
    cfg["deep_learning"]["loss"] = "huber"
    cfg["deep_learning"]["use_amp"] = False
    cfg["paths"]["data_coco"] = os.path.dirname(tiny_data_dir)
    cfg["paths"]["models_deep"] = os.path.join(tiny_data_dir, "models")
    return cfg


def _make_trainer(tiny_cfg, tiny_data_dir):
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


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


class TestGradientClipping:
    """Trainer must clip gradients to prevent NaN blow-up.

    The smoke-v2 run hit loss=NaN at iter 1222; absent grad clipping, a single
    bad batch can push BN/Conv weights to inf, which then propagates."""

    def test_max_grad_norm_attribute_exists(self, tiny_cfg, tiny_data_dir):
        trainer = _make_trainer(tiny_cfg, tiny_data_dir)
        assert hasattr(trainer, "max_grad_norm"), \
            "Trainer must expose max_grad_norm so training is reproducible across runs"
        assert trainer.max_grad_norm > 0

    def test_config_max_grad_norm_is_honored(self, tiny_cfg, tiny_data_dir):
        tiny_cfg["deep_learning"]["max_grad_norm"] = 2.5
        trainer = _make_trainer(tiny_cfg, tiny_data_dir)
        assert trainer.max_grad_norm == 2.5

    def test_gradients_are_clipped_during_train_epoch(self, tiny_cfg, tiny_data_dir, monkeypatch):
        """clip_grad_norm_ must be called once per batch."""
        from src.deep_learning import train as train_module

        calls = []
        real_clip = torch.nn.utils.clip_grad_norm_

        def spy_clip(params, max_norm, **kw):
            calls.append(float(max_norm))
            return real_clip(params, max_norm, **kw)

        monkeypatch.setattr(train_module.torch.nn.utils, "clip_grad_norm_", spy_clip)
        trainer = _make_trainer(tiny_cfg, tiny_data_dir)
        trainer.train_epoch(1)
        assert len(calls) >= 1, "clip_grad_norm_ must be invoked at least once per epoch"
        assert all(c == trainer.max_grad_norm for c in calls)


# ---------------------------------------------------------------------------
# NaN guard
# ---------------------------------------------------------------------------


class _ExplodingLoss(nn.Module):
    """Loss that returns NaN — used to force the guard to trip."""

    def forward(self, pred, target):
        # Compute a real graph so backward works, then NaN-ify the value.
        base = pred.sum() * 0.0
        return base + float("nan")


class TestNaNGuard:
    """If loss is non-finite for too many consecutive batches, training must abort.

    Previously, the smoke-v2 run continued for 8h after loss became NaN. The guard
    bounds wasted compute to a small number of batches."""

    def test_nan_loss_raises_runtime_error(self, tiny_cfg, tiny_data_dir):
        from src.deep_learning.model import Zhang16Regression
        from src.deep_learning.dataset import ColorizationDataset
        from src.deep_learning.train import Trainer
        from torch.utils.data import DataLoader

        # Tiny dataset has only 4 batches; default patience=5 wouldn't trip.
        tiny_cfg["deep_learning"]["nan_patience"] = 2
        ds = ColorizationDataset(tiny_data_dir, input_size=(64, 64))
        loader = DataLoader(ds, batch_size=2)
        trainer = Trainer(
            Zhang16Regression(), loader, loader,
            _ExplodingLoss(), tiny_cfg, torch.device("cpu"),
        )
        with pytest.raises(RuntimeError, match="(?i)non-finite|nan"):
            trainer.train_epoch(1)


# ---------------------------------------------------------------------------
# Batch-level MLflow logging
# ---------------------------------------------------------------------------


class _RecordingMlflow:
    """Drop-in mlflow stub that records log_metrics calls."""

    def __init__(self):
        self.metric_calls = []

    def log_metrics(self, metrics, step=None):
        self.metric_calls.append((dict(metrics), step))

    # Accept anything else silently
    def __getattr__(self, name):
        def _noop(*a, **kw):
            return None
        return _noop


class TestBatchMlflowLogging:
    """Train loss must be flushed to MLflow during the epoch, not only at the end.

    The smoke-v2 run took 8.5h to finish epoch 1 and crashed in validation BEFORE
    any epoch-end MLflow logging, leaving zero metrics — completely undebuggable.
    Per-batch logging means we always have a curve to inspect."""

    def test_train_epoch_logs_batch_loss(self, tiny_cfg, tiny_data_dir, monkeypatch):
        from src.deep_learning import train as train_module

        rec = _RecordingMlflow()
        monkeypatch.setattr(train_module, "mlflow", rec)

        tiny_cfg["deep_learning"]["log_every_n_batches"] = 1
        trainer = _make_trainer(tiny_cfg, tiny_data_dir)
        trainer.train_epoch(1)

        batch_calls = [c for c in rec.metric_calls if "batch_train_loss" in c[0]]
        assert batch_calls, \
            "train_epoch must log per-batch train loss to MLflow during the epoch"


# ---------------------------------------------------------------------------
# Mid-epoch checkpointing
# ---------------------------------------------------------------------------


class TestMidEpochCheckpoint:
    """A long epoch must periodically save mid-epoch checkpoints.

    Smoke-v2 burned 8.5h on epoch 1 then crashed → zero checkpoints. With this
    guard, the worst-case loss is the work done since the last interval save."""

    def test_mid_epoch_checkpoint_is_written(self, tiny_cfg, tiny_data_dir):
        # 8 images / batch_size 2 = 4 batches; save every 2 batches → ≥1 save.
        tiny_cfg["deep_learning"]["checkpoint_every_n_batches"] = 2
        trainer = _make_trainer(tiny_cfg, tiny_data_dir)
        trainer.train_epoch(1)
        path = os.path.join(trainer.save_dir, "in_progress.pth")
        assert os.path.exists(path), \
            "mid-epoch checkpoint must be written to in_progress.pth"

    def test_zero_interval_disables_mid_epoch_save(self, tiny_cfg, tiny_data_dir):
        tiny_cfg["deep_learning"]["checkpoint_every_n_batches"] = 0
        trainer = _make_trainer(tiny_cfg, tiny_data_dir)
        trainer.train_epoch(1)
        path = os.path.join(trainer.save_dir, "in_progress.pth")
        assert not os.path.exists(path), \
            "checkpoint_every_n_batches=0 must skip the mid-epoch save"


# ---------------------------------------------------------------------------
# Validation resilience
# ---------------------------------------------------------------------------


class _NaNModel(nn.Module):
    """Model whose forward output is all NaN — to test validate() resilience."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out * float("nan")

    def predict_ab(self, L, quantizer=None, temperature=None):
        return self.forward(L) * 110.0


class TestMaxTrainBatches:
    """Caps batches per epoch so smoke tests don't take 9h on real data."""

    def test_max_train_batches_caps_iteration(self, tiny_cfg, tiny_data_dir, monkeypatch):
        from src.deep_learning import train as train_module

        tiny_cfg["deep_learning"]["max_train_batches"] = 2
        trainer = _make_trainer(tiny_cfg, tiny_data_dir)

        calls = []
        real_clip = torch.nn.utils.clip_grad_norm_

        def spy_clip(params, max_norm, **kw):
            calls.append(1)
            return real_clip(params, max_norm, **kw)

        monkeypatch.setattr(train_module.torch.nn.utils, "clip_grad_norm_", spy_clip)
        trainer.train_epoch(1)
        # Tiny dataset has 4 batches; cap of 2 must stop after 2.
        assert len(calls) == 2

    def test_zero_means_unlimited(self, tiny_cfg, tiny_data_dir, monkeypatch):
        from src.deep_learning import train as train_module

        tiny_cfg["deep_learning"]["max_train_batches"] = 0
        trainer = _make_trainer(tiny_cfg, tiny_data_dir)

        calls = []
        real_clip = torch.nn.utils.clip_grad_norm_
        monkeypatch.setattr(
            train_module.torch.nn.utils, "clip_grad_norm_",
            lambda p, n, **kw: (calls.append(1), real_clip(p, n, **kw))[1],
        )
        trainer.train_epoch(1)
        assert len(calls) == 4  # all 4 batches consumed


class _OutlierLoss(nn.Module):
    """Loss returning a HUGE-but-finite value — to exercise the outlier filter.

    Distinct from _ExplodingLoss (which returns NaN). 2.4M is the actual
    epoch-7 outlier seen in Phase 3; that single batch poisoned the val_loss
    mean and prevented best-model save even though PSNR was higher than
    the previous best. The outlier filter prevents that regression.
    """

    def forward(self, pred, target):
        base = pred.sum() * 0.0
        return base + 2_400_000.0


class TestValidateOutlierFilter:
    def test_outlier_batches_are_excluded_from_val_loss(self, tiny_cfg, tiny_data_dir):
        from src.deep_learning.model import Zhang16Regression
        from src.deep_learning.dataset import ColorizationDataset
        from src.deep_learning.train import Trainer
        from torch.utils.data import DataLoader

        ds = ColorizationDataset(tiny_data_dir, input_size=(64, 64))
        loader = DataLoader(ds, batch_size=2)
        trainer = Trainer(
            Zhang16Regression(), loader, loader,
            _OutlierLoss(), tiny_cfg, torch.device("cpu"),
        )
        avg_loss, _ = trainer.validate()
        # Every batch returns 2.4M — they should ALL be excluded as outliers,
        # leaving num_batches=0 and avg_loss=nan (the no-valid-samples sentinel).
        assert math.isnan(avg_loss), \
            "huge-but-finite loss batches must be dropped, not averaged in"


class TestValidateResilientToNaN:
    """validate() must not crash when the model output is NaN.

    During smoke-v2, validation tried to call lab2rgb on a NaN ab tensor and the
    process eventually OOM'd. Returning a sentinel keeps the loop alive long
    enough to log + checkpoint the epoch summary."""

    def test_validate_with_nan_output_returns_without_raising(self, tiny_cfg, tiny_data_dir):
        from src.deep_learning.loss import HuberColorLoss
        from src.deep_learning.dataset import ColorizationDataset
        from src.deep_learning.train import Trainer
        from torch.utils.data import DataLoader

        tiny_cfg["deep_learning"]["loss"] = "huber"
        ds = ColorizationDataset(tiny_data_dir, input_size=(64, 64))
        loader = DataLoader(ds, batch_size=2)
        trainer = Trainer(
            _NaNModel(), loader, loader,
            HuberColorLoss(), tiny_cfg, torch.device("cpu"),
        )
        # Must NOT raise even though every forward pass produces NaN ab values.
        avg_loss, metrics = trainer.validate()
        assert "psnr" in metrics and "ssim" in metrics
        # PSNR/SSIM should be the sentinel for "no valid samples".
        assert math.isnan(metrics["psnr"]) or metrics["psnr"] == 0.0
