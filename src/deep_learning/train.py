"""
Training loop for deep learning colorization.

Provides a Trainer class that handles:
- Training epochs with progress tracking
- Validation with metric computation
- MLflow experiment logging
- Checkpoint saving/loading
- Mixed precision training for memory efficiency
"""

import os
import math
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import mlflow
except ImportError:
    mlflow = None

from .utils import denormalize_L, denormalize_ab, lab_to_rgb
from .quantize import ABQuantizer


# Validation aggregation outlier bound.
#
# Observed during Phase 3: one batch in 1250 val batches of epoch 7 produced
# loss=2.4e6 (a finite-but-pathological number), inflating the epoch mean
# to ~2.4M and preventing best-checkpoint save even though that epoch's
# PSNR/SSIM were higher than the prior best. CE with 233 bins is naturally
# bounded around log(233) * max_class_weight ~= 22, so any batch over a
# few hundred is unambiguously an outlier we want to drop, not average in.
_VAL_LOSS_OUTLIER_BOUND = 1_000.0


class Trainer:
    """Handles model training, validation, and experiment tracking.

    Args:
        model: nn.Module (Zhang16Net or Zhang16Regression).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_fn: nn.Module loss function.
        cfg: Full config dict.
        device: torch.device to train on.
        quantizer: ABQuantizer instance (for classification model decoding).
    """

    def __init__(self, model, train_loader, val_loader, loss_fn, cfg, device, quantizer=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device) if hasattr(loss_fn, 'to') else loss_fn
        self.cfg = cfg
        self.device = device
        self.quantizer = quantizer

        dl_cfg = cfg["deep_learning"]

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=dl_cfg["learning_rate"],
            betas=tuple(dl_cfg.get("betas", [0.9, 0.999])),
            weight_decay=dl_cfg.get("weight_decay", 0.0),
        )

        # Scheduler
        scheduler_type = dl_cfg.get("scheduler", "step")
        if scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=dl_cfg.get("scheduler_step_size", 20),
                gamma=dl_cfg.get("scheduler_gamma", 0.5),
            )
        elif scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=dl_cfg["epochs"]
            )
        else:
            self.scheduler = None

        # Mixed precision
        self.use_amp = dl_cfg.get("use_amp", False) and device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Loss type
        self.loss_type = dl_cfg.get("loss", "cross_entropy")
        self.temperature = dl_cfg.get("temperature", 0.38)

        # Safety / observability knobs — defaults aim for "training survives 9h
        # epochs on slow GPUs without losing work or going silently NaN".
        # max_grad_norm: clip gradients to bound BN/Conv weight blow-up; without
        #   it the smoke-v2 run went loss=NaN at iter 1222 and kept churning.
        # nan_patience: abort if loss is non-finite this many batches in a row.
        # log_every_n_batches: per-batch MLflow logging cadence so a crash mid-
        #   epoch still leaves a debuggable curve.
        # checkpoint_every_n_batches: persist `in_progress.pth` every N batches;
        #   set 0 to disable.
        self.max_grad_norm = float(dl_cfg.get("max_grad_norm", 1.0))
        self.nan_patience = int(dl_cfg.get("nan_patience", 5))
        self.log_every_n_batches = int(dl_cfg.get("log_every_n_batches", 50))
        self.checkpoint_every_n_batches = int(dl_cfg.get("checkpoint_every_n_batches", 500))
        # max_train_batches: 0 = unlimited. Caps batches/epoch for smoke tests.
        self.max_train_batches = int(dl_cfg.get("max_train_batches", 0))
        self._global_step = 0

        # Tracking
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}
        # Number of epochs completed before this fit() call — set by load_checkpoint().
        # Used to offset the epoch counter so MLflow step numbers stay monotonic
        # across multi-session resumed training (1 epoch/session on slow GPUs).
        self.completed_epochs = 0

        # Paths
        self.save_dir = cfg["paths"].get("models_deep", "models/deep_learning")
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        """Run one training epoch with grad clipping, NaN guard, and batch logging.

        Returns:
            float: Average training loss for the epoch.

        Raises:
            RuntimeError: If loss is non-finite for ``self.nan_patience`` batches
                in a row. We abort rather than churn — the prior smoke-v2 run lost
                8h after loss went NaN with no early stop.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        nan_streak = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if self.max_train_batches and batch_idx >= self.max_train_batches:
                break
            L = batch["L"].to(self.device)
            ab = batch["ab"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    output = self.model(L)
                    loss = self.loss_fn(output, ab)
                self.scaler.scale(loss).backward()
                # Unscale before clipping so max_grad_norm is in real units.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(L)
                loss = self.loss_fn(output, ab)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                nan_streak += 1
                if nan_streak >= self.nan_patience:
                    raise RuntimeError(
                        f"Training aborted: loss has been non-finite for "
                        f"{nan_streak} consecutive batches (epoch {epoch}, "
                        f"batch {batch_idx}). Last loss={loss_val}. "
                        f"Lower learning_rate, raise max_grad_norm scrutiny, or "
                        f"disable AMP and retry."
                    )
            else:
                nan_streak = 0
                total_loss += loss_val
                num_batches += 1

            self._global_step += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            # Per-batch MLflow logging — keeps a curve even if the epoch crashes.
            if (
                self.log_every_n_batches > 0
                and (batch_idx + 1) % self.log_every_n_batches == 0
                and mlflow is not None
            ):
                try:
                    mlflow.log_metrics(
                        {
                            "batch_train_loss": loss_val,
                            "batch_lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self._global_step,
                    )
                except Exception:
                    pass

            # Mid-epoch checkpoint — bounds work lost to a crash at one interval.
            if (
                self.checkpoint_every_n_batches > 0
                and (batch_idx + 1) % self.checkpoint_every_n_batches == 0
            ):
                self.save_checkpoint(
                    os.path.join(self.save_dir, "in_progress.pth"), epoch
                )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Run validation and compute metrics.

        Skips image-metric computation for any batch whose forward output
        contains NaN/Inf — a NaN-poisoned model used to crash this loop
        downstream in skimage.lab2rgb after exhausting host memory.

        Returns:
            Tuple of (avg_loss, metrics_dict) where metrics_dict has psnr, ssim.
            When every batch is NaN, psnr/ssim come back as ``float('nan')``
            instead of a misleading 0.0.
        """
        if self.val_loader is None:
            return 0.0, {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_psnr = []
        all_ssim = []

        # Any per-batch loss > this bound is treated as an outlier — see the
        # docstring of `_VAL_LOSS_OUTLIER_BOUND` for why this matters.
        outlier_bound = _VAL_LOSS_OUTLIER_BOUND

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            L = batch["L"].to(self.device)
            ab = batch["ab"].to(self.device)

            output = self.model(L)
            loss = self.loss_fn(output, ab)

            loss_val = loss.item()
            # Cap pathological-but-finite batches. ClassRebalancedCELoss with
            # 233 bins is theoretically bounded around log(233) * max_class_weight
            # ~= 22, but we've observed single batches hitting 1e6+ (probably
            # one image whose ab distribution drives the rebalance weight chain
            # into a numerical edge case). One such batch in 1250 val batches
            # poisons the mean by ~2000, flipping best-model selection even
            # when PSNR/SSIM say the epoch is actually better. We drop it.
            if math.isfinite(loss_val) and loss_val <= outlier_bound:
                total_loss += loss_val
                num_batches += 1

            # Skip per-image metrics if forward output is NaN/Inf — the smoke-v2
            # crash chain started here (NaN ab -> lab2rgb -> RAM exhaustion).
            if not torch.isfinite(output).all():
                continue

            # Compute image metrics on a subset (first 2 images per batch)
            n_eval = min(2, L.shape[0])
            for i in range(n_eval):
                L_single = L[i:i+1]
                ab_target = ab[i:i+1]

                # Get predicted ab at full resolution — reuse `output` instead
                # of calling forward() again per image.
                if self.loss_type == "cross_entropy" and self.quantizer is not None:
                    ab_pred = self._decode_logits_to_ab(output[i:i+1], L_single.shape)
                else:
                    _, _, H, W = L_single.shape
                    ab_pred_small = output[i:i+1]
                    ab_pred = F.interpolate(ab_pred_small, size=(H, W), mode="bilinear", align_corners=False)
                    ab_pred = ab_pred * 110.0  # always denormalize regression output

                if not torch.isfinite(ab_pred).all():
                    continue

                # Convert to RGB for metrics
                L_np = denormalize_L(L_single[0, 0].cpu().numpy())
                ab_pred_np = ab_pred[0].permute(1, 2, 0).cpu().numpy()
                ab_gt_np = denormalize_ab(ab_target[0].permute(1, 2, 0).cpu().numpy())

                pred_rgb = lab_to_rgb(L_np, ab_pred_np)
                gt_rgb = lab_to_rgb(L_np, ab_gt_np)

                psnr = peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=255)
                ssim = structural_similarity(gt_rgb, pred_rgb, channel_axis=2, data_range=255)
                all_psnr.append(psnr)
                all_ssim.append(ssim)

        avg_loss = total_loss / max(num_batches, 1) if num_batches > 0 else float("nan")
        metrics = {
            "psnr": float(np.mean(all_psnr)) if all_psnr else float("nan"),
            "ssim": float(np.mean(all_ssim)) if all_ssim else float("nan"),
        }
        return avg_loss, metrics

    def _decode_logits_to_ab(self, logits, full_shape):
        """Decode classification logits to ab tensor via annealed-mean.

        Mirrors `Zhang16Net.predict_ab` but reuses precomputed logits so we
        don't run forward() twice per validation image.

        Args:
            logits: (1, Q, H', W') logits from one image.
            full_shape: (1, 1, H, W) tuple — target full resolution.

        Returns:
            (1, 2, H, W) ab tensor in [-110, 110].
        """
        _, _, H, W = full_shape
        probs = F.softmax(logits, dim=1)
        probs_full = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
        ab_bins = torch.from_numpy(self.quantizer.ab_bins).float().to(logits.device)
        log_probs = torch.log(probs_full + 1e-8)
        annealed = torch.exp(log_probs / self.temperature)
        annealed = annealed / (annealed.sum(dim=1, keepdim=True) + 1e-8)
        B, Q, H_out, W_out = annealed.shape
        flat = annealed.permute(0, 2, 3, 1).reshape(-1, Q)
        ab_pred = flat @ ab_bins
        return ab_pred.reshape(B, H_out, W_out, 2).permute(0, 3, 1, 2)

    def fit(self, num_epochs=None):
        """Full training loop with MLflow logging.

        Honors ``self.completed_epochs`` (set by ``load_checkpoint``) so the
        global epoch counter is monotonic across resumed sessions — useful
        for ``--epochs 1`` runs that resume the previous checkpoint to do one
        more epoch at a time.

        Args:
            num_epochs: Number of NEW epochs to train this call. If None,
                uses the config value.
        """
        if num_epochs is None:
            num_epochs = self.cfg["deep_learning"]["epochs"]

        start_epoch = self.completed_epochs
        total_epochs = start_epoch + num_epochs

        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch} -> training epochs {start_epoch + 1}..{total_epochs}")
        print(f"Training on {self.device} for {num_epochs} epoch(s) this run")
        print(f"Loss: {self.loss_type} | AMP: {self.use_amp}")
        if self.train_loader:
            print(f"Train batches: {len(self.train_loader)} | Batch size: {self.cfg['deep_learning']['batch_size']}")

        for local_epoch in range(1, num_epochs + 1):
            epoch = start_epoch + local_epoch
            start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate()
            self.history["val_loss"].append(val_loss)
            self.history["val_psnr"].append(val_metrics.get("psnr", 0.0))
            self.history["val_ssim"].append(val_metrics.get("ssim", 0.0))

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"PSNR: {val_metrics.get('psnr', 0):.2f} | "
                f"SSIM: {val_metrics.get('ssim', 0):.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # MLflow logging — step=epoch (global) so resumed sessions append
            if mlflow is not None:
                try:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_psnr": val_metrics.get("psnr", 0.0),
                        "val_ssim": val_metrics.get("ssim", 0.0),
                        "learning_rate": lr,
                    }, step=epoch)
                except Exception:
                    pass  # MLflow not active, skip

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(self.save_dir, "best_model.pth"), epoch)
                print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

            # Track progress for any subsequent call to fit() in the same process
            self.completed_epochs = epoch

        # Save final model
        self.save_checkpoint(os.path.join(self.save_dir, "last_model.pth"), self.completed_epochs)

        # Save training history (full accumulated history, not just this session)
        history_path = os.path.join(self.save_dir, "training_log.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, path, epoch):
        """Save model checkpoint, including LR scheduler state and history.

        Including scheduler_state_dict and history lets `--resume` continue
        the LR schedule cleanly across sessions and accumulate per-epoch
        training metrics in `training_log.json` instead of overwriting them.
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg["deep_learning"],
            "history": self.history,
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path):
        """Load model checkpoint.

        Restores model + optimizer state, best_val_loss, scheduler state (if
        present), and per-epoch history (if present). Missing keys are
        tolerated so legacy checkpoints from earlier runs still load.

        Returns the epoch number stored in the checkpoint (used by fit() to
        offset the epoch counter for MLflow step numbering).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        # Restore scheduler (new field; old checkpoints don't have it)
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Restore history (new field; old checkpoints don't have it)
        saved_history = checkpoint.get("history")
        if isinstance(saved_history, dict):
            # Merge to keep any default keys the current Trainer expects
            for k, v in saved_history.items():
                self.history[k] = list(v)
        epoch = int(checkpoint.get("epoch", 0))
        self.completed_epochs = epoch
        return epoch
