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

        # Tracking
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": []}

        # Paths
        self.save_dir = cfg["paths"].get("models_deep", "models/deep_learning")
        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self, epoch):
        """Run one training epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            L = batch["L"].to(self.device)
            ab = batch["ab"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    output = self.model(L)
                    loss = self.loss_fn(output, ab)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(L)
                loss = self.loss_fn(output, ab)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Run validation and compute metrics.

        Returns:
            Tuple of (avg_loss, metrics_dict) where metrics_dict has psnr, ssim.
        """
        if self.val_loader is None:
            return 0.0, {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_psnr = []
        all_ssim = []

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            L = batch["L"].to(self.device)
            ab = batch["ab"].to(self.device)

            output = self.model(L)
            loss = self.loss_fn(output, ab)
            total_loss += loss.item()
            num_batches += 1

            # Compute image metrics on a subset (first 2 images per batch)
            n_eval = min(2, L.shape[0])
            for i in range(n_eval):
                L_single = L[i:i+1]
                ab_target = ab[i:i+1]

                # Get predicted ab at full resolution
                if self.loss_type == "cross_entropy" and self.quantizer is not None:
                    ab_pred = self.model.predict_ab(L_single, self.quantizer, self.temperature)
                else:
                    _, _, H, W = L_single.shape
                    ab_pred_small = output[i:i+1]
                    ab_pred = F.interpolate(ab_pred_small, size=(H, W), mode="bilinear", align_corners=False)
                    ab_pred = ab_pred * 110.0  # always denormalize regression output

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

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {
            "psnr": float(np.mean(all_psnr)) if all_psnr else 0.0,
            "ssim": float(np.mean(all_ssim)) if all_ssim else 0.0,
        }
        return avg_loss, metrics

    def fit(self, num_epochs=None):
        """Full training loop with MLflow logging.

        Args:
            num_epochs: Number of epochs to train. If None, uses config value.
        """
        if num_epochs is None:
            num_epochs = self.cfg["deep_learning"]["epochs"]

        print(f"Training on {self.device} for {num_epochs} epochs")
        print(f"Loss: {self.loss_type} | AMP: {self.use_amp}")
        if self.train_loader:
            print(f"Train batches: {len(self.train_loader)} | Batch size: {self.cfg['deep_learning']['batch_size']}")

        for epoch in range(1, num_epochs + 1):
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
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"PSNR: {val_metrics.get('psnr', 0):.2f} | "
                f"SSIM: {val_metrics.get('ssim', 0):.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # MLflow logging
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

        # Save final model
        self.save_checkpoint(os.path.join(self.save_dir, "last_model.pth"), num_epochs)

        # Save training history
        history_path = os.path.join(self.save_dir, "training_log.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"Training complete. Best val_loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, path, epoch):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.cfg["deep_learning"],
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return checkpoint.get("epoch", 0)
