"""
Train / fine-tune the deep learning colorization model.

Usage:
    python tools/train_deep.py --config configs/config.yaml
    python tools/train_deep.py --epochs 10 --batch-size 4 --lr 0.0001
    python tools/train_deep.py --resume models/deep_learning/last_model.pth

Trains the Zhang 2016 model on COCO 2017 with MLflow experiment tracking.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:
    import mlflow
except ImportError:
    mlflow = None

from src.deep_learning.utils import load_config, get_device
from src.deep_learning.quantize import ABQuantizer
from src.deep_learning.model import build_model, Zhang16Net, load_zhang16_eccv_weights
from src.deep_learning.loss import build_loss
from src.deep_learning.dataset import get_dataloaders
from src.deep_learning.train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train deep colorization model")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", default=None, help="Device: cuda, cpu, or auto")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--experiment-name", default="deep-colorization",
                        help="MLflow experiment name")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Cap batches per epoch (0 = no cap). Useful for smoke tests.")
    args = parser.parse_args()

    # Load and override config
    cfg = load_config(args.config)
    dl_cfg = cfg["deep_learning"]

    if args.epochs is not None:
        dl_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        dl_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        dl_cfg["learning_rate"] = args.lr
    if args.max_train_batches is not None:
        dl_cfg["max_train_batches"] = args.max_train_batches

    # Device
    if args.device:
        cfg["device"] = args.device
    device = get_device(cfg)
    print(f"Device: {device}")

    # Quantizer
    quantizer = ABQuantizer()
    print(f"Ab quantization: {quantizer.num_bins} bins")

    # Model
    model = build_model(cfg, quantizer=quantizer)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__} ({total_params:,} params)")

    # Load pretrained weights for fine-tuning
    pretrained_path = dl_cfg.get("pretrained_weights")
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            if isinstance(model, Zhang16Net):
                summary = load_zhang16_eccv_weights(model, pretrained_path, device=device)
                print(
                    f"Loaded {summary['loaded']} weight keys from {pretrained_path}; "
                    f"dropped {len(summary['dropped'])}, missing {len(summary['missing'])} "
                    f"(head + any architecture-mismatched layers stay random-init)."
                )
            else:
                # Fallback for non-Zhang16Net architectures: legacy strict=False load.
                checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
                sd = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(sd, strict=False)
                print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    # Data — load first so the loss factory can compute empirical class weights.
    train_loader, val_loader, _ = get_dataloaders(cfg)
    if train_loader is None:
        print("ERROR: No training data found. Run 'python tools/download_coco.py' first.")
        sys.exit(1)
    print(f"Train: {len(train_loader.dataset)} images, Val: {len(val_loader.dataset) if val_loader else 0} images")

    # Loss — pass the train dataset so ClassRebalancedCELoss gets a real
    # empirical ab distribution (not the silent-uniform fallback).
    loss_fn = build_loss(cfg, quantizer=quantizer, train_dataset=train_loader.dataset)
    print(f"Loss: {type(loss_fn).__name__}")

    # MLflow
    if mlflow is not None:
        mlflow.set_experiment(args.experiment_name)
        run = mlflow.start_run()
        mlflow.log_params({
            "model": type(model).__name__,
            "loss": type(loss_fn).__name__,
            "epochs": dl_cfg["epochs"],
            "batch_size": dl_cfg["batch_size"],
            "learning_rate": dl_cfg["learning_rate"],
            "optimizer": dl_cfg.get("optimizer", "adam"),
            "num_classes": quantizer.num_bins,
            "device": str(device),
            "use_amp": dl_cfg.get("use_amp", False),
        })
        print(f"MLflow run: {run.info.run_id}")

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, loss_fn, cfg, device, quantizer=quantizer)

    # Resume from checkpoint
    if args.resume:
        epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {epoch}")

    # Train
    trainer.fit()

    # End MLflow run
    if mlflow is not None:
        # Log final model as artifact
        best_path = os.path.join(cfg["paths"]["models_deep"], "best_model.pth")
        if os.path.exists(best_path):
            mlflow.log_artifact(best_path)
        mlflow.end_run()

    print("Training complete!")


if __name__ == "__main__":
    main()
