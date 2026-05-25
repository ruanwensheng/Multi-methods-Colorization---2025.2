"""
DeepColorizer - inference wrapper for deep learning colorization.

Provides a unified API matching the scribble pipeline's interface:
    colorize(gray_img) -> (result_bgr, info_dict)
"""

import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .model import Zhang16Net, Zhang16Regression, build_model, load_zhang16_eccv_weights
from .quantize import ABQuantizer
from .utils import bgr_to_lab, lab_to_bgr, normalize_L, denormalize_L, denormalize_ab


class DeepColorizer:
    """Deep learning image colorizer using Zhang et al. 2016.

    Wraps model inference with preprocessing and postprocessing to provide
    a clean API that accepts grayscale images and returns colorized results.

    Args:
        model_path: Path to model checkpoint (.pth file). If None, uses random weights.
        cfg: Config dict. If None, uses defaults.
        device: "cuda", "cpu", or "auto".
        temperature: Annealing temperature for classification model decoding.
    """

    def __init__(self, model_path=None, cfg=None, device="auto", temperature=0.38):
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.temperature = temperature
        self.input_size = (256, 256)

        # Determine model type
        if cfg is not None:
            self.loss_type = cfg["deep_learning"].get("loss", "cross_entropy")
            self.input_size = tuple(cfg["image"].get("input_size", [256, 256]))
        else:
            self.loss_type = "cross_entropy"

        # Quantizer (for classification model)
        self.quantizer = ABQuantizer() if self.loss_type == "cross_entropy" else None

        # Build model
        if cfg:
            self.model = build_model(cfg, quantizer=self.quantizer)
        elif self.loss_type == "huber":
            self.model = Zhang16Regression()
        else:
            num_classes = self.quantizer.num_bins if self.quantizer else 313
            self.model = Zhang16Net(num_classes=num_classes)

        # Load weights — auto-detect checkpoint format:
        #   1. Our training checkpoint: {"model_state_dict": ..., "epoch": ...}
        #   2. Official Zhang16 ECCV checkpoint: raw dict with `modelN.*` keys
        #   3. Bare state_dict with our `convN.*` keys
        # Without (2), trying to evaluate the pretrained ECCV baseline (T3.1)
        # would crash on strict-load mismatch; the remapper handles the
        # `modelN -> convN` rename + shape filtering.
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            sd = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if isinstance(sd, dict) and any(k.startswith("model") and "." in k for k in sd.keys()):
                summary = load_zhang16_eccv_weights(self.model, model_path, device=self.device)
                print(
                    f"Loaded {summary['loaded']} ECCV weight keys from {model_path}; "
                    f"dropped {len(summary['dropped'])}, missing {len(summary['missing'])}"
                )
            else:
                self.model.load_state_dict(sd)
                print(f"Loaded weights from {model_path}")

        self.model = self.model.to(self.device)
        self.model.eval()

    def colorize(self, gray_image):
        """Colorize a grayscale image.

        Args:
            gray_image: np.ndarray, either:
                - (H, W) single-channel grayscale
                - (H, W, 3) BGR grayscale (all channels equal)
                - (H, W, 3) BGR image (L channel will be extracted)

        Returns:
            result_bgr: np.ndarray (H, W, 3) uint8 BGR colorized image.
            info_dict: dict with metadata (elapsed_sec, model_name, device, etc.)
        """
        start = time.time()

        # Handle input formats
        if gray_image.ndim == 2:
            # Single channel -> make 3-channel BGR
            bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = gray_image

        orig_h, orig_w = bgr.shape[:2]

        # Convert to Lab and extract L
        L_orig, _ = bgr_to_lab(bgr)

        # Resize for model input
        L_resized = cv2.resize(L_orig, (self.input_size[1], self.input_size[0]),
                               interpolation=cv2.INTER_LINEAR)

        # Normalize and convert to tensor
        L_norm = normalize_L(L_resized)
        L_tensor = torch.from_numpy(L_norm).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        L_tensor = L_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            if self.loss_type == "cross_entropy" and self.quantizer is not None:
                ab_pred = self.model.predict_ab(L_tensor, self.quantizer, self.temperature)
            else:
                ab_pred_small = self.model(L_tensor)
                ab_pred = F.interpolate(
                    ab_pred_small,
                    size=(self.input_size[0], self.input_size[1]),
                    mode="bilinear", align_corners=False
                )
                ab_pred = ab_pred * 110.0  # denormalize

        # Convert back to numpy
        ab_pred_np = ab_pred[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

        # Resize ab to original resolution
        ab_full = cv2.resize(ab_pred_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Reconstruct colorized image
        result_bgr = lab_to_bgr(L_orig, ab_full)

        elapsed = time.time() - start

        info_dict = {
            "elapsed_sec": elapsed,
            "model_name": type(self.model).__name__,
            "device": str(self.device),
            "temperature": self.temperature,
            "input_size": self.input_size,
            "original_size": (orig_h, orig_w),
            "loss_type": self.loss_type,
        }

        return result_bgr, info_dict

    def colorize_batch(self, image_paths):
        """Colorize multiple images.

        Args:
            image_paths: List of paths to grayscale images.

        Returns:
            List of (result_bgr, info_dict) tuples.
        """
        results = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                result, info = self.colorize(gray)
                info["path"] = path
                results.append((result, info))
        return results
