"""
Loss functions for image colorization training.

Implements:
1. ClassRebalancedCELoss - Multinomial cross-entropy with class rebalancing
   (Zhang et al. 2016, Equations 2-4)
2. HuberColorLoss - Smooth-L1 loss on ab channels (simpler baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassRebalancedCELoss(nn.Module):
    """Class-rebalanced cross-entropy loss for colorization.

    Treats colorization as a classification problem over 313 quantized ab bins.
    Applies per-pixel class rebalancing to handle the dataset bias toward
    desaturated (grayish) colors.

    Reference: Zhang et al. 2016, ECCV, Eq. 2-4.

    Args:
        quantizer: ABQuantizer instance providing encode() and ab_bins.
        class_weights: Optional precomputed class weights (num_bins,).
            If None, uniform weights are used.
    """

    def __init__(self, quantizer, class_weights=None):
        super().__init__()
        self.quantizer = quantizer
        self.num_classes = quantizer.num_bins

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.from_numpy(class_weights).float()
            )
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(self.num_classes, dtype=torch.float32)
            )

    def forward(self, pred_logits, target_ab):
        """Compute rebalanced cross-entropy loss.

        Args:
            pred_logits: (B, num_classes, H', W') raw logits from the model.
            target_ab: (B, 2, H, W) target ab channels normalized to [-1, 1].

        Returns:
            Scalar loss tensor.
        """
        B, _, H_pred, W_pred = pred_logits.shape

        # Denormalize target ab
        target_ab_denorm = target_ab * 110.0  # [-110, 110]

        # Resize target to match prediction spatial size
        target_resized = F.interpolate(
            target_ab_denorm, size=(H_pred, W_pred), mode="bilinear", align_corners=False
        )

        # Quantize target ab values to bin indices: (B, H', W')
        target_np = target_resized.permute(0, 2, 3, 1).detach().cpu().numpy()
        target_indices = self.quantizer.encode(target_np)
        target_indices = torch.from_numpy(target_indices).long().to(pred_logits.device)

        # Compute per-pixel rebalancing weight
        # v(Z_h,w) = w_q where q = quantized bin of the target pixel
        pixel_weights = self.class_weights[target_indices]  # (B, H', W')

        # Cross-entropy loss with per-pixel weights
        loss = F.cross_entropy(pred_logits, target_indices, reduction="none")  # (B, H', W')
        weighted_loss = loss * pixel_weights

        return weighted_loss.mean()


class HuberColorLoss(nn.Module):
    """Smooth-L1 (Huber) loss on ab channels.

    A simpler alternative to the classification approach.
    Directly regresses ab values using Huber loss, which is more robust
    than MSE to outliers but tends to produce more desaturated results.

    Reference: Zhang et al. 2017, SIGGRAPH, Eq. 4.
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred_ab, target_ab):
        """Compute Huber loss between predicted and target ab channels.

        Args:
            pred_ab: (B, 2, H', W') predicted ab, normalized to [-1, 1].
            target_ab: (B, 2, H, W) target ab, normalized to [-1, 1].

        Returns:
            Scalar loss tensor.
        """
        # Resize target to match prediction size if needed
        if pred_ab.shape[2:] != target_ab.shape[2:]:
            target_ab = F.interpolate(
                target_ab, size=pred_ab.shape[2:], mode="bilinear", align_corners=False
            )

        return F.smooth_l1_loss(pred_ab, target_ab, beta=self.delta)


def build_loss(cfg, quantizer=None):
    """Build loss function from config.

    Args:
        cfg: dict with cfg["deep_learning"]["loss"] specifying loss type.
        quantizer: ABQuantizer instance (required for cross_entropy loss).

    Returns:
        nn.Module loss function.
    """
    loss_type = cfg["deep_learning"].get("loss", "cross_entropy")

    if loss_type == "cross_entropy":
        if quantizer is None:
            raise ValueError("ABQuantizer required for cross_entropy loss")
        class_weights = quantizer.compute_class_weights()
        return ClassRebalancedCELoss(quantizer, class_weights)
    elif loss_type == "huber":
        return HuberColorLoss(delta=1.0)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
