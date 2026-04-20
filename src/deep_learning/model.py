"""
Zhang et al. 2016 - "Colorful Image Colorization" network architecture.

Reimplementation in PyTorch of the CNN described in arXiv:1603.08511.
The network takes an L channel image and predicts a probability distribution
over 313 quantized ab color bins.

Architecture: 8 convolutional blocks using dilated convolutions,
batch normalization, and ReLU activations. No pooling layers -
spatial downsampling is done via stride-2 convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Zhang16Net(nn.Module):
    """Zhang et al. 2016 colorization network.

    Input:  L channel tensor of shape (B, 1, H, W), normalized to [-1, 1]
    Output: Logits over 313 ab bins of shape (B, 313, H/8, W/8)

    To get full-resolution ab predictions, use decode_output().

    Args:
        num_classes: Number of quantized ab bins (default: 313).
    """

    def __init__(self, num_classes=313):
        super().__init__()
        self.num_classes = num_classes

        # --- Block 1: 64 filters ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )

        # --- Block 2: 128 filters ---
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        # --- Block 3: 256 filters ---
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        # --- Block 4: 512 filters, dilation=1 ---
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        # --- Block 5: 512 filters, dilation=2 ---
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        # --- Block 6: 512 filters, dilation=2 ---
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        # --- Block 7: 512 filters, dilation=1 ---
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        # --- Block 8: Decode to num_classes ---
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 1, H, W) normalized L channel tensor.

        Returns:
            (B, num_classes, H/4, W/4) logits over ab bins.
        """
        x = self.conv1(x)   # -> H/2
        x = self.conv2(x)   # -> H/4
        x = self.conv3(x)   # -> H/8
        x = self.conv4(x)   # -> H/8 (dilated)
        x = self.conv5(x)   # -> H/8 (dilated)
        x = self.conv6(x)   # -> H/8 (dilated)
        x = self.conv7(x)   # -> H/8 (dilated)
        x = self.conv8(x)   # -> H/4 (upsample)
        return x

    def predict_ab(self, L_tensor, quantizer, temperature=0.38):
        """Full inference: L tensor -> ab values at full resolution.

        Args:
            L_tensor: (B, 1, H, W) normalized L channel.
            quantizer: ABQuantizer instance with ab_bins.
            temperature: Annealing temperature for decoding.

        Returns:
            (B, 2, H, W) predicted ab values in original scale [-110, 110].
        """
        B, _, H, W = L_tensor.shape

        # Forward pass
        logits = self.forward(L_tensor)  # (B, 313, H/4, W/4)
        probs = F.softmax(logits, dim=1)

        # Upsample probabilities to full resolution
        probs_full = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)

        # Decode to ab using annealed-mean
        ab_bins_tensor = torch.from_numpy(quantizer.ab_bins).float().to(L_tensor.device)

        # Apply temperature annealing
        log_probs = torch.log(probs_full + 1e-8)
        annealed = torch.exp(log_probs / temperature)
        annealed = annealed / (annealed.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum: (B, Q, H, W) @ (Q, 2) -> (B, 2, H, W)
        # Reshape for matrix multiply
        B, Q, H_out, W_out = annealed.shape
        annealed_flat = annealed.permute(0, 2, 3, 1).reshape(-1, Q)  # (B*H*W, Q)
        ab_pred = annealed_flat @ ab_bins_tensor  # (B*H*W, 2)
        ab_pred = ab_pred.reshape(B, H_out, W_out, 2).permute(0, 3, 1, 2)  # (B, 2, H, W)

        return ab_pred


class Zhang16Regression(nn.Module):
    """Simplified regression variant that directly predicts ab channels.

    Uses the same encoder architecture as Zhang16Net but outputs 2 channels
    (a, b) instead of 313 class probabilities. Trained with Huber/L1 loss.

    This is simpler to train but tends to produce more desaturated results.
    """

    def __init__(self):
        super().__init__()

        # Same encoder as Zhang16Net
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        # Decoder outputs 2 channels (a, b) normalized to [-1, 1]
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 1, H, W) normalized L channel.

        Returns:
            (B, 2, H/4, W/4) predicted ab values normalized to [-1, 1].
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x

    def predict_ab(self, L_tensor, quantizer=None, temperature=None):
        """Full inference: L tensor -> ab values at full resolution.

        Args:
            L_tensor: (B, 1, H, W) normalized L channel.
            quantizer: Unused (kept for API compatibility).
            temperature: Unused (kept for API compatibility).

        Returns:
            (B, 2, H, W) predicted ab values in original scale [-110, 110].
        """
        B, _, H, W = L_tensor.shape
        ab_norm = self.forward(L_tensor)  # (B, 2, H/4, W/4)
        ab_full = F.interpolate(ab_norm, size=(H, W), mode="bilinear", align_corners=False)
        return ab_full * 110.0  # Denormalize to [-110, 110]


def build_model(cfg, quantizer=None):
    """Build model from config.

    Args:
        cfg: dict with cfg["deep_learning"]["loss"] to determine model type.
        quantizer: ABQuantizer instance. If provided, num_classes is set
            to match quantizer.num_bins for consistency.

    Returns:
        nn.Module: Zhang16Net (classification) or Zhang16Regression.
    """
    dl_cfg = cfg["deep_learning"]
    loss_type = dl_cfg.get("loss", "cross_entropy")

    if loss_type == "huber":
        return Zhang16Regression()
    else:
        if quantizer is not None:
            num_classes = quantizer.num_bins
        else:
            num_classes = dl_cfg.get("num_classes", 313)
        return Zhang16Net(num_classes=num_classes)
