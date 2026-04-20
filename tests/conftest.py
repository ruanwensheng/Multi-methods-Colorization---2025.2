"""Shared fixtures for deep learning colorization tests."""

import os
import sys
import pytest
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def cfg():
    """Load project config."""
    from src.deep_learning.utils import load_config
    return load_config("configs/config.yaml")


@pytest.fixture
def quantizer():
    """Create an ABQuantizer instance."""
    from src.deep_learning.quantize import ABQuantizer
    return ABQuantizer()


@pytest.fixture
def sample_rgb():
    """A 64x64 synthetic RGB image with diverse colors."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32, :32] = [255, 0, 0]      # Red
    img[:32, 32:] = [0, 255, 0]      # Green
    img[32:, :32] = [0, 0, 255]      # Blue
    img[32:, 32:] = [255, 255, 0]    # Yellow
    return img


@pytest.fixture
def sample_gray():
    """A 64x64 grayscale image."""
    return np.random.randint(0, 255, (64, 64), dtype=np.uint8)


@pytest.fixture
def sample_L_tensor():
    """A normalized L channel tensor (1, 1, 256, 256)."""
    return torch.randn(1, 1, 256, 256)


@pytest.fixture
def batch_L_tensor():
    """A batch of L channel tensors (4, 1, 128, 128)."""
    return torch.randn(4, 1, 128, 128)


@pytest.fixture
def sample_ab_tensor():
    """A normalized ab channel tensor (1, 2, 256, 256) in [-1, 1]."""
    return torch.randn(1, 2, 256, 256).clamp(-1, 1)
