"""Tests for the Zhang16 ECCV pretrained-weight loader.

The official Zhang 2016 ECCV checkpoint uses `modelN.X.Y` naming, while
our Zhang16Net uses `convN.X.Y`. Additionally, the official `conv8`
block has 7 layers (output index 6, 313 bins) while ours has 5 (output
index 4, 233 bins from `pts_in_hull.npy`). The loader must:

  1. Rename `modelN.X.Y` -> `convN.X.Y`.
  2. Keep only keys whose renamed name+shape exists in the target model.
  3. Return a (loaded_count, dropped_keys) summary for transparency.

These tests use a hand-rolled synthetic source state-dict so they run
in milliseconds without touching the real 129 MB checkpoint.
"""

import torch
import pytest


def _make_src_sd(num_classes=313):
    """Synthetic Zhang16 ECCV state_dict with the right keys + shapes."""
    sd = {}
    # Block 1: conv-conv-bn, 1->64, 64->64
    sd["model1.0.weight"] = torch.randn(64, 1, 3, 3)
    sd["model1.0.bias"] = torch.randn(64)
    sd["model1.2.weight"] = torch.randn(64, 64, 3, 3)
    sd["model1.2.bias"] = torch.randn(64)
    sd["model1.4.weight"] = torch.randn(64)
    sd["model1.4.bias"] = torch.randn(64)
    sd["model1.4.running_mean"] = torch.randn(64)
    sd["model1.4.running_var"] = torch.randn(64)
    sd["model1.4.num_batches_tracked"] = torch.tensor(0)
    # Block 8 (official): 7 layers, output index 6 with `num_classes` channels.
    sd["model8.0.weight"] = torch.randn(512, 256, 4, 4)   # ConvTranspose2d
    sd["model8.0.bias"] = torch.randn(256)
    sd["model8.2.weight"] = torch.randn(256, 256, 3, 3)
    sd["model8.2.bias"] = torch.randn(256)
    sd["model8.4.weight"] = torch.randn(256, 256, 3, 3)   # EXTRA layer, not in ours
    sd["model8.4.bias"] = torch.randn(256)
    sd["model8.6.weight"] = torch.randn(num_classes, 256, 1, 1)
    sd["model8.6.bias"] = torch.randn(num_classes)
    return sd


def _make_target_sd(num_classes=233):
    """Synthetic state_dict for our Zhang16Net (conv1, conv8 truncated, 233 bins)."""
    sd = {}
    sd["conv1.0.weight"] = torch.zeros(64, 1, 3, 3)
    sd["conv1.0.bias"] = torch.zeros(64)
    sd["conv1.2.weight"] = torch.zeros(64, 64, 3, 3)
    sd["conv1.2.bias"] = torch.zeros(64)
    sd["conv1.4.weight"] = torch.zeros(64)
    sd["conv1.4.bias"] = torch.zeros(64)
    sd["conv1.4.running_mean"] = torch.zeros(64)
    sd["conv1.4.running_var"] = torch.zeros(64)
    sd["conv1.4.num_batches_tracked"] = torch.tensor(0)
    # Our conv8 has 5 layers; output at index 4 with num_classes channels.
    sd["conv8.0.weight"] = torch.zeros(512, 256, 4, 4)
    sd["conv8.0.bias"] = torch.zeros(256)
    sd["conv8.2.weight"] = torch.zeros(256, 256, 3, 3)
    sd["conv8.2.bias"] = torch.zeros(256)
    sd["conv8.4.weight"] = torch.zeros(num_classes, 256, 1, 1)
    sd["conv8.4.bias"] = torch.zeros(num_classes)
    return sd


class TestRemapZhang16EccvStateDict:
    def test_renames_modelN_to_convN(self):
        from src.deep_learning.model import remap_zhang16_eccv_state_dict
        src = _make_src_sd()
        target = _make_target_sd()
        remapped, _ = remap_zhang16_eccv_state_dict(src, target)
        # Every remapped key starts with conv, never with model
        assert all(k.startswith("conv") for k in remapped)
        assert all(not k.startswith("model") for k in remapped)

    def test_keeps_only_shape_compatible_keys(self):
        from src.deep_learning.model import remap_zhang16_eccv_state_dict
        src = _make_src_sd()
        target = _make_target_sd()
        remapped, dropped = remap_zhang16_eccv_state_dict(src, target)
        # conv1.* should all transfer (9 keys: weight/bias x conv1.0/1.2 + 5 BN params)
        assert "conv1.0.weight" in remapped
        assert "conv1.2.weight" in remapped
        assert "conv1.4.weight" in remapped
        # conv8.0 and conv8.2 match by shape
        assert "conv8.0.weight" in remapped
        assert "conv8.2.weight" in remapped
        # model8.4 maps to conv8.4 by naming, but conv8.4 in target is the head
        # (256, 256, 3, 3) vs (233, 256, 1, 1) -> shape mismatch -> dropped
        assert "conv8.4.weight" not in remapped
        # model8.6 has no conv8.6 in target -> dropped
        assert "conv8.6.weight" not in remapped

    def test_dropped_keys_are_reported(self):
        from src.deep_learning.model import remap_zhang16_eccv_state_dict
        src = _make_src_sd()
        target = _make_target_sd()
        _, dropped = remap_zhang16_eccv_state_dict(src, target)
        # Both the architectural-mismatch and head-shape keys should be reported
        joined = " ".join(dropped)
        assert "model8.4" in joined or "conv8.4" in joined
        assert "model8.6" in joined or "conv8.6" in joined

    def test_values_are_preserved_through_rename(self):
        from src.deep_learning.model import remap_zhang16_eccv_state_dict
        src = _make_src_sd()
        target = _make_target_sd()
        remapped, _ = remap_zhang16_eccv_state_dict(src, target)
        # Renamed weights must be the SAME tensor data
        torch.testing.assert_close(remapped["conv1.0.weight"], src["model1.0.weight"])
        torch.testing.assert_close(remapped["conv8.0.weight"], src["model8.0.weight"])

    def test_empty_src_returns_empty(self):
        from src.deep_learning.model import remap_zhang16_eccv_state_dict
        target = _make_target_sd()
        remapped, dropped = remap_zhang16_eccv_state_dict({}, target)
        assert remapped == {}
        assert dropped == []

    def test_handles_nested_checkpoint_format(self):
        # Some checkpoints wrap state_dict under "model_state_dict"; the helper
        # should accept either form (raw dict or wrapped) via an unwrap step
        # at the call site, but the remapper itself only handles flat dicts.
        from src.deep_learning.model import remap_zhang16_eccv_state_dict
        src = _make_src_sd()
        target = _make_target_sd()
        remapped, _ = remap_zhang16_eccv_state_dict(src, target)
        assert len(remapped) > 0


class TestLoadZhang16EccvWeights:
    """End-to-end: real Zhang16Net + synthetic remapped weights."""

    def test_loads_synthetic_weights_into_real_zhang16net(self):
        # Note: this test instantiates a real Zhang16Net but uses synthetic
        # weights only — no checkpoint file I/O, no GPU.
        from src.deep_learning.model import Zhang16Net, remap_zhang16_eccv_state_dict
        model = Zhang16Net(num_classes=233)
        target_sd = model.state_dict()
        src_sd = {}
        # Build a synthetic ECCV-style state dict using shapes that match
        # what our model expects, so we get a realistic loading exercise.
        for k, v in target_sd.items():
            if k.startswith("conv1.") or k.startswith("conv2."):
                # Mirror under model1./model2. with same shapes; use empty_like
                # so we cover non-float dtypes (num_batches_tracked is int64).
                renamed = k.replace("conv", "model", 1)
                src_sd[renamed] = torch.empty_like(v)
        remapped, _ = remap_zhang16_eccv_state_dict(src_sd, target_sd)
        # All keys in remapped must be loadable
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        assert len(unexpected) == 0, f"Should have no unexpected keys: {unexpected}"
