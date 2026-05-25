"""Tests for DeepColorizer - end-to-end inference wrapper."""

import numpy as np
import cv2
import pytest


class TestDeepColorizerInit:
    """Test colorizer initialization."""

    def test_creates_without_weights(self):
        from src.deep_learning.colorizer import DeepColorizer
        colorizer = DeepColorizer(device="cpu")
        assert colorizer.model is not None

    def test_default_temperature(self):
        from src.deep_learning.colorizer import DeepColorizer
        colorizer = DeepColorizer(device="cpu")
        assert colorizer.temperature == 0.38

    def test_model_in_eval_mode(self):
        from src.deep_learning.colorizer import DeepColorizer
        colorizer = DeepColorizer(device="cpu")
        assert not colorizer.model.training


class TestDeepColorizerColorize:
    """Test the colorize() method with various inputs."""

    @pytest.fixture
    def colorizer(self):
        from src.deep_learning.colorizer import DeepColorizer
        return DeepColorizer(device="cpu", temperature=0.38)

    def test_grayscale_input(self, colorizer, sample_gray):
        result, info = colorizer.colorize(sample_gray)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_bgr_3channel_input(self, colorizer):
        bgr = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result, info = colorizer.colorize(bgr)
        assert result.shape == (100, 80, 3)

    def test_preserves_original_resolution(self, colorizer):
        for h, w in [(50, 50), (100, 200), (300, 150)]:
            gray = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            result, _ = colorizer.colorize(gray)
            assert result.shape == (h, w, 3)

    def test_info_dict_has_required_keys(self, colorizer, sample_gray):
        _, info = colorizer.colorize(sample_gray)
        assert "elapsed_sec" in info
        assert "model_name" in info
        assert "device" in info
        assert "temperature" in info

    def test_elapsed_time_positive(self, colorizer, sample_gray):
        _, info = colorizer.colorize(sample_gray)
        assert info["elapsed_sec"] > 0

    def test_output_is_valid_bgr(self, colorizer, sample_gray):
        result, _ = colorizer.colorize(sample_gray)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_deterministic_with_same_input(self, colorizer):
        gray = np.full((32, 32), 128, dtype=np.uint8)
        r1, _ = colorizer.colorize(gray)
        r2, _ = colorizer.colorize(gray)
        np.testing.assert_array_equal(r1, r2)


class TestDeepColorizerEccvFormat:
    """DeepColorizer must accept raw Zhang 2016 ECCV checkpoints.

    The official checkpoint uses `modelN.*` keys; our Zhang16Net uses `convN.*`.
    Without remapping, `strict=False` load_state_dict drops every weight and
    we'd be evaluating a random model on the benchmark — which is exactly the
    T3.1 deliverable (pretrained baseline), so silently broken results would
    look like "Zhang16 trained from scratch produces PSNR 20 dB" instead of
    the real ECCV number (~25 dB). Detect the ECCV format and route through
    the existing load_zhang16_eccv_weights remapper.
    """

    def _make_eccv_checkpoint(self, tmp_path, num_classes=313):
        """Build a synthetic ECCV-format checkpoint with the official `modelN.*`
        naming and the right shapes to exercise the remapper end-to-end.
        """
        import torch
        from src.deep_learning.model import Zhang16Net

        # Use our target model's state_dict to discover the expected shapes,
        # then rebuild them under the ECCV `modelN.*` naming.
        target = Zhang16Net(num_classes=num_classes if num_classes else 233).state_dict()
        src = {}
        for tk, tv in target.items():
            if tk.startswith("conv"):
                ek = "model" + tk[len("conv"):]
                src[ek] = torch.empty_like(tv).normal_(0, 0.01) if tv.dtype.is_floating_point \
                    else torch.zeros_like(tv)
        ckpt_path = str(tmp_path / "zhang16_eccv_synth.pth")
        torch.save(src, ckpt_path)
        return ckpt_path

    def test_loads_raw_eccv_checkpoint_without_crashing(self, tmp_path):
        from src.deep_learning.colorizer import DeepColorizer
        ckpt = self._make_eccv_checkpoint(tmp_path)
        # Must not raise — without ECCV remapping, strict load fails on model* keys.
        colorizer = DeepColorizer(model_path=ckpt, device="cpu")
        assert colorizer.model is not None

    def test_eccv_weights_actually_transfer(self, tmp_path):
        # The point of remapping is to actually load the upstream weights, not
        # silently start from random init. Sample one conv weight from the
        # source and verify it survived the rename.
        import torch
        from src.deep_learning.colorizer import DeepColorizer

        ckpt_path = self._make_eccv_checkpoint(tmp_path)
        src_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Pick a key that we know exists in both formats (conv1 is small + present)
        ecvv_key = "model1.0.weight"
        target_key = "conv1.0.weight"
        assert ecvv_key in src_sd

        colorizer = DeepColorizer(model_path=ckpt_path, device="cpu")
        loaded = dict(colorizer.model.state_dict())
        torch.testing.assert_close(loaded[target_key], src_sd[ecvv_key])

    def test_wrapped_checkpoint_still_loads(self, tmp_path):
        # Don't regress the existing path: our training output wraps the state
        # dict under "model_state_dict" with conv* keys, and must still work.
        import torch
        from src.deep_learning.colorizer import DeepColorizer
        from src.deep_learning.model import Zhang16Net

        model = Zhang16Net(num_classes=233)
        payload = {"model_state_dict": model.state_dict(), "epoch": 5}
        ckpt_path = str(tmp_path / "ours_wrapped.pth")
        torch.save(payload, ckpt_path)

        colorizer = DeepColorizer(model_path=ckpt_path, device="cpu")
        assert colorizer.model is not None


class TestDeepColorizerBatch:
    """Test batch colorization."""

    def test_colorize_batch_returns_list(self, tmp_path):
        from src.deep_learning.colorizer import DeepColorizer

        # Create temp images
        for i in range(3):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i}.png"), img)

        colorizer = DeepColorizer(device="cpu")
        paths = [str(tmp_path / f"img_{i}.png") for i in range(3)]
        results = colorizer.colorize_batch(paths)
        assert len(results) == 3
        for result_bgr, info in results:
            assert result_bgr.shape[2] == 3
            assert "path" in info
