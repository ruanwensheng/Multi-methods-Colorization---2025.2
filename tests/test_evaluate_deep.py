"""Tests for tools/evaluate_deep.py — output routing.

The eval tool runs once per checkpoint (T3.1 pretrained, T3.2 finetuned).
If both runs write to the same dir, the second clobbers the first and we
lose the baseline. The --tag flag is what keeps them separated; this
suite locks that behavior.
"""

import os
import sys
import importlib

import pytest


@pytest.fixture(scope="module")
def evaluate_deep():
    """Import the script as a module so we can call its helpers directly.

    Adding tools/ to sys.path keeps the import side-effect-free (no argv
    parsing happens at import time — only inside main()).
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tools_dir = os.path.join(root, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    return importlib.import_module("evaluate_deep")


def _mkcfg(tmp_path):
    return {"paths": {"results_deep": str(tmp_path / "results_deep")}}


class TestResolveOutputDir:
    """`--tag` must route output to a per-tag subdirectory."""

    def test_no_tag_uses_legacy_metrics_dir(self, tmp_path, evaluate_deep):
        cfg = _mkcfg(tmp_path)
        out = evaluate_deep.resolve_output_dir(cfg)
        # Legacy callers (no tag) still get the flat layout
        assert out.endswith(os.path.join("results_deep", "metrics"))

    def test_tag_appends_subdir(self, tmp_path, evaluate_deep):
        cfg = _mkcfg(tmp_path)
        out = evaluate_deep.resolve_output_dir(cfg, tag="pretrained")
        # T3.1 lands in metrics/pretrained
        assert out.endswith(os.path.join("metrics", "pretrained"))

    def test_explicit_override_wins(self, tmp_path, evaluate_deep):
        cfg = _mkcfg(tmp_path)
        forced = str(tmp_path / "elsewhere")
        out = evaluate_deep.resolve_output_dir(cfg, tag="pretrained", output_dir_override=forced)
        # If the caller passes --output-dir, the tag must be ignored — the
        # override is the most-specific signal.
        assert out == forced

    def test_two_tags_dont_collide(self, tmp_path, evaluate_deep):
        cfg = _mkcfg(tmp_path)
        pre = evaluate_deep.resolve_output_dir(cfg, tag="pretrained")
        fine = evaluate_deep.resolve_output_dir(cfg, tag="finetuned")
        # The whole point of --tag: T3.1 and T3.2 outputs are independent.
        assert pre != fine


class TestBuildColorizer:
    """The factory must route --method to the right comparison-model wrapper.

    T3.3/T3.4/T3.5 reuse evaluate_deep's metric pipeline (PSNR/SSIM/LPIPS +
    bootstrap CIs + same JSON shape) but swap in a non-Zhang16 model. The
    factory keeps the CLI surface uniform: pick our trained model via
    --model-path, or a comparison model via --method.
    """

    def test_method_zhang2017_returns_zhang2017_colorizer(self, evaluate_deep):
        from src.deep_learning.pretrained import Zhang2017Colorizer
        c = evaluate_deep.build_colorizer(method="zhang2017", model_path=None, cfg={}, device="cpu")
        assert isinstance(c, Zhang2017Colorizer)

    def test_method_deoldify_returns_deoldify_colorizer(self, evaluate_deep):
        from src.deep_learning.pretrained import DeOldifyColorizer
        c = evaluate_deep.build_colorizer(method="deoldify", model_path=None, cfg={}, device="cpu")
        assert isinstance(c, DeOldifyColorizer)

    def test_method_controlnet_returns_controlnet_colorizer(self, evaluate_deep):
        from src.deep_learning.pretrained import ControlNetColorizer
        c = evaluate_deep.build_colorizer(method="controlnet", model_path=None, cfg={}, device="cpu")
        assert isinstance(c, ControlNetColorizer)

    def test_unknown_method_raises(self, evaluate_deep):
        with pytest.raises(ValueError, match="(?i)unknown method"):
            evaluate_deep.build_colorizer(method="bogus", model_path=None, cfg={}, device="cpu")
