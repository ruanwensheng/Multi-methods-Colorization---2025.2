"""Tests for pretrained comparison model wrappers."""

import pytest


class TestPretrainedAPI:
    """Test that all pretrained wrappers expose the correct interface."""

    def test_zhang2017_has_required_properties(self):
        from src.deep_learning.pretrained import Zhang2017Colorizer
        model = Zhang2017Colorizer()
        assert hasattr(model, "colorize")
        assert hasattr(model, "name")
        assert hasattr(model, "category")
        assert hasattr(model, "is_available")

    def test_deoldify_has_required_properties(self):
        from src.deep_learning.pretrained import DeOldifyColorizer
        model = DeOldifyColorizer()
        assert hasattr(model, "colorize")
        assert hasattr(model, "name")
        assert hasattr(model, "category")
        assert hasattr(model, "is_available")

    def test_controlnet_has_required_properties(self):
        from src.deep_learning.pretrained import ControlNetColorizer
        model = ControlNetColorizer()
        assert hasattr(model, "colorize")
        assert hasattr(model, "name")
        assert hasattr(model, "category")
        assert hasattr(model, "is_available")


class TestModelCategories:
    """Test that models report correct categories."""

    def test_zhang2017_category(self):
        from src.deep_learning.pretrained import Zhang2017Colorizer
        assert Zhang2017Colorizer().category == "Interactive CNN"

    def test_deoldify_category(self):
        from src.deep_learning.pretrained import DeOldifyColorizer
        assert DeOldifyColorizer().category == "GAN"

    def test_controlnet_category(self):
        from src.deep_learning.pretrained import ControlNetColorizer
        assert ControlNetColorizer().category == "Diffusion"


class TestModelNames:
    """Test that models report readable names."""

    def test_zhang2017_name(self):
        from src.deep_learning.pretrained import Zhang2017Colorizer
        name = Zhang2017Colorizer().name
        assert "2017" in name
        assert "CNN" in name

    def test_deoldify_name(self):
        from src.deep_learning.pretrained import DeOldifyColorizer
        name = DeOldifyColorizer().name
        assert "DeOldify" in name
        assert "GAN" in name

    def test_controlnet_name(self):
        from src.deep_learning.pretrained import ControlNetColorizer
        name = ControlNetColorizer().name
        assert "ControlNet" in name
        assert "Diffusion" in name


class TestVendoredColorizers:
    """The Zhang17 wrapper must find the vendored richzhang/colorization package.

    The upstream `colorizers` package isn't on PyPI (no setup.py), so we vendor
    its Python source at `src/vendor/colorizers/`. Without a sys.path bridge,
    `from colorizers import siggraph17` raises ImportError and T3.3 silently
    falls through to the manual-load path that just `raise ImportError(...)`s.
    """

    def test_vendored_colorizers_importable(self):
        # Tests the bridge in Zhang2017Colorizer: even with no system-installed
        # `colorizers`, the wrapper should arrive at `_use_package=True` via the
        # vendored fallback.
        from src.deep_learning.pretrained import Zhang2017Colorizer
        m = Zhang2017Colorizer()
        assert m.is_available, "Zhang2017 should be available via vendored colorizers"
        # _use_package is the signal that the import worked; the manual path is
        # documented-as-broken so we don't want to be on it.
        assert getattr(m, "_use_package", False), \
            "Zhang2017Colorizer must use the vendored colorizers package, not the unfinished manual fallback"


class TestGetComparisonModels:
    """Test the factory function."""

    def test_returns_dict(self, cfg):
        from src.deep_learning.pretrained import get_comparison_models
        models = get_comparison_models(cfg)
        assert isinstance(models, dict)

    def test_disabled_models_excluded(self, cfg):
        from src.deep_learning.pretrained import get_comparison_models
        cfg["deep_learning"]["comparison"]["zhang2017"]["enabled"] = False
        models = get_comparison_models(cfg)
        assert "Zhang 2017 (Interactive CNN)" not in models

    def test_all_values_are_colorizers(self, cfg):
        from src.deep_learning.pretrained import get_comparison_models, PretrainedColorizer
        models = get_comparison_models(cfg)
        for name, model in models.items():
            assert isinstance(model, PretrainedColorizer)
