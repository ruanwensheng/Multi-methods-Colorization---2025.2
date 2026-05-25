"""Tests for the cache-redirection bootstrap.

The bootstrap module must pin HF / Torch caches to the project drive so a
full C: drive doesn't crash training with WinError 1455 (paging file too
small) when HF tries to extract weights.
"""

import os
import importlib

import pytest


@pytest.fixture(autouse=True)
def _clean_cache_env(monkeypatch):
    """Each test starts with no cache env vars set, so we observe what
    bootstrap *actually* does, not what was inherited from the parent shell.
    """
    for var in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME"):
        monkeypatch.delenv(var, raising=False)
    yield


class TestSetupCaches:
    def test_pins_HF_HOME_under_project(self, tmp_path):
        from src.deep_learning.bootstrap import setup_caches
        setup_caches(project_root=str(tmp_path))
        # All cache env vars must live under the project (no fallback to ~/.cache)
        assert os.environ["HF_HOME"].startswith(str(tmp_path))
        assert os.environ["TORCH_HOME"].startswith(str(tmp_path))
        assert os.environ["TRANSFORMERS_CACHE"].startswith(str(tmp_path))

    def test_user_override_wins(self, tmp_path, monkeypatch):
        # If the user has a real env var, bootstrap must NOT clobber it.
        # Forcing caches off-project on purpose is a legitimate developer choice.
        forced = str(tmp_path / "user_chose_here")
        monkeypatch.setenv("HF_HOME", forced)
        from src.deep_learning.bootstrap import setup_caches
        setup_caches(project_root=str(tmp_path))
        assert os.environ["HF_HOME"] == forced

    def test_creates_cache_dirs(self, tmp_path):
        from src.deep_learning.bootstrap import setup_caches
        setup_caches(project_root=str(tmp_path))
        # Cache dirs must exist on disk after setup so HF doesn't fail with
        # "no such file or directory" on first download.
        assert os.path.isdir(os.environ["HF_HOME"])
        assert os.path.isdir(os.environ["TORCH_HOME"])

    def test_returns_dict_of_what_was_set(self, tmp_path):
        from src.deep_learning.bootstrap import setup_caches
        result = setup_caches(project_root=str(tmp_path))
        # All four cache vars were unset, so all should report as set.
        assert {"HF_HOME", "TORCH_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"} <= set(result)


class TestNoImportSideEffect:
    """Importing bootstrap MUST NOT itself set env vars.

    Reason: bare import is unsuppressible — if any test or library imports
    bootstrap to use ``setup_caches`` as a function, we don't want a hidden
    side effect to clobber the caller's env. Tools call setup_caches() explicitly.
    """

    def test_bare_import_does_not_touch_env(self):
        for var in ("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "HUGGINGFACE_HUB_CACHE"):
            assert var not in os.environ, \
                f"bootstrap import set {var} as a side effect; that breaks user overrides"
        # Now import — must still leave env clean
        import src.deep_learning.bootstrap  # noqa: F401
        for var in ("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "HUGGINGFACE_HUB_CACHE"):
            assert var not in os.environ, f"importing bootstrap set {var}; refactor"
