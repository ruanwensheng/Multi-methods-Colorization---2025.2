"""Bootstrap: pin caches to the project drive BEFORE any heavy import.

Why this exists:
    On Windows, the default HF/Torch caches land in ``%USERPROFILE%\\.cache\\``
    on the C: drive. With a 6 GB GPU + a 200+ GB C: drive that fills with
    games/apps, the HF + DeOldify caches (5-10 GB) exhausted
    free space, the pagefile couldn't grow, and training crashed with
    WinError 1455 (paging file too small).

    This module ensures HF_HOME / TORCH_HOME / TRANSFORMERS_CACHE point at
    ``<project>/.cache/`` (gitignored, on D:) regardless of how the entry
    point script was launched — `python tools/foo.py`, `conda run python …`,
    or via a notebook kernel.

How to use:
    Import this module at the TOP of every tool/notebook entrypoint, BEFORE
    importing `torch`, `diffusers`, `transformers`, `huggingface_hub`, etc.
    `setdefault` is used so users can still override via shell env vars.

    Example:
        from src.deep_learning import bootstrap  # noqa: F401  (side effect)
        import torch
        ...
"""

import os
import sys


def _project_root() -> str:
    # src/deep_learning/bootstrap.py  ->  <project_root>
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_caches(project_root: str | None = None) -> dict:
    """Point cache env vars at <project>/.cache/.

    Uses ``os.environ.setdefault`` so an existing user override (e.g., from a
    .env loaded by their shell) wins. Returns the dict of paths it actually
    set, for visibility / logging.
    """
    root = project_root or _project_root()
    cache_root = os.path.join(root, ".cache")
    mapping = {
        "HF_HOME":            os.path.join(cache_root, "huggingface"),
        "TRANSFORMERS_CACHE": os.path.join(cache_root, "huggingface", "hub"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(cache_root, "huggingface", "hub"),
        "TORCH_HOME":         os.path.join(cache_root, "torch"),
    }
    set_now = {}
    for k, v in mapping.items():
        if os.environ.setdefault(k, v) == v:
            set_now[k] = v
    # Create the dirs lazily — HF will do it itself, but creating them up
    # front catches typos in the path immediately.
    for v in mapping.values():
        os.makedirs(v, exist_ok=True)
    return set_now


# Note: we intentionally do NOT run setup_caches() at import time. Doing so
# made tests unstable (Python caches imported modules, so monkeypatching env
# vars after the import couldn't undo the side effect). Tools should call
# setup_caches() explicitly before importing torch/diffusers/HF.
