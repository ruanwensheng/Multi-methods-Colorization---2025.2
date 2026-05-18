"""Deep learning colorization package.

Submodules are imported lazily — `from src.deep_learning.X import Y` works
without triggering imports of unrelated modules. This keeps pure-numpy utilities
(e.g., src.deep_learning.stats) importable in environments where torch / skimage
aren't installed yet.
"""
