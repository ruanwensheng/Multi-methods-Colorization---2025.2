"""Structural sanity tests for notebooks/04_method_comparison.ipynb.

This branch is deep-learning only. Notebook 04 must compare the five DL
model variants (Zhang16 pretrained/finetuned, Zhang17, DeOldify, ControlNet)
— NOT the cross-method scribble/example/DL comparison that lives on `main`.

These tests don't execute the notebook; they only check its declarative
content so the scaffold can't silently regress to cross-method.
"""

import json
import os

import pytest

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "04_method_comparison.ipynb",
)


@pytest.fixture(scope="module")
def nb():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_source(nb):
    """Concatenated source of every cell (markdown + code)."""
    chunks = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        chunks.append(src)
    return "\n".join(chunks)


class TestNotebookStructure:
    def test_is_valid_nbformat_4(self, nb):
        assert nb.get("nbformat") == 4

    def test_has_kernelspec(self, nb):
        assert nb.get("metadata", {}).get("kernelspec") is not None

    def test_has_at_least_one_code_cell(self, nb):
        kinds = {c.get("cell_type") for c in nb.get("cells", [])}
        assert "code" in kinds


class TestNotebookContent:
    def test_references_all_five_models(self, all_source):
        text = all_source.lower()
        # Each of the 5 model variants must be named somewhere.
        assert "zhang16" in text or "zhang 2016" in text
        assert "pretrained" in text and "fine-tuned" in text or "finetuned" in text
        assert "zhang17" in text or "zhang 2017" in text or "siggraph" in text
        assert "deoldify" in text
        assert "controlnet" in text

    def test_no_scribble_or_example_content(self, all_source):
        text = all_source.lower()
        assert "scribble" not in text
        assert "example-based" not in text
        assert "example_based" not in text

    def test_loads_from_comparison_results_dir(self, all_source):
        assert "results_deep" in all_source or "comparison" in all_source

    def test_mentions_three_metrics(self, all_source):
        text = all_source.lower()
        assert "psnr" in text
        assert "ssim" in text
        assert "lpips" in text
