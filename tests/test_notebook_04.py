"""Structural sanity tests for notebooks/04_method_comparison.ipynb.

This branch is deep-learning only. Notebook 04 must compare the four DL
model variants (Zhang16 pretrained/finetuned, Zhang17, DeOldify) — NOT the
cross-method scribble/example/DL comparison that lives on `main`. The
diffusion paradigm is explicitly excluded from the benchmark, see
reports/deep_learning/sections/experiments.tex.

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
    def test_references_all_four_models(self, all_source):
        text = all_source.lower()
        # Each of the 4 model variants must be named somewhere.
        assert "zhang16" in text or "zhang 2016" in text
        assert "pretrained" in text and "fine-tuned" in text or "finetuned" in text
        assert "zhang17" in text or "zhang 2017" in text or "siggraph" in text
        assert "deoldify" in text

    def test_no_scribble_or_example_content(self, all_source):
        text = all_source.lower()
        assert "scribble" not in text
        assert "example-based" not in text
        assert "example_based" not in text

    def test_no_diffusion_row_in_comparison_code(self, nb):
        """Diffusion paradigm is excluded from the benchmark
        (reports/deep_learning/sections/experiments.tex). The notebook can
        explain *why* it's excluded in prose, but no comparison **code cell**
        should reference ControlNet."""
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            assert "controlnet" not in src.lower(), (
                "code cell mentions ControlNet — diffusion paradigm is excluded"
            )

    def test_loads_from_comparison_results_dir(self, all_source):
        assert "results_deep" in all_source or "comparison" in all_source

    def test_mentions_three_metrics(self, all_source):
        text = all_source.lower()
        assert "psnr" in text
        assert "ssim" in text
        assert "lpips" in text


# ---------------------------------------------------------------------------
# End-to-end execution (Phase 6, T6.2). Gated on nbconvert + ipykernel so it
# only runs in environments equipped to execute notebooks (e.g., conda env
# AI). In the base env this skips cleanly rather than failing.
# ---------------------------------------------------------------------------
def test_notebook_04_executes_end_to_end(tmp_path):
    """Run every cell of notebook 04 against the canonical Phase-4 artifacts."""
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbconvert")
    from nbconvert.preprocessors import ExecutePreprocessor
    with open(NB_PATH, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=300, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(NB_PATH)}})
    out = tmp_path / "executed_04.ipynb"
    with open(out, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    assert out.exists() and out.stat().st_size > 0
