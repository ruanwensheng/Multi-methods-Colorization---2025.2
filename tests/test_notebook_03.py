"""Structural + end-to-end sanity tests for notebooks/03_deep_learning_pipeline.ipynb.

Notebook 03 is the pipeline walkthrough (data -> quantization -> model ->
training -> evaluation -> comparison). For Phase 6 (T6.1) it must be
re-runnable without re-doing 50 epochs of training -- the training cell carries
a skip-if-already-trained guard, locked in by ``test_training_cell_has_skip_guard``.

The end-to-end execution test is gated on ``nbconvert`` so it runs in the AI
env (which has jupyter/ipykernel) and skips cleanly in base.
"""

import json
import os

import pytest

NB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks",
    "03_deep_learning_pipeline.ipynb",
)


@pytest.fixture(scope="module")
def nb():
    with open(NB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_source(nb):
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

    def test_has_code_and_markdown_cells(self, nb):
        kinds = {c.get("cell_type") for c in nb.get("cells", [])}
        assert "code" in kinds and "markdown" in kinds


class TestNotebookContent:
    def test_loads_config_from_yaml(self, all_source):
        # The notebook must drive off the single config (no hard-coded hyperparams).
        assert "load_config" in all_source
        assert "configs/config.yaml" in all_source

    def test_covers_pipeline_stages(self, all_source):
        text = all_source.lower()
        for stage in ("quantization", "training", "evaluation"):
            assert stage in text, f"notebook 03 missing pipeline stage: {stage}"

    def test_training_cell_has_skip_guard(self, all_source):
        """T6.1 fix: the training cell must short-circuit when best_model.pth
        already exists, so the demo notebook is re-runnable without 50 epochs."""
        # Look for the guard pattern --- robust to whitespace.
        assert "best_model.pth" in all_source and "Skipping training" in all_source, (
            "training cell is missing the skip-if-already-trained guard "
            "(the notebook would re-train on every execution)"
        )

    def test_no_scribble_or_example_content(self, all_source):
        text = all_source.lower()
        assert "scribble" not in text
        assert "example-based" not in text


# ---------------------------------------------------------------------------
# End-to-end execution (T6.1). Gated on nbconvert so it only runs in
# environments equipped to execute notebooks.
# ---------------------------------------------------------------------------
def test_notebook_03_executes_end_to_end(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbconvert")
    from nbconvert.preprocessors import ExecutePreprocessor
    with open(NB_PATH, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(NB_PATH)}})
    out = tmp_path / "executed_03.ipynb"
    with open(out, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    assert out.exists() and out.stat().st_size > 0
