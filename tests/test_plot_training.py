"""Tests for tools/plot_training.py.

Covers:
- tqdm-line parser (per-batch loss from in-flight training log)
- training_log.json loader (per-epoch curves once epochs finish)
- the script's graceful fallback when neither source exists
- the figure-save path resolution

The script itself is a CLI in tools/, so we import its helpers directly
rather than invoking via subprocess. Plotting is skipped on the CI path —
we only verify the data-extraction layer; figure writing is exercised
indirectly by `matplotlib.use('Agg')` in plot_training.py.
"""

import importlib.util
import json
import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLOT_TRAINING_PATH = os.path.join(_PROJECT_ROOT, "tools", "plot_training.py")


def _load_plot_training_module():
    """Import tools/plot_training.py as a module (it's not on the path)."""
    sys.path.insert(0, _PROJECT_ROOT)
    spec = importlib.util.spec_from_file_location("plot_training", _PLOT_TRAINING_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pt():
    return _load_plot_training_module()


class TestParseTqdmLog:
    """The training log produced by `train_deep.py` contains tqdm lines like:

        Epoch 1:   1%|          | 130/14785 [04:55<8:46:25,  2.15s/it, loss=2.5463]

    `parse_tqdm_log_for_batch_loss(text)` should extract a list of
    (epoch, batch_idx, loss) tuples.
    """

    def test_extracts_single_line(self, pt):
        line = "Epoch 1:   1%|          | 130/14785 [04:55<8:46:25,  2.15s/it, loss=2.5463]"
        result = pt.parse_tqdm_log_for_batch_loss(line)
        assert result == [(1, 130, 2.5463)]

    def test_extracts_multiple_lines(self, pt):
        text = (
            "Epoch 1:   0%|          | 1/14785 [00:21<87:16:02, 21.25s/it, loss=5.3466]\n"
            "Epoch 1:   0%|          | 2/14785 [00:23<41:07:33, 10.02s/it, loss=5.3707]\n"
            "Epoch 2:   0%|          | 1/14785 [00:21<87:16:02, 21.25s/it, loss=4.1000]\n"
        )
        result = pt.parse_tqdm_log_for_batch_loss(text)
        assert result == [(1, 1, 5.3466), (1, 2, 5.3707), (2, 1, 4.1)]

    def test_ignores_non_tqdm_lines(self, pt):
        text = (
            "Device: cuda\n"
            "Model: Zhang16Net (31,624,745 params)\n"
            "Epoch 1:   0%|          | 1/14785 [00:21<87:16:02, 21.25s/it, loss=5.3466]\n"
            "Loaded 77 weight keys from models/pretrained/zhang16_eccv.pth\n"
        )
        result = pt.parse_tqdm_log_for_batch_loss(text)
        assert result == [(1, 1, 5.3466)]

    def test_returns_empty_for_no_matches(self, pt):
        result = pt.parse_tqdm_log_for_batch_loss("no matching content here")
        assert result == []

    def test_handles_carriage_returns_within_a_single_line(self, pt):
        # tqdm uses \r to overwrite the same line; tee captures them concatenated.
        # Parser should treat each `Epoch N: ...loss=X.XXXX` segment as a sample.
        text = (
            "Epoch 1:   0%|          | 1/14785 [00:21<87:16:02, 21.25s/it, loss=5.3466]"
            "Epoch 1:   0%|          | 2/14785 [00:23<41:07:33, 10.02s/it, loss=5.3707]"
        )
        result = pt.parse_tqdm_log_for_batch_loss(text)
        assert result == [(1, 1, 5.3466), (1, 2, 5.3707)]


class TestLoadEpochHistory:
    """`load_epoch_history(path)` reads training_log.json and returns the dict."""

    def test_loads_valid_history(self, pt, tmp_path):
        history = {
            "train_loss": [3.0, 2.5, 2.1],
            "val_loss": [2.8, 2.4, 2.0],
            "val_psnr": [12.0, 14.5, 16.0],
            "val_ssim": [0.5, 0.6, 0.65],
        }
        path = tmp_path / "training_log.json"
        path.write_text(json.dumps(history))
        result = pt.load_epoch_history(str(path))
        assert result == history

    def test_returns_none_when_missing(self, pt, tmp_path):
        result = pt.load_epoch_history(str(tmp_path / "does_not_exist.json"))
        assert result is None

    def test_returns_none_when_empty(self, pt, tmp_path):
        path = tmp_path / "training_log.json"
        path.write_text(json.dumps({"train_loss": []}))
        result = pt.load_epoch_history(str(path))
        # Empty histories aren't useful for plotting; treat as no-data
        assert result is None


class TestMainCli:
    """Smoke: invoking main() with no data sources prints a useful message + returns non-error."""

    def test_no_data_exits_cleanly(self, pt, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # No training_log.json, no --log arg, no figure output path required.
        rc = pt.main([
            "--history", str(tmp_path / "nope.json"),
            "--out", str(tmp_path / "fig.png"),
        ])
        # Should print a hint and exit 0 (graceful no-op)
        captured = capsys.readouterr()
        assert rc == 0
        assert "no" in (captured.out + captured.err).lower()

    def test_plots_per_epoch_history_to_file(self, pt, tmp_path, monkeypatch):
        # End-to-end: with a valid history, the script must write a PNG.
        history = {
            "train_loss": [3.0, 2.5, 2.1],
            "val_loss": [2.8, 2.4, 2.0],
            "val_psnr": [12.0, 14.5, 16.0],
            "val_ssim": [0.5, 0.6, 0.65],
        }
        history_path = tmp_path / "training_log.json"
        history_path.write_text(json.dumps(history))
        out_path = tmp_path / "curves.png"
        rc = pt.main(["--history", str(history_path), "--out", str(out_path)])
        assert rc == 0
        assert out_path.exists()
        assert out_path.stat().st_size > 0
