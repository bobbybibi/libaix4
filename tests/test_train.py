"""Tests for train.py — logic-gate training CLI."""
from __future__ import annotations

import numpy as np

from train import DATASETS, INPUTS, train_and_eval


class TestDatasets:
    def test_inputs_shape(self):
        assert INPUTS.shape == (4, 2)

    def test_all_datasets_present(self):
        assert set(DATASETS) == {"xor", "and", "or", "nand"}

    def test_output_shapes(self):
        for name, y in DATASETS.items():
            assert y.shape == (4, 1), f"{name} has wrong shape"

    def test_xor_values(self):
        expected = np.array([[0], [1], [1], [0]], dtype=np.float64)
        np.testing.assert_array_equal(DATASETS["xor"], expected)

    def test_and_values(self):
        expected = np.array([[0], [0], [0], [1]], dtype=np.float64)
        np.testing.assert_array_equal(DATASETS["and"], expected)

    def test_or_values(self):
        expected = np.array([[0], [1], [1], [1]], dtype=np.float64)
        np.testing.assert_array_equal(DATASETS["or"], expected)

    def test_nand_values(self):
        expected = np.array([[1], [1], [1], [0]], dtype=np.float64)
        np.testing.assert_array_equal(DATASETS["nand"], expected)


class TestTrainAndEval:
    def test_and_gate_sigmoid(self, capsys, tmp_path):
        save = str(tmp_path / "and_model.npz")
        ok = train_and_eval("and", DATASETS["and"], "sigmoid", "sgd", 1.0, 5000, save)
        assert ok is True
        assert (tmp_path / "and_model.npz").exists()

    def test_or_gate_sigmoid(self, capsys):
        ok = train_and_eval("or", DATASETS["or"], "sigmoid", "sgd", 1.0, 5000, None)
        assert ok is True

    def test_short_training_may_fail(self, capsys):
        # Very few epochs — XOR usually can't converge in 10 steps
        result = train_and_eval("xor", DATASETS["xor"], "sigmoid", "sgd", 0.5, 10, None)
        assert isinstance(result, bool)

    def test_output_contains_gate_name(self, capsys):
        train_and_eval("and", DATASETS["and"], "sigmoid", "sgd", 1.0, 100, None)
        captured = capsys.readouterr()
        assert "AND" in captured.out

    def test_adam_optimizer(self, capsys):
        ok = train_and_eval("and", DATASETS["and"], "sigmoid", "adam", 0.01, 3000, None)
        assert ok is True

    def test_tanh_activation(self, capsys):
        ok = train_and_eval("or", DATASETS["or"], "tanh", "sgd", 0.5, 3000, None)
        assert ok is True
