"""Unit tests for the NeuralNetwork class."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neural_network import NeuralNetwork


# ── Fixtures ──────────────────────────────────────────────────────────
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=np.float64)
AND_Y = np.array([[0], [0], [0], [1]], dtype=np.float64)


class TestNeuralNetworkInit:
    def test_requires_at_least_two_layers(self):
        with pytest.raises(ValueError, match="at least"):
            NeuralNetwork(layer_sizes=[5])

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="activation"):
            NeuralNetwork([2, 3, 1], activation="softmax")

    def test_invalid_optimizer(self):
        with pytest.raises(ValueError, match="optimizer"):
            NeuralNetwork([2, 3, 1], optimizer="rmsprop")

    def test_invalid_loss(self):
        with pytest.raises(ValueError, match="loss"):
            NeuralNetwork([2, 3, 1], loss="huber")

    def test_weight_shapes(self):
        nn = NeuralNetwork([3, 4, 2])
        assert nn.weights[0].shape == (3, 4)
        assert nn.weights[1].shape == (4, 2)

    def test_bias_shapes(self):
        nn = NeuralNetwork([3, 4, 2])
        assert nn.biases[0].shape == (1, 4)
        assert nn.biases[1].shape == (1, 2)

    def test_seed_reproducibility(self):
        nn1 = NeuralNetwork([2, 3, 1], seed=7)
        nn2 = NeuralNetwork([2, 3, 1], seed=7)
        for w1, w2 in zip(nn1.weights, nn2.weights):
            np.testing.assert_array_equal(w1, w2)


class TestActivations:
    @pytest.mark.parametrize("act", ["sigmoid", "tanh", "relu"])
    def test_forward_runs(self, act):
        nn = NeuralNetwork([2, 4, 1], activation=act, seed=0)
        out = nn.forward(np.array([[0.5, 0.5]]))
        assert out.shape == (1, 1)

    def test_sigmoid_output_range(self):
        nn = NeuralNetwork([2, 4, 1], activation="sigmoid", seed=0)
        out = nn.forward(XOR_X)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_tanh_output_range(self):
        nn = NeuralNetwork([2, 4, 1], activation="tanh", seed=0)
        out = nn.forward(XOR_X)
        assert np.all(out >= -1) and np.all(out <= 1)

    def test_relu_non_negative(self):
        nn = NeuralNetwork([2, 4, 1], activation="relu", seed=0)
        out = nn.forward(XOR_X)
        assert np.all(out >= 0)


class TestForwardPass:
    def test_output_shape(self):
        nn = NeuralNetwork([2, 5, 1], seed=0)
        out = nn.forward(np.array([[0.0, 1.0]]))
        assert out.shape == (1, 1)

    def test_output_in_zero_one(self):
        nn = NeuralNetwork([2, 4, 1], seed=0)
        out = nn.forward(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert np.all(out >= 0) and np.all(out <= 1)


class TestBackwardPass:
    def test_loss_decreases(self):
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        nn.forward(XOR_X)
        first_loss = nn.backward(XOR_Y)
        for _ in range(500):
            nn.forward(XOR_X)
            nn.backward(XOR_Y)
        nn.forward(XOR_X)
        later_loss = nn.backward(XOR_Y)
        assert later_loss < first_loss


class TestOptimizers:
    @pytest.mark.parametrize("opt", ["sgd", "momentum", "adam"])
    def test_loss_decreases(self, opt):
        nn = NeuralNetwork([2, 4, 1], learning_rate=0.5, optimizer=opt, seed=42)
        nn.forward(XOR_X)
        first = nn.backward(XOR_Y)
        for _ in range(300):
            nn.forward(XOR_X)
            nn.backward(XOR_Y)
        nn.forward(XOR_X)
        last = nn.backward(XOR_Y)
        assert last < first


class TestXORLearning:
    def test_xor_convergence(self):
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=10_000, log_every=0)
        preds = nn.predict(XOR_X)
        rounded = np.round(preds).astype(int)
        np.testing.assert_array_equal(rounded, XOR_Y.astype(int))

    def test_and_convergence(self):
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        nn.train(XOR_X, AND_Y, epochs=5_000, log_every=0)
        preds = nn.predict(XOR_X)
        rounded = np.round(preds).astype(int)
        np.testing.assert_array_equal(rounded, AND_Y.astype(int))


class TestPredict:
    def test_predict_matches_forward(self):
        nn = NeuralNetwork([2, 3, 1], seed=1)
        x = np.array([[0.5, 0.5]])
        out_forward = nn.forward(x).copy()
        out_predict = nn.predict(x)
        np.testing.assert_array_almost_equal(out_forward, out_predict)


class TestSaveLoad:
    def test_round_trip(self):
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, activation="tanh",
                           optimizer="adam", seed=42)
        nn.train(XOR_X, XOR_Y, epochs=500, log_every=0)
        preds_before = nn.predict(XOR_X)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.npz"
            nn.save(path)
            loaded = NeuralNetwork.load(path)

        preds_after = loaded.predict(XOR_X)
        np.testing.assert_array_almost_equal(preds_before, preds_after)
        assert loaded.activation == "tanh"
        assert loaded.optimizer == "adam"
        assert loaded.layer_sizes == [2, 4, 1]

    def test_round_trip_softmax_cross_entropy(self):
        """Save/load preserves softmax_output and loss_fn settings."""
        nn = NeuralNetwork([4, 8, 3], learning_rate=0.01, activation="tanh",
                           optimizer="adam", loss="cross_entropy",
                           softmax_output=True, seed=42)
        x = np.random.default_rng(0).random((4, 4))
        y = np.eye(3, dtype=np.float64)[:3]  # 3 samples, pad
        y = np.vstack([y, np.eye(3, dtype=np.float64)[:1]])  # 4 samples
        nn.train(x, y, epochs=100, log_every=0)
        preds_before = nn.predict(x)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.npz"
            nn.save(path)
            loaded = NeuralNetwork.load(path)

        assert loaded.loss_fn == "cross_entropy"
        assert loaded.softmax_output is True
        preds_after = loaded.predict(x)
        np.testing.assert_array_almost_equal(preds_before, preds_after)


class TestSoftmaxCrossEntropy:
    """Tests for softmax output and cross-entropy loss."""

    def test_softmax_output_sums_to_one(self):
        nn = NeuralNetwork([4, 8, 3], softmax_output=True, seed=0)
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        out = nn.forward(x)
        np.testing.assert_almost_equal(np.sum(out, axis=1), [1.0])
        assert np.all(out >= 0)

    def test_cross_entropy_loss_decreases(self):
        nn = NeuralNetwork([4, 8, 3], learning_rate=0.05, activation="tanh",
                           optimizer="adam", loss="cross_entropy",
                           softmax_output=True, seed=42)
        rng = np.random.default_rng(0)
        x = rng.random((6, 4))
        # One-hot labels for 3 classes
        y = np.zeros((6, 3), dtype=np.float64)
        for i in range(6):
            y[i, i % 3] = 1.0

        nn.forward(x)
        first_loss = nn.backward(y)
        for _ in range(300):
            nn.forward(x)
            nn.backward(y)
        nn.forward(x)
        last_loss = nn.backward(y)
        assert last_loss < first_loss

    def test_multiclass_convergence(self):
        """Train a 3-class classifier and verify correct predictions."""
        nn = NeuralNetwork([4, 16, 3], learning_rate=0.01, activation="tanh",
                           optimizer="adam", loss="cross_entropy",
                           softmax_output=True, seed=42)
        # Simple clustering: each class in a different quadrant
        x = np.array([
            [1, 0, 0, 0], [0.9, 0.1, 0, 0],   # class 0
            [0, 1, 0, 0], [0.1, 0.9, 0, 0],   # class 1
            [0, 0, 1, 0], [0, 0.1, 0.9, 0],   # class 2
        ], dtype=np.float64)
        y = np.array([
            [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ], dtype=np.float64)

        nn.train(x, y, epochs=2000, log_every=0)
        preds = nn.predict(x)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(y, axis=1)
        np.testing.assert_array_equal(pred_classes, true_classes)
