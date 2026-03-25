"""Unit tests for the NeuralNetwork class."""

import numpy as np
import pytest

from neural_network import NeuralNetwork


class TestNeuralNetworkInit:
    def test_requires_at_least_two_layers(self):
        with pytest.raises(ValueError, match="at least"):
            NeuralNetwork(layer_sizes=[5])

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
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        nn.forward(x)
        first_loss = nn.backward(y)

        for _ in range(500):
            nn.forward(x)
            nn.backward(y)

        nn.forward(x)
        later_loss = nn.backward(y)

        assert later_loss < first_loss


class TestXORLearning:
    def test_xor_convergence(self):
        """The network should learn XOR within 10,000 epochs."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        nn.train(x, y, epochs=10_000, log_every=0)
        preds = nn.predict(x)
        rounded = np.round(preds).astype(int)
        np.testing.assert_array_equal(rounded, y.astype(int))


class TestPredict:
    def test_predict_matches_forward(self):
        nn = NeuralNetwork([2, 3, 1], seed=1)
        x = np.array([[0.5, 0.5]])
        out_forward = nn.forward(x).copy()
        out_predict = nn.predict(x)
        np.testing.assert_array_almost_equal(out_forward, out_predict)
