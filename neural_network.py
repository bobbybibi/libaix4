"""
neural_network.py — A neural network built from scratch using only NumPy.

Implements:
  • Configurable multi-layer feed-forward network
  • Multiple activations: sigmoid, tanh, relu
  • Xavier/Glorot weight initialisation (He for ReLU)
  • Optimizers: SGD, Momentum, Adam
  • Model save / load (NumPy .npz)
  • Stochastic-gradient-descent back-propagation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ======================================================================
# Activation helpers
# ======================================================================
ACTIVATIONS = ("sigmoid", "tanh", "relu")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _sigmoid_deriv(a: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return a * (1.0 - a)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_deriv(a: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return 1.0 - a ** 2


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_deriv(a: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    return (z > 0).astype(z.dtype)


_ACT_FN = {"sigmoid": (_sigmoid, _sigmoid_deriv), "tanh": (_tanh, _tanh_deriv), "relu": (_relu, _relu_deriv)}

# ======================================================================
# Optimizer helpers
# ======================================================================
OPTIMIZERS = ("sgd", "momentum", "adam")


class NeuralNetwork:
    """A fully-connected feed-forward neural network.

    Parameters
    ----------
    layer_sizes : list[int]
        Number of neurons in each layer, e.g. ``[2, 4, 1]``.
    learning_rate : float
        Step size for gradient descent (default ``0.5``).
    activation : str
        ``"sigmoid"`` (default), ``"tanh"``, or ``"relu"``.
    optimizer : str
        ``"sgd"`` (default), ``"momentum"``, or ``"adam"``.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.5,
        activation: str = "sigmoid",
        optimizer: str = "sgd",
        seed: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Need at least an input and an output layer.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self._act_fn, self._act_deriv = _ACT_FN[activation]
        self._rng = np.random.default_rng(seed)

        # Weight initialisation (He for ReLU, Xavier otherwise)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            if activation == "relu":
                std = np.sqrt(2.0 / fan_in)
                w = self._rng.normal(0, std, size=(fan_in, fan_out))
            else:
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                w = self._rng.uniform(-limit, limit, size=(fan_in, fan_out))
            self.weights.append(w)
            self.biases.append(np.zeros((1, fan_out)))

        # Optimizer state
        self._step = 0
        self._vel_w: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        self._vel_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        if optimizer == "adam":
            self._m_w = [np.zeros_like(w) for w in self.weights]
            self._m_b = [np.zeros_like(b) for b in self.biases]
            self._v_w = [np.zeros_like(w) for w in self.weights]
            self._v_b = [np.zeros_like(b) for b in self.biases]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the network output for input ``x``."""
        self._activations: list[np.ndarray] = [x]
        self._zs: list[np.ndarray] = []
        a = x
        for w, b in zip(self.weights, self.biases):
            z = a @ w + b
            a = self._act_fn(z)
            self._zs.append(z)
            self._activations.append(a)
        return a

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, y: np.ndarray) -> float:
        """Back-propagate error and update weights. Returns MSE loss."""
        output = self._activations[-1]
        error = y - output
        loss = float(np.mean(error ** 2))
        n_layers = len(self.weights)
        batch = y.shape[0]

        deltas: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]
        deltas[-1] = error * self._act_deriv(output, self._zs[-1])

        for i in range(n_layers - 2, -1, -1):
            err_h = deltas[i + 1] @ self.weights[i + 1].T
            deltas[i] = err_h * self._act_deriv(self._activations[i + 1], self._zs[i])

        self._step += 1
        for i in range(n_layers):
            gw = self._activations[i].T @ deltas[i] / batch
            gb = np.sum(deltas[i], axis=0, keepdims=True) / batch
            self._apply_update(i, gw, gb)

        return loss

    def _apply_update(self, i: int, gw: np.ndarray, gb: np.ndarray) -> None:
        lr = self.learning_rate
        if self.optimizer == "sgd":
            self.weights[i] += lr * gw
            self.biases[i] += lr * gb
        elif self.optimizer == "momentum":
            mu = 0.9
            self._vel_w[i] = mu * self._vel_w[i] + lr * gw
            self._vel_b[i] = mu * self._vel_b[i] + lr * gb
            self.weights[i] += self._vel_w[i]
            self.biases[i] += self._vel_b[i]
        elif self.optimizer == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            self._m_w[i] = beta1 * self._m_w[i] + (1 - beta1) * gw
            self._m_b[i] = beta1 * self._m_b[i] + (1 - beta1) * gb
            self._v_w[i] = beta2 * self._v_w[i] + (1 - beta2) * gw ** 2
            self._v_b[i] = beta2 * self._v_b[i] + (1 - beta2) * gb ** 2
            mw_hat = self._m_w[i] / (1 - beta1 ** self._step)
            mb_hat = self._m_b[i] / (1 - beta1 ** self._step)
            vw_hat = self._v_w[i] / (1 - beta2 ** self._step)
            vb_hat = self._v_b[i] / (1 - beta2 ** self._step)
            self.weights[i] += lr * mw_hat / (np.sqrt(vw_hat) + eps)
            self.biases[i] += lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 10_000,
        log_every: int = 1_000,
    ) -> list[float]:
        """Train for *epochs* iterations. Returns per-epoch loss list."""
        losses: list[float] = []
        for epoch in range(1, epochs + 1):
            self.forward(x)
            loss = self.backward(y)
            losses.append(loss)
            if log_every and epoch % log_every == 0:
                print(f"Epoch {epoch:>6d}  |  Loss: {loss:.6f}")
        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Inference-only forward pass (no gradient storage)."""
        a = x
        for w, b in zip(self.weights, self.biases):
            z = a @ w + b
            a = self._act_fn(z)
        return a

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save model weights, biases, and config to a ``.npz`` file."""
        path = Path(path)
        arrays = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        meta = {
            "layer_sizes": self.layer_sizes,
            "learning_rate": self.learning_rate,
            "activation": self.activation,
            "optimizer": self.optimizer,
        }
        arrays["_meta"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        """Load a model from a ``.npz`` file previously saved with :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["_meta"]))
        nn = cls.__new__(cls)
        nn.layer_sizes = meta["layer_sizes"]
        nn.learning_rate = meta["learning_rate"]
        nn.activation = meta["activation"]
        nn.optimizer = meta["optimizer"]
        nn._act_fn, nn._act_deriv = _ACT_FN[nn.activation]
        nn._rng = np.random.default_rng()
        nn.weights = [data[f"w{i}"] for i in range(len(nn.layer_sizes) - 1)]
        nn.biases = [data[f"b{i}"] for i in range(len(nn.layer_sizes) - 1)]
        nn._step = 0
        nn._vel_w = [np.zeros_like(w) for w in nn.weights]
        nn._vel_b = [np.zeros_like(b) for b in nn.biases]
        if nn.optimizer == "adam":
            nn._m_w = [np.zeros_like(w) for w in nn.weights]
            nn._m_b = [np.zeros_like(b) for b in nn.biases]
            nn._v_w = [np.zeros_like(w) for w in nn.weights]
            nn._v_b = [np.zeros_like(b) for b in nn.biases]
        return nn
