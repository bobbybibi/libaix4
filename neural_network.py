"""
neural_network.py — A neural network built from scratch using only NumPy.

Implements:
  • Configurable multi-layer feed-forward network
  • Multiple activations: sigmoid, tanh, relu
  • Softmax output layer for multi-class classification
  • Cross-entropy and MSE loss functions
  • Xavier/Glorot weight initialisation (He for ReLU)
  • Optimizers: SGD, Momentum, Adam
  • Early stopping with patience
  • Learning rate scheduling (step decay, cosine annealing)
  • Dropout regularization
  • Gradient clipping
  • Model save / load (NumPy .npz)
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


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


_ACT_FN = {
    "sigmoid": (_sigmoid, _sigmoid_deriv),
    "tanh": (_tanh, _tanh_deriv),
    "relu": (_relu, _relu_deriv),
}

# ======================================================================
# Optimizer helpers
# ======================================================================
OPTIMIZERS = ("sgd", "momentum", "adam")
LOSS_FUNCTIONS = ("mse", "cross_entropy")
LR_SCHEDULES = ("step", "cosine")


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
    loss : str
        ``"mse"`` (default) or ``"cross_entropy"`` (for classification).
    softmax_output : bool
        Apply softmax on the output layer (default ``False``).
    seed : int | None
        Random seed for reproducibility.
    dropout_rate : float
        Fraction of neurons to drop during training (default ``0.0`` = no dropout).
    grad_clip : float | None
        Maximum gradient norm. ``None`` disables clipping (default).
    lr_schedule : str | None
        Learning rate schedule: ``"step"`` (halve every ``lr_step_every`` epochs),
        ``"cosine"`` (cosine annealing to 0), or ``None`` (constant).
    lr_step_every : int
        Epoch interval for ``"step"`` schedule (default ``1000``).
    """

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.5,
        activation: str = "sigmoid",
        optimizer: str = "sgd",
        loss: str = "mse",
        softmax_output: bool = False,
        seed: int | None = None,
        dropout_rate: float = 0.0,
        grad_clip: float | None = None,
        lr_schedule: str | None = None,
        lr_step_every: int = 1000,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Need at least an input and an output layer.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
        if loss not in LOSS_FUNCTIONS:
            raise ValueError(f"loss must be one of {LOSS_FUNCTIONS}, got {loss!r}")
        if lr_schedule is not None and lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {lr_schedule!r}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate!r}")

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self._base_lr = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss_fn = loss
        self.softmax_output = softmax_output
        self.dropout_rate = dropout_rate
        self.grad_clip = grad_clip
        self.lr_schedule = lr_schedule
        self.lr_step_every = lr_step_every
        self._act_fn, self._act_deriv = _ACT_FN[activation]
        self._rng = np.random.default_rng(seed)
        self._masks: list[np.ndarray | None] = []
        self._pre_dropout: list[np.ndarray] = []
        self._training = False  # toggled during train()

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
        self._pre_dropout: list[np.ndarray] = []  # activations before dropout
        self._zs: list[np.ndarray] = []
        self._masks: list[np.ndarray | None] = []
        a = x
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            self._zs.append(z)
            # Apply softmax only on the last layer if enabled
            if self.softmax_output and idx == len(self.weights) - 1:
                a = _softmax(z)
            else:
                a = self._act_fn(z)
            self._pre_dropout.append(a)
            # Dropout: apply to hidden layers only (not the output layer) during training
            if (
                self._training
                and self.dropout_rate > 0
                and idx < len(self.weights) - 1
            ):
                mask = (self._rng.random(a.shape) > self.dropout_rate).astype(a.dtype)
                a = a * mask / (1.0 - self.dropout_rate)  # inverted dropout
                self._masks.append(mask)
            else:
                self._masks.append(None)
            self._activations.append(a)
        return a

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, y: np.ndarray) -> float:
        """Back-propagate error and update weights. Returns loss."""
        output = self._activations[-1]
        n_layers = len(self.weights)
        batch = y.shape[0]

        # Compute loss and output-layer delta
        if self.loss_fn == "cross_entropy":
            eps = 1e-12
            clipped = np.clip(output, eps, 1.0 - eps)
            loss = float(-np.mean(np.sum(y * np.log(clipped), axis=1)))
            # For softmax + cross-entropy, the gradient simplifies
            if self.softmax_output:
                deltas: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]
                deltas[-1] = output - y
            else:
                deltas = [None] * n_layers  # type: ignore[list-item]
                deltas[-1] = (output - y) * self._act_deriv(output, self._zs[-1])
        else:
            error = y - output
            loss = float(np.mean(error ** 2))
            deltas = [None] * n_layers  # type: ignore[list-item]
            if self.softmax_output:
                deltas[-1] = output - y
            else:
                deltas[-1] = error * self._act_deriv(output, self._zs[-1])

        for i in range(n_layers - 2, -1, -1):
            err_h = deltas[i + 1] @ self.weights[i + 1].T
            # Apply dropout mask to error signal (same mask as forward pass)
            mask = self._masks[i + 1] if i + 1 < len(self._masks) else None
            if mask is not None:
                err_h = err_h * mask / (1.0 - self.dropout_rate)
            # Use pre-dropout activation for derivative computation
            act_for_deriv = self._pre_dropout[i]
            deltas[i] = err_h * self._act_deriv(act_for_deriv, self._zs[i])

        self._step += 1
        sign = -1.0 if (self.loss_fn == "cross_entropy" or self.softmax_output) else 1.0
        for i in range(n_layers):
            gw = sign * (self._activations[i].T @ deltas[i]) / batch
            gb = sign * np.sum(deltas[i], axis=0, keepdims=True) / batch
            self._apply_update(i, gw, gb)

        return loss

    def _apply_update(self, i: int, gw: np.ndarray, gb: np.ndarray) -> None:
        # Gradient clipping
        if self.grad_clip is not None:
            gw_norm = np.linalg.norm(gw)
            if gw_norm > self.grad_clip:
                gw = gw * (self.grad_clip / gw_norm)
            gb_norm = np.linalg.norm(gb)
            if gb_norm > self.grad_clip:
                gb = gb * (self.grad_clip / gb_norm)

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
    # Learning rate scheduling
    # ------------------------------------------------------------------
    def _update_lr(self, epoch: int, total_epochs: int) -> None:
        """Adjust learning rate according to the chosen schedule."""
        if self.lr_schedule is None:
            return
        if self.lr_schedule == "step":
            n_drops = epoch // self.lr_step_every
            self.learning_rate = self._base_lr * (0.5 ** n_drops)
        elif self.lr_schedule == "cosine":
            self.learning_rate = self._base_lr * 0.5 * (
                1 + np.cos(np.pi * epoch / total_epochs)
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 10_000,
        log_every: int = 1_000,
        early_stopping: bool = False,
        patience: int = 500,
        min_delta: float = 1e-6,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> list[float]:
        """Train for *epochs* iterations. Returns per-epoch loss list.

        Parameters
        ----------
        early_stopping : bool
            Stop when loss hasn't improved by *min_delta* for *patience* epochs.
        patience : int
            Number of epochs to wait for improvement before stopping.
        min_delta : float
            Minimum decrease in loss to count as improvement.
        x_val, y_val : np.ndarray | None
            Optional validation data. If provided, early stopping monitors
            validation loss instead of training loss.
        """
        losses: list[float] = []
        best_loss = float("inf")
        wait = 0
        self._training = True

        for epoch in range(1, epochs + 1):
            self._update_lr(epoch, epochs)
            self.forward(x)
            loss = self.backward(y)
            losses.append(loss)
            if log_every and epoch % log_every == 0:
                print(f"Epoch {epoch:>6d}  |  Loss: {loss:.6f}")

            # Early stopping check
            if early_stopping:
                if x_val is not None and y_val is not None:
                    self._training = False
                    self.forward(x_val)
                    output = self._activations[-1]
                    if self.loss_fn == "cross_entropy":
                        eps = 1e-12
                        clipped = np.clip(output, eps, 1.0 - eps)
                        monitor_loss = float(-np.mean(np.sum(y_val * np.log(clipped), axis=1)))
                    else:
                        monitor_loss = float(np.mean((y_val - output) ** 2))
                    self._training = True
                else:
                    monitor_loss = loss

                if monitor_loss < best_loss - min_delta:
                    best_loss = monitor_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        if log_every:
                            print(f"Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
                        break

        self._training = False
        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Inference-only forward pass (no gradient storage)."""
        a = x
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if self.softmax_output and idx == len(self.weights) - 1:
                a = _softmax(z)
            else:
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
            "learning_rate": self._base_lr,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "softmax_output": self.softmax_output,
            "dropout_rate": self.dropout_rate,
            "grad_clip": self.grad_clip,
            "lr_schedule": self.lr_schedule,
            "lr_step_every": self.lr_step_every,
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
        nn._base_lr = meta["learning_rate"]
        nn.activation = meta["activation"]
        nn.optimizer = meta["optimizer"]
        nn.loss_fn = meta.get("loss_fn", "mse")
        nn.softmax_output = meta.get("softmax_output", False)
        nn.dropout_rate = meta.get("dropout_rate", 0.0)
        nn.grad_clip = meta.get("grad_clip", None)
        nn.lr_schedule = meta.get("lr_schedule", None)
        nn.lr_step_every = meta.get("lr_step_every", 1000)
        nn._act_fn, nn._act_deriv = _ACT_FN[nn.activation]
        nn._rng = np.random.default_rng()
        nn._training = False
        nn._masks = []
        nn._pre_dropout = []
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
