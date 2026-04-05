"""
neural_network.py — A neural network built from scratch using only NumPy.

Implements:
  • Configurable multi-layer feed-forward network
  • Multiple activations: sigmoid, tanh, relu, leaky_relu, elu, swish, gelu
  • Softmax output layer for multi-class classification
  • Cross-entropy and MSE loss functions
  • Xavier/Glorot weight initialisation (He for ReLU variants)
  • Optimizers: SGD, Momentum, Adam
  • Early stopping with patience
  • Learning rate scheduling (step decay, cosine annealing) with warmup
  • Dropout regularization
  • L1/L2 weight regularization
  • Gradient clipping
  • Mini-batch training
  • Model save / load (NumPy .npz)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ======================================================================
# Activation helpers
# ======================================================================
ACTIVATIONS = ("sigmoid", "tanh", "relu", "leaky_relu", "elu", "swish", "gelu")


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


def _leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)


def _leaky_relu_deriv(a: np.ndarray, z: np.ndarray, alpha: float = 0.01) -> np.ndarray:  # noqa: ARG001
    return np.where(z > 0, 1.0, alpha).astype(z.dtype)


def _elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1.0))


def _elu_deriv(a: np.ndarray, z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(z > 0, 1.0, a + alpha).astype(z.dtype)


def _swish(z: np.ndarray) -> np.ndarray:
    sig = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return z * sig


def _swish_deriv(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    sig = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return a + sig * (1.0 - a)


def _gelu(z: np.ndarray) -> np.ndarray:
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))


def _gelu_deriv(a: np.ndarray, z: np.ndarray) -> np.ndarray:  # noqa: ARG001
    c = np.sqrt(2.0 / np.pi)
    inner = c * (z + 0.044715 * z ** 3)
    tanh_val = np.tanh(inner)
    sech2 = 1.0 - tanh_val ** 2
    d_inner = c * (1.0 + 0.134145 * z ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * z * sech2 * d_inner


def _softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


_ACT_FN = {
    "sigmoid": (_sigmoid, _sigmoid_deriv),
    "tanh": (_tanh, _tanh_deriv),
    "relu": (_relu, _relu_deriv),
    "leaky_relu": (_leaky_relu, _leaky_relu_deriv),
    "elu": (_elu, _elu_deriv),
    "swish": (_swish, _swish_deriv),
    "gelu": (_gelu, _gelu_deriv),
}

_HE_INIT_ACTIVATIONS = {"relu", "leaky_relu", "elu", "swish", "gelu"}

# ======================================================================
# Optimizer helpers
# ======================================================================
OPTIMIZERS = ("sgd", "momentum", "adam")
LOSS_FUNCTIONS = ("mse", "cross_entropy")
LR_SCHEDULES = ("step", "cosine", "cosine_restarts")
WEIGHT_INITS = ("auto", "he", "xavier", "lecun", "orthogonal")


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
        l2_lambda: float = 0.0,
        l1_lambda: float = 0.0,
        warmup_epochs: int = 0,
        batch_size: int | None = None,
        weight_init: str = "auto",
        batch_norm: bool = False,
        accumulation_steps: int = 1,
        label_smoothing: float = 0.0,
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
        if weight_init not in WEIGHT_INITS:
            raise ValueError(f"weight_init must be one of {WEIGHT_INITS}, got {weight_init!r}")

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
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.warmup_epochs = warmup_epochs
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.batch_norm = batch_norm
        self.accumulation_steps = max(1, accumulation_steps)
        self.label_smoothing = label_smoothing
        self._act_fn, self._act_deriv = _ACT_FN[activation]
        self._rng = np.random.default_rng(seed)
        self._masks: list[np.ndarray | None] = []
        self._pre_dropout: list[np.ndarray] = []
        self._training = False  # toggled during train()

        # Weight initialisation
        # "auto" picks He for ReLU-family, Xavier for sigmoid/tanh
        init = weight_init
        if init == "auto":
            init = "he" if activation in _HE_INIT_ACTIVATIONS else "xavier"

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            if init == "he":
                std = np.sqrt(2.0 / fan_in)
                w = self._rng.normal(0, std, size=(fan_in, fan_out))
            elif init == "lecun":
                std = np.sqrt(1.0 / fan_in)
                w = self._rng.normal(0, std, size=(fan_in, fan_out))
            elif init == "orthogonal":
                flat = self._rng.normal(0, 1, size=(fan_in, fan_out))
                u, _, vt = np.linalg.svd(flat, full_matrices=False)
                w = u if fan_in >= fan_out else vt
            else:  # xavier
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                w = self._rng.uniform(-limit, limit, size=(fan_in, fan_out))
            self.weights.append(w)
            self.biases.append(np.zeros((1, fan_out)))

        # Batch normalisation parameters (applied to hidden layers only)
        n_hidden = len(layer_sizes) - 2  # number of hidden layers
        if batch_norm and n_hidden > 0:
            self._bn_gamma: list[np.ndarray] = []
            self._bn_beta: list[np.ndarray] = []
            self._bn_running_mean: list[np.ndarray] = []
            self._bn_running_var: list[np.ndarray] = []
            for i in range(n_hidden):
                size = layer_sizes[i + 1]
                self._bn_gamma.append(np.ones((1, size)))
                self._bn_beta.append(np.zeros((1, size)))
                self._bn_running_mean.append(np.zeros((1, size)))
                self._bn_running_var.append(np.ones((1, size)))
        # BN cache for backward pass
        self._bn_cache: list[dict] = []

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
        self._bn_cache = []
        a = x
        is_output = lambda idx: idx == len(self.weights) - 1
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b

            # Batch normalisation on hidden layers (before activation)
            if self.batch_norm and not is_output(idx):
                z = self._bn_forward(idx, z)

            self._zs.append(z)
            # Apply softmax only on the last layer if enabled
            if self.softmax_output and is_output(idx):
                a = _softmax(z)
            else:
                a = self._act_fn(z)
            self._pre_dropout.append(a)
            # Dropout: apply to hidden layers only (not the output layer) during training
            if (
                self._training
                and self.dropout_rate > 0
                and not is_output(idx)
            ):
                mask = (self._rng.random(a.shape) > self.dropout_rate).astype(a.dtype)
                a = a * mask / (1.0 - self.dropout_rate)  # inverted dropout
                self._masks.append(mask)
            else:
                self._masks.append(None)
            self._activations.append(a)
        return a

    # ------------------------------------------------------------------
    # Batch normalisation helpers
    # ------------------------------------------------------------------
    def _bn_forward(self, layer_idx: int, z: np.ndarray) -> np.ndarray:
        """Apply batch normalisation to pre-activation ``z``."""
        gamma = self._bn_gamma[layer_idx]
        beta = self._bn_beta[layer_idx]
        eps = 1e-5
        momentum = 0.1

        if self._training:
            mean = np.mean(z, axis=0, keepdims=True)
            var = np.var(z, axis=0, keepdims=True)
            z_norm = (z - mean) / np.sqrt(var + eps)
            out = gamma * z_norm + beta
            # Update running stats
            self._bn_running_mean[layer_idx] = (
                (1 - momentum) * self._bn_running_mean[layer_idx] + momentum * mean
            )
            self._bn_running_var[layer_idx] = (
                (1 - momentum) * self._bn_running_var[layer_idx] + momentum * var
            )
            self._bn_cache.append({"z": z, "mean": mean, "var": var, "z_norm": z_norm, "idx": layer_idx})
        else:
            mean = self._bn_running_mean[layer_idx]
            var = self._bn_running_var[layer_idx]
            z_norm = (z - mean) / np.sqrt(var + eps)
            out = gamma * z_norm + beta
            self._bn_cache.append(None)
        return out

    def _bn_backward(self, cache_idx: int, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass through batch normalisation.

        Returns (dz, dgamma, dbeta).
        """
        cache = self._bn_cache[cache_idx]
        z_norm = cache["z_norm"]
        mean = cache["mean"]
        var = cache["var"]
        layer_idx = cache["idx"]
        gamma = self._bn_gamma[layer_idx]
        eps = 1e-5
        N = dout.shape[0]

        dgamma = np.sum(dout * z_norm, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        dz_norm = dout * gamma
        inv_std = 1.0 / np.sqrt(var + eps)
        dz = (1.0 / N) * inv_std * (
            N * dz_norm - np.sum(dz_norm, axis=0, keepdims=True)
            - z_norm * np.sum(dz_norm * z_norm, axis=0, keepdims=True)
        )
        return dz, dgamma, dbeta

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, y: np.ndarray, accumulate: bool = False) -> float:
        """Back-propagate error and update weights. Returns loss.

        Parameters
        ----------
        accumulate : bool
            If ``True``, compute and store gradients in ``_accum_gw`` /
            ``_accum_gb`` without applying weight updates.  Call
            :meth:`_apply_accumulated` afterwards to commit them.
        """
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
            # Apply dropout mask matching the forward-pass mask for layer i
            mask = self._masks[i] if i < len(self._masks) else None
            if mask is not None:
                err_h = err_h * mask / (1.0 - self.dropout_rate)  # inverted dropout
            # Use pre-dropout activation for derivative computation
            act_for_deriv = self._pre_dropout[i]
            delta = err_h * self._act_deriv(act_for_deriv, self._zs[i])
            # Batch normalisation backward
            if self.batch_norm and i < len(self.weights) - 1:
                dz, dgamma, dbeta = self._bn_backward(i, delta)
                if not accumulate:
                    self._bn_gamma[i] -= self.learning_rate * dgamma
                    self._bn_beta[i] -= self.learning_rate * dbeta
                delta = dz
            deltas[i] = delta

        # Add regularisation penalty to reported loss
        if self.l2_lambda > 0:
            l2_term = 0.5 * self.l2_lambda * sum(float(np.sum(w ** 2)) for w in self.weights)
            loss += l2_term
        if self.l1_lambda > 0:
            l1_term = self.l1_lambda * sum(float(np.sum(np.abs(w))) for w in self.weights)
            loss += l1_term

        sign = -1.0 if (self.loss_fn == "cross_entropy" or self.softmax_output) else 1.0

        if accumulate:
            # Accumulate gradients
            for i in range(n_layers):
                gw = sign * (self._activations[i].T @ deltas[i]) / batch
                gb = sign * np.sum(deltas[i], axis=0, keepdims=True) / batch
                self._accum_gw[i] += gw
                self._accum_gb[i] += gb
            self._accum_count += 1
        else:
            self._step += 1
            for i in range(n_layers):
                gw = sign * (self._activations[i].T @ deltas[i]) / batch
                gb = sign * np.sum(deltas[i], axis=0, keepdims=True) / batch
                self._apply_update(i, gw, gb)

        return loss

    def _init_accumulators(self) -> None:
        """Initialise / reset gradient accumulators."""
        self._accum_gw = [np.zeros_like(w) for w in self.weights]
        self._accum_gb = [np.zeros_like(b) for b in self.biases]
        self._accum_count = 0

    def _apply_accumulated(self) -> None:
        """Apply the averaged accumulated gradients and reset accumulators."""
        if self._accum_count == 0:
            return
        self._step += 1
        for i in range(len(self.weights)):
            gw = self._accum_gw[i] / self._accum_count
            gb = self._accum_gb[i] / self._accum_count
            self._apply_update(i, gw, gb)
        self._init_accumulators()

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

        # Weight decay (L2) and sparsity (L1) — applied to weights only
        if self.l2_lambda > 0:
            self.weights[i] -= lr * self.l2_lambda * self.weights[i]
        if self.l1_lambda > 0:
            self.weights[i] -= lr * self.l1_lambda * np.sign(self.weights[i])
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
        # Warmup phase: linearly ramp from 0 to _base_lr
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            self.learning_rate = self._base_lr * (epoch / self.warmup_epochs)
            return

        if self.lr_schedule is None:
            return
        if self.lr_schedule == "step":
            n_drops = epoch // self.lr_step_every
            self.learning_rate = self._base_lr * (0.5 ** n_drops)
        elif self.lr_schedule == "cosine":
            self.learning_rate = self._base_lr * 0.5 * (
                1 + np.cos(np.pi * epoch / total_epochs)
            )
        elif self.lr_schedule == "cosine_restarts":
            # Cosine annealing with warm restarts (SGDR, T_mult=2)
            T_0 = max(self.lr_step_every, 1)
            T_cur = epoch
            T_i = T_0
            while T_cur >= T_i:
                T_cur -= T_i
                T_i *= 2
            self.learning_rate = self._base_lr * 0.5 * (
                1 + np.cos(np.pi * T_cur / T_i)
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

        When *early_stopping* is ``True`` the method automatically saves and
        restores the best weights observed during training.
        """
        losses: list[float] = []
        best_loss = float("inf")
        wait = 0
        self._training = True
        val_losses: list[float] = []
        lr_history: list[float] = []

        # Apply label smoothing (only for classification with multiple classes)
        if self.label_smoothing > 0 and y.shape[1] > 1:
            n_classes = y.shape[1]
            y = y * (1.0 - self.label_smoothing) + self.label_smoothing / n_classes

        # Snapshot best weights for restore-on-early-stop
        best_weights: list[np.ndarray] | None = None
        best_biases: list[np.ndarray] | None = None
        best_bn: tuple | None = None

        batch_size = self.batch_size
        use_accum = self.accumulation_steps > 1

        for epoch in range(1, epochs + 1):
            self._update_lr(epoch, epochs)

            if batch_size is not None and batch_size < x.shape[0]:
                # Mini-batch training (with optional gradient accumulation)
                indices = self._rng.permutation(x.shape[0])
                epoch_loss = 0.0
                n_batches = 0
                if use_accum:
                    self._init_accumulators()
                for start in range(0, x.shape[0], batch_size):
                    batch_idx = indices[start : start + batch_size]
                    self.forward(x[batch_idx])
                    if use_accum:
                        epoch_loss += self.backward(y[batch_idx], accumulate=True)
                        n_batches += 1
                        if n_batches % self.accumulation_steps == 0:
                            self._apply_accumulated()
                    else:
                        epoch_loss += self.backward(y[batch_idx])
                        n_batches += 1
                # Flush remaining accumulated gradients
                if use_accum and self._accum_count > 0:
                    self._apply_accumulated()
                loss = epoch_loss / max(n_batches, 1)
            else:
                self.forward(x)
                loss = self.backward(y)

            losses.append(loss)
            lr_history.append(self.learning_rate)
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
                    val_losses.append(monitor_loss)
                    self._training = True
                else:
                    monitor_loss = loss

                if monitor_loss < best_loss - min_delta:
                    best_loss = monitor_loss
                    wait = 0
                    # Checkpoint best weights
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    best_bn = None
                    if self.batch_norm:
                        best_bn = (
                            [g.copy() for g in self._bn_gamma],
                            [b_.copy() for b_ in self._bn_beta],
                            [m.copy() for m in self._bn_running_mean],
                            [v.copy() for v in self._bn_running_var],
                        )
                else:
                    wait += 1
                    if wait >= patience:
                        if log_every:
                            print(f"Early stopping at epoch {epoch} (best loss: {best_loss:.6f})")
                        break

        # Restore best weights if we checkpointed any
        if early_stopping and best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
            if self.batch_norm and best_bn is not None:
                self._bn_gamma, self._bn_beta = best_bn[0], best_bn[1]
                self._bn_running_mean, self._bn_running_var = best_bn[2], best_bn[3]

        self._training = False

        # Store training history for inspection
        self.history = {
            "train_loss": losses,
            "lr": lr_history,
        }
        if val_losses:
            self.history["val_loss"] = val_losses

        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Inference-only forward pass (no gradient storage)."""
        a = x
        is_output = lambda idx: idx == len(self.weights) - 1
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if self.batch_norm and not is_output(idx):
                # Use running stats for inference BN
                eps = 1e-5
                mean = self._bn_running_mean[idx]
                var = self._bn_running_var[idx]
                z = self._bn_gamma[idx] * ((z - mean) / np.sqrt(var + eps)) + self._bn_beta[idx]
            if self.softmax_output and is_output(idx):
                a = _softmax(z)
            else:
                a = self._act_fn(z)
        return a

    def lr_find(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr_min: float = 1e-7,
        lr_max: float = 10.0,
        steps: int = 100,
    ) -> tuple[float, list[float], list[float]]:
        """Run a learning-rate range test (Smith 2017).

        Gradually increases the learning rate from *lr_min* to *lr_max*
        over *steps* iterations and records the loss.  Returns
        ``(suggested_lr, lrs, losses)`` where *suggested_lr* is the LR
        one order of magnitude before the minimum loss.

        The method restores the original weights afterwards so the model
        is unchanged.
        """
        # Snapshot weights
        saved_w = [w.copy() for w in self.weights]
        saved_b = [b.copy() for b in self.biases]
        saved_lr = self.learning_rate

        mult = (lr_max / lr_min) ** (1 / steps)
        lr = lr_min
        lrs: list[float] = []
        losses: list[float] = []
        best_loss_val = float("inf")
        best_lr = lr_min

        self._training = True
        for _ in range(steps):
            self.learning_rate = lr
            self.forward(x)
            loss = self.backward(y)
            lrs.append(lr)
            losses.append(loss)
            if loss < best_loss_val:
                best_loss_val = loss
                best_lr = lr
            # If loss has diverged (>4x best), stop
            if loss > 4 * best_loss_val:
                break
            lr *= mult
        self._training = False

        # Restore weights
        self.weights = saved_w
        self.biases = saved_b
        self.learning_rate = saved_lr

        # Suggest LR one order of magnitude below the minimum-loss LR
        suggested = best_lr / 10.0
        return suggested, lrs, losses

    def calibrate_temperature(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Learn a temperature scaling parameter on validation data.

        Uses gradient descent to minimise negative-log-likelihood on
        *x_val* / *y_val*.  Stores the learned temperature in
        ``self._temperature`` and returns it.
        """
        if not self.softmax_output:
            raise ValueError("Temperature scaling requires softmax_output=True")

        # Get logits (pre-softmax values) for validation set
        logits = self._get_logits(x_val)

        T = 1.0
        for _ in range(max_iter):
            scaled = logits / T
            probs = _softmax(scaled)
            # NLL gradient w.r.t. T
            eps = 1e-12
            # dNLL/dT = (1/N) * sum_i (p_i - y_i) @ logits_i / T^2
            grad = -np.mean(np.sum((probs - y_val) * logits, axis=1)) / (T ** 2)
            T -= lr * grad
            T = max(T, 0.01)  # keep temperature positive

        self._temperature = T
        return T

    def _get_logits(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning raw logits (pre-softmax)."""
        a = x
        is_output = lambda idx: idx == len(self.weights) - 1
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if self.batch_norm and not is_output(idx):
                eps = 1e-5
                mean = self._bn_running_mean[idx]
                var = self._bn_running_var[idx]
                z = self._bn_gamma[idx] * ((z - mean) / np.sqrt(var + eps)) + self._bn_beta[idx]
            if is_output(idx):
                return z  # raw logits
            a = self._act_fn(z)
        return a  # fallback (single-layer network)

    def predict_calibrated(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities scaled by the learned temperature."""
        T = getattr(self, "_temperature", 1.0)
        logits = self._get_logits(x)
        return _softmax(logits / T)

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
            "l2_lambda": self.l2_lambda,
            "l1_lambda": self.l1_lambda,
            "warmup_epochs": self.warmup_epochs,
            "batch_size": self.batch_size,
            "weight_init": self.weight_init,
            "batch_norm": self.batch_norm,
            "accumulation_steps": self.accumulation_steps,
            "label_smoothing": self.label_smoothing,
            "temperature": getattr(self, "_temperature", None),
        }
        if self.batch_norm:
            n_hidden = len(self.layer_sizes) - 2
            for i in range(n_hidden):
                arrays[f"bn_gamma{i}"] = self._bn_gamma[i]
                arrays[f"bn_beta{i}"] = self._bn_beta[i]
                arrays[f"bn_rmean{i}"] = self._bn_running_mean[i]
                arrays[f"bn_rvar{i}"] = self._bn_running_var[i]
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
        nn.l2_lambda = meta.get("l2_lambda", 0.0)
        nn.l1_lambda = meta.get("l1_lambda", 0.0)
        nn.warmup_epochs = meta.get("warmup_epochs", 0)
        nn.batch_size = meta.get("batch_size", None)
        nn.weight_init = meta.get("weight_init", "auto")
        nn.batch_norm = meta.get("batch_norm", False)
        nn.accumulation_steps = meta.get("accumulation_steps", 1)
        nn.label_smoothing = meta.get("label_smoothing", 0.0)
        temp = meta.get("temperature", None)
        if temp is not None:
            nn._temperature = temp
        nn._act_fn, nn._act_deriv = _ACT_FN[nn.activation]
        nn._rng = np.random.default_rng()
        nn._training = False
        nn._masks = []
        nn._pre_dropout = []
        nn._bn_cache = []
        nn.weights = [data[f"w{i}"] for i in range(len(nn.layer_sizes) - 1)]
        nn.biases = [data[f"b{i}"] for i in range(len(nn.layer_sizes) - 1)]
        if nn.batch_norm:
            n_hidden = len(nn.layer_sizes) - 2
            nn._bn_gamma = [data[f"bn_gamma{i}"] for i in range(n_hidden)]
            nn._bn_beta = [data[f"bn_beta{i}"] for i in range(n_hidden)]
            nn._bn_running_mean = [data[f"bn_rmean{i}"] for i in range(n_hidden)]
            nn._bn_running_var = [data[f"bn_rvar{i}"] for i in range(n_hidden)]
        nn._step = 0
        nn._vel_w = [np.zeros_like(w) for w in nn.weights]
        nn._vel_b = [np.zeros_like(b) for b in nn.biases]
        if nn.optimizer == "adam":
            nn._m_w = [np.zeros_like(w) for w in nn.weights]
            nn._m_b = [np.zeros_like(b) for b in nn.biases]
            nn._v_w = [np.zeros_like(w) for w in nn.weights]
            nn._v_b = [np.zeros_like(b) for b in nn.biases]
        return nn
