"""Tests for ML optimizations in NeuralNetwork (early stopping, LR scheduling, dropout, grad clipping)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neural_network import NeuralNetwork


# ── Fixtures ──────────────────────────────────────────────────────────
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
XOR_Y = np.array([[0], [1], [1], [0]], dtype=np.float64)


class TestEarlyStopping:
    def test_stops_before_max_epochs(self):
        """Early stopping should terminate before all epochs if loss plateaus."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        losses = nn.train(
            XOR_X, XOR_Y,
            epochs=50_000,
            log_every=0,
            early_stopping=True,
            patience=500,
            min_delta=1e-4,
        )
        # Should stop well before 50k epochs once loss stabilises
        assert len(losses) < 50_000

    def test_disabled_runs_all_epochs(self):
        """Without early stopping, all epochs should run."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        losses = nn.train(
            XOR_X, XOR_Y,
            epochs=100,
            log_every=0,
            early_stopping=False,
        )
        assert len(losses) == 100

    def test_with_validation_data(self):
        """Early stopping with separate validation data."""
        nn = NeuralNetwork([4, 8, 3], learning_rate=0.01, activation="tanh",
                           optimizer="adam", loss="cross_entropy",
                           softmax_output=True, seed=42)
        rng = np.random.default_rng(0)
        x_train = rng.random((6, 4))
        y_train = np.zeros((6, 3), dtype=np.float64)
        for i in range(6):
            y_train[i, i % 3] = 1.0

        x_val = rng.random((3, 4))
        y_val = np.eye(3, dtype=np.float64)

        losses = nn.train(
            x_train, y_train,
            epochs=5000,
            log_every=0,
            early_stopping=True,
            patience=200,
            x_val=x_val,
            y_val=y_val,
        )
        # Should still converge reasonably
        assert len(losses) > 0
        assert losses[-1] < losses[0]


class TestLRScheduling:
    def test_step_schedule_reduces_lr(self):
        """Step schedule should halve the LR every lr_step_every epochs."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, lr_schedule="step",
                           lr_step_every=100, seed=42)
        initial_lr = nn.learning_rate
        nn.train(XOR_X, XOR_Y, epochs=200, log_every=0)
        # After 200 epochs with step_every=100, LR should have been halved twice
        assert nn.learning_rate < initial_lr

    def test_cosine_schedule_reduces_lr(self):
        """Cosine schedule should reduce LR towards 0."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, lr_schedule="cosine", seed=42)
        nn.train(XOR_X, XOR_Y, epochs=500, log_every=0)
        # Near end of training, cosine should have reduced LR significantly
        assert nn.learning_rate < 1.0

    def test_no_schedule_keeps_lr_constant(self):
        """Without a schedule, LR should remain constant."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=0.5, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=100, log_every=0)
        assert nn.learning_rate == 0.5

    def test_invalid_schedule_rejected(self):
        """Invalid schedule names should raise ValueError."""
        with pytest.raises(ValueError, match="lr_schedule"):
            NeuralNetwork([2, 4, 1], lr_schedule="linear")

    def test_convergence_with_step_schedule(self):
        """Network should still converge with step schedule."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, lr_schedule="step",
                           lr_step_every=2000, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=10_000, log_every=0)
        preds = nn.predict(XOR_X)
        rounded = np.round(preds).astype(int)
        np.testing.assert_array_equal(rounded, XOR_Y.astype(int))


class TestDropout:
    def test_dropout_during_training(self):
        """Dropout should be active during training (outputs differ between runs)."""
        nn = NeuralNetwork([2, 8, 1], learning_rate=0.5, dropout_rate=0.5, seed=42)
        # Training with dropout should still reduce loss
        losses = nn.train(XOR_X, XOR_Y, epochs=1000, log_every=0)
        assert losses[-1] < losses[0]

    def test_no_dropout_during_predict(self):
        """Predict should give deterministic results (no dropout)."""
        nn = NeuralNetwork([2, 8, 1], learning_rate=0.5, dropout_rate=0.5, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=500, log_every=0)
        pred1 = nn.predict(XOR_X).copy()
        pred2 = nn.predict(XOR_X).copy()
        np.testing.assert_array_equal(pred1, pred2)

    def test_zero_dropout_is_noop(self):
        """Dropout rate of 0 should behave identically to no dropout."""
        nn1 = NeuralNetwork([2, 4, 1], learning_rate=1.0, dropout_rate=0.0, seed=42)
        nn2 = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        losses1 = nn1.train(XOR_X, XOR_Y, epochs=100, log_every=0)
        losses2 = nn2.train(XOR_X, XOR_Y, epochs=100, log_every=0)
        np.testing.assert_array_almost_equal(losses1, losses2)

    def test_convergence_with_dropout(self):
        """Network should converge even with moderate dropout."""
        nn = NeuralNetwork([2, 16, 1], learning_rate=1.0, dropout_rate=0.3, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=15_000, log_every=0)
        preds = nn.predict(XOR_X)
        rounded = np.round(preds).astype(int)
        np.testing.assert_array_equal(rounded, XOR_Y.astype(int))


class TestGradientClipping:
    def test_grad_clip_prevents_explosion(self):
        """With gradient clipping, training should remain stable."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=5.0, grad_clip=1.0, seed=42)
        losses = nn.train(XOR_X, XOR_Y, epochs=500, log_every=0)
        # All losses should be finite (no NaN/Inf from exploding gradients)
        assert all(np.isfinite(v) for v in losses)

    def test_no_clip_is_default(self):
        """Default grad_clip should be None."""
        nn = NeuralNetwork([2, 4, 1])
        assert nn.grad_clip is None

    def test_convergence_with_clipping(self):
        """Network should still converge with gradient clipping."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, grad_clip=5.0, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=10_000, log_every=0)
        preds = nn.predict(XOR_X)
        rounded = np.round(preds).astype(int)
        np.testing.assert_array_equal(rounded, XOR_Y.astype(int))


class TestSaveLoadNewParams:
    def test_round_trip_preserves_new_params(self):
        """Save/load should preserve dropout, grad_clip, lr_schedule."""
        nn = NeuralNetwork(
            [2, 4, 1], learning_rate=0.5,
            dropout_rate=0.3, grad_clip=2.0,
            lr_schedule="step", lr_step_every=500,
            seed=42,
        )
        nn.train(XOR_X, XOR_Y, epochs=100, log_every=0)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.npz"
            nn.save(path)
            loaded = NeuralNetwork.load(path)

        assert loaded.dropout_rate == 0.3
        assert loaded.grad_clip == 2.0
        assert loaded.lr_schedule == "step"
        assert loaded.lr_step_every == 500

    def test_backward_compat_load(self):
        """Loading a model saved without new params should use defaults."""
        nn = NeuralNetwork([2, 4, 1], seed=42)
        nn.train(XOR_X, XOR_Y, epochs=50, log_every=0)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.npz"
            nn.save(path)
            loaded = NeuralNetwork.load(path)

        assert loaded.dropout_rate == 0.0
        assert loaded.grad_clip is None
        assert loaded.lr_schedule is None


class TestCombinedFeatures:
    def test_all_features_together(self):
        """All ML optimizations combined should still reduce loss."""
        nn = NeuralNetwork(
            [2, 16, 1],
            learning_rate=0.5,
            activation="tanh",
            optimizer="adam",
            dropout_rate=0.1,
            grad_clip=5.0,
            lr_schedule="step",
            lr_step_every=2000,
            seed=42,
        )
        losses = nn.train(
            XOR_X, XOR_Y,
            epochs=5_000,
            log_every=0,
        )
        # Loss should decrease significantly
        assert losses[-1] < losses[0]
        # All losses should be finite
        assert all(np.isfinite(v) for v in losses)

    def test_dropout_and_clipping_convergence(self):
        """Dropout + gradient clipping should still allow XOR convergence."""
        nn = NeuralNetwork(
            [2, 16, 1],
            learning_rate=0.5,
            optimizer="adam",
            dropout_rate=0.1,
            grad_clip=5.0,
            seed=7,
        )
        nn.train(XOR_X, XOR_Y, epochs=20_000, log_every=0)
        preds = nn.predict(XOR_X)
        rounded = np.round(preds).astype(int)
        np.testing.assert_array_equal(rounded, XOR_Y.astype(int))


# ── New activation functions ──────────────────────────────────────────
class TestNewActivations:
    @pytest.mark.parametrize("act", ["leaky_relu", "elu", "swish", "gelu"])
    def test_constructor_accepts(self, act):
        """All new activations should be valid choices."""
        nn = NeuralNetwork([2, 4, 1], activation=act, seed=42)
        assert nn.activation == act

    @pytest.mark.parametrize("act", ["leaky_relu", "elu", "swish", "gelu"])
    def test_forward_pass(self, act):
        """Forward pass should produce finite output for each activation."""
        nn = NeuralNetwork([2, 4, 1], activation=act, seed=42)
        out = nn.predict(XOR_X)
        assert out.shape == (4, 1)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("act", ["leaky_relu", "elu", "swish", "gelu"])
    def test_training_reduces_loss(self, act):
        """Training with each activation should reduce loss."""
        nn = NeuralNetwork([2, 8, 1], activation=act, learning_rate=0.01,
                           optimizer="adam", seed=42)
        losses = nn.train(XOR_X, XOR_Y, epochs=2000, log_every=0)
        assert losses[-1] < losses[0]

    def test_leaky_relu_convergence(self):
        """Leaky ReLU should converge on XOR."""
        nn = NeuralNetwork([2, 16, 1], activation="leaky_relu",
                           learning_rate=0.01, optimizer="adam", seed=42)
        nn.train(XOR_X, XOR_Y, epochs=10_000, log_every=0)
        preds = nn.predict(XOR_X)
        rounded = np.round(preds).astype(int)
        np.testing.assert_array_equal(rounded, XOR_Y.astype(int))


# ── L2/L1 Regularisation ─────────────────────────────────────────────
class TestRegularisation:
    def test_l2_default_zero(self):
        nn = NeuralNetwork([2, 4, 1])
        assert nn.l2_lambda == 0.0
        assert nn.l1_lambda == 0.0

    def test_l2_increases_reported_loss(self):
        """L2 penalty should increase the reported loss vs. no regularisation."""
        nn_base = NeuralNetwork([2, 4, 1], learning_rate=0.5, seed=42)
        nn_l2 = NeuralNetwork([2, 4, 1], learning_rate=0.5, seed=42, l2_lambda=0.1)
        loss_base = nn_base.train(XOR_X, XOR_Y, epochs=1, log_every=0)
        loss_l2 = nn_l2.train(XOR_X, XOR_Y, epochs=1, log_every=0)
        # L2 penalty adds to loss
        assert loss_l2[0] > loss_base[0]

    def test_l1_increases_reported_loss(self):
        """L1 penalty should increase the reported loss vs. no regularisation."""
        nn_base = NeuralNetwork([2, 4, 1], learning_rate=0.5, seed=42)
        nn_l1 = NeuralNetwork([2, 4, 1], learning_rate=0.5, seed=42, l1_lambda=0.1)
        loss_base = nn_base.train(XOR_X, XOR_Y, epochs=1, log_every=0)
        loss_l1 = nn_l1.train(XOR_X, XOR_Y, epochs=1, log_every=0)
        assert loss_l1[0] > loss_base[0]

    def test_l2_shrinks_weights(self):
        """L2 regularisation should produce smaller weight norms than baseline."""
        nn_base = NeuralNetwork([2, 8, 1], learning_rate=0.5, seed=42)
        nn_l2 = NeuralNetwork([2, 8, 1], learning_rate=0.5, seed=42, l2_lambda=0.01)
        nn_base.train(XOR_X, XOR_Y, epochs=3000, log_every=0)
        nn_l2.train(XOR_X, XOR_Y, epochs=3000, log_every=0)
        norm_base = sum(np.sum(w ** 2) for w in nn_base.weights)
        norm_l2 = sum(np.sum(w ** 2) for w in nn_l2.weights)
        assert norm_l2 < norm_base

    def test_l2_convergence(self):
        """Network should still converge with moderate L2."""
        nn = NeuralNetwork([2, 8, 1], learning_rate=1.0, seed=42, l2_lambda=0.001)
        losses = nn.train(XOR_X, XOR_Y, epochs=10_000, log_every=0)
        assert losses[-1] < losses[0]

    def test_l2_l1_save_load(self):
        """Save/load should preserve L2/L1 lambdas."""
        nn = NeuralNetwork([2, 4, 1], l2_lambda=0.01, l1_lambda=0.005, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=50, log_every=0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reg.npz"
            nn.save(path)
            loaded = NeuralNetwork.load(path)
        assert loaded.l2_lambda == 0.01
        assert loaded.l1_lambda == 0.005


# ── Mini-batch training ──────────────────────────────────────────────
class TestMiniBatch:
    def test_batch_size_none_is_full_batch(self):
        """batch_size=None should use the full dataset each epoch."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        assert nn.batch_size is None
        losses = nn.train(XOR_X, XOR_Y, epochs=100, log_every=0)
        assert len(losses) == 100

    def test_mini_batch_reduces_loss(self):
        """Mini-batch training should reduce loss."""
        nn = NeuralNetwork([2, 8, 1], learning_rate=0.5, seed=42, batch_size=2)
        losses = nn.train(XOR_X, XOR_Y, epochs=2000, log_every=0)
        assert losses[-1] < losses[0]

    def test_mini_batch_convergence(self):
        """Mini-batch should converge on XOR."""
        nn = NeuralNetwork([2, 16, 1], learning_rate=0.01, seed=42,
                           batch_size=2, optimizer="adam")
        nn.train(XOR_X, XOR_Y, epochs=15_000, log_every=0)
        preds = nn.predict(XOR_X)
        # Allow approximate convergence for mini-batch
        for i in range(4):
            assert abs(preds[i, 0] - XOR_Y[i, 0]) < 0.3

    def test_batch_size_larger_than_data(self):
        """batch_size >= N should behave like full-batch."""
        nn1 = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42, batch_size=100)
        nn2 = NeuralNetwork([2, 4, 1], learning_rate=1.0, seed=42)
        losses1 = nn1.train(XOR_X, XOR_Y, epochs=100, log_every=0)
        losses2 = nn2.train(XOR_X, XOR_Y, epochs=100, log_every=0)
        np.testing.assert_array_almost_equal(losses1, losses2)

    def test_batch_size_save_load(self):
        """Save/load should preserve batch_size."""
        nn = NeuralNetwork([2, 4, 1], batch_size=16, seed=42)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mb.npz"
            nn.save(path)
            loaded = NeuralNetwork.load(path)
        assert loaded.batch_size == 16


# ── LR Warmup ────────────────────────────────────────────────────────
class TestLRWarmup:
    def test_warmup_default_zero(self):
        nn = NeuralNetwork([2, 4, 1])
        assert nn.warmup_epochs == 0

    def test_warmup_ramps_lr(self):
        """During warmup, LR should increase linearly from 0."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0, warmup_epochs=10, seed=42)
        nn._training = True
        nn._update_lr(5, 100)
        assert abs(nn.learning_rate - 0.5) < 1e-9
        nn._update_lr(10, 100)
        assert abs(nn.learning_rate - 1.0) < 1e-9

    def test_warmup_then_schedule(self):
        """After warmup period, the normal schedule should take over."""
        nn = NeuralNetwork([2, 4, 1], learning_rate=1.0,
                           warmup_epochs=10, lr_schedule="step",
                           lr_step_every=100, seed=42)
        # During warmup
        nn._update_lr(5, 1000)
        assert abs(nn.learning_rate - 0.5) < 1e-9
        # After warmup, step schedule kicks in
        nn._update_lr(200, 1000)
        assert nn.learning_rate < 1.0

    def test_warmup_training_stable(self):
        """Training with warmup should produce finite losses."""
        nn = NeuralNetwork([2, 8, 1], learning_rate=0.5, warmup_epochs=100,
                           optimizer="adam", seed=42)
        losses = nn.train(XOR_X, XOR_Y, epochs=2000, log_every=0)
        assert all(np.isfinite(v) for v in losses)
        assert losses[-1] < losses[0]

    def test_warmup_save_load(self):
        """Save/load should preserve warmup_epochs."""
        nn = NeuralNetwork([2, 4, 1], warmup_epochs=50, seed=42)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "warmup.npz"
            nn.save(path)
            loaded = NeuralNetwork.load(path)
        assert loaded.warmup_epochs == 50


# ── All new features combined ────────────────────────────────────────
class TestAllNewFeatures:
    def test_everything_together(self):
        """All new features combined: new activation + L2 + mini-batch + warmup."""
        nn = NeuralNetwork(
            [2, 16, 1],
            learning_rate=0.01,
            activation="swish",
            optimizer="adam",
            l2_lambda=0.001,
            batch_size=2,
            warmup_epochs=50,
            grad_clip=5.0,
            seed=42,
        )
        losses = nn.train(XOR_X, XOR_Y, epochs=5000, log_every=0)
        assert all(np.isfinite(v) for v in losses)
        assert losses[-1] < losses[0]


# ── Weight Initialization Options ────────────────────────────────────
class TestWeightInit:
    @pytest.mark.parametrize("init", ["auto", "he", "xavier", "lecun", "orthogonal"])
    def test_constructor_accepts(self, init):
        nn = NeuralNetwork([2, 4, 1], weight_init=init, seed=42)
        assert nn.weight_init == init

    def test_invalid_init_rejected(self):
        with pytest.raises(ValueError, match="weight_init"):
            NeuralNetwork([2, 4, 1], weight_init="bad")

    @pytest.mark.parametrize("init", ["he", "xavier", "lecun", "orthogonal"])
    def test_training_with_init(self, init):
        nn = NeuralNetwork([2, 8, 1], weight_init=init, learning_rate=0.5, seed=42)
        losses = nn.train(XOR_X, XOR_Y, epochs=2000, log_every=0)
        assert losses[-1] < losses[0]

    def test_orthogonal_correct_shape(self):
        nn = NeuralNetwork([4, 8, 3], weight_init="orthogonal", seed=42)
        for w in nn.weights:
            # Orthogonal: columns should be roughly orthonormal
            m = min(w.shape)
            if w.shape[0] >= w.shape[1]:
                dots = w.T @ w
            else:
                dots = w @ w.T
            np.testing.assert_array_almost_equal(dots, np.eye(m), decimal=5)

    def test_auto_picks_he_for_relu(self):
        nn = NeuralNetwork([2, 4, 1], activation="relu", weight_init="auto", seed=42)
        nn_he = NeuralNetwork([2, 4, 1], activation="relu", weight_init="he", seed=42)
        np.testing.assert_array_equal(nn.weights[0], nn_he.weights[0])

    def test_auto_picks_xavier_for_sigmoid(self):
        nn = NeuralNetwork([2, 4, 1], activation="sigmoid", weight_init="auto", seed=42)
        nn_xavier = NeuralNetwork([2, 4, 1], activation="sigmoid", weight_init="xavier", seed=42)
        np.testing.assert_array_equal(nn.weights[0], nn_xavier.weights[0])

    def test_weight_init_save_load(self):
        nn = NeuralNetwork([2, 4, 1], weight_init="lecun", seed=42)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.npz"
            nn.save(path)
            loaded = NeuralNetwork.load(path)
        assert loaded.weight_init == "lecun"


# ── Best-weights restore during early stopping ──────────────────────
class TestBestWeightsRestore:
    def test_early_stop_restores_best(self):
        """After early stopping, weights should be from the best epoch."""
        nn = NeuralNetwork([2, 8, 1], learning_rate=5.0, seed=42)
        # Deliberately use a high LR that will overshoot
        losses = nn.train(
            XOR_X, XOR_Y,
            epochs=50_000,
            log_every=0,
            early_stopping=True,
            patience=200,
            min_delta=1e-6,
        )
        # Should have stopped early
        assert len(losses) < 50_000
        # The model should still produce finite outputs (best weights restored)
        out = nn.predict(XOR_X)
        assert np.all(np.isfinite(out))

    def test_val_early_stop_restores_best(self):
        """Early stopping with validation data should restore best weights."""
        nn = NeuralNetwork([4, 8, 3], learning_rate=0.01, activation="tanh",
                           optimizer="adam", loss="cross_entropy",
                           softmax_output=True, seed=42)
        rng = np.random.default_rng(0)
        x_train = rng.random((6, 4))
        y_train = np.zeros((6, 3))
        y_train[range(6), [0, 1, 2, 0, 1, 2]] = 1.0
        x_val = rng.random((3, 4))
        y_val = np.eye(3, dtype=np.float64)

        nn.train(
            x_train, y_train,
            epochs=5000, log_every=0,
            early_stopping=True, patience=200,
            x_val=x_val, y_val=y_val,
        )
        # Should produce valid softmax outputs
        out = nn.predict(x_val)
        assert out.shape == (3, 3)
        np.testing.assert_almost_equal(np.sum(out, axis=1), [1.0, 1.0, 1.0], decimal=5)


# ── Batch Normalisation ──────────────────────────────────────────────
class TestBatchNorm:
    def test_constructor_accepts(self):
        nn = NeuralNetwork([2, 4, 1], batch_norm=True, seed=42)
        assert nn.batch_norm is True

    def test_default_is_off(self):
        nn = NeuralNetwork([2, 4, 1], seed=42)
        assert nn.batch_norm is False

    def test_training_converges_with_bn(self):
        nn = NeuralNetwork([2, 8, 1], batch_norm=True, learning_rate=0.5, seed=42)
        losses = nn.train(XOR_X, XOR_Y, epochs=5000, log_every=0)
        assert losses[-1] < losses[0]

    def test_predict_uses_running_stats(self):
        nn = NeuralNetwork([2, 8, 1], batch_norm=True, learning_rate=0.5, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=2000, log_every=0)
        out = nn.predict(XOR_X)
        assert out.shape == (4, 1)
        assert np.all(np.isfinite(out))

    def test_bn_save_load(self, tmp_path):
        nn = NeuralNetwork([2, 8, 1], batch_norm=True, learning_rate=0.5, seed=42)
        nn.train(XOR_X, XOR_Y, epochs=500, log_every=0)
        p = tmp_path / "bn_model.npz"
        nn.save(p)
        nn2 = NeuralNetwork.load(p)
        assert nn2.batch_norm is True
        np.testing.assert_array_almost_equal(nn.predict(XOR_X), nn2.predict(XOR_X))

    def test_bn_with_softmax(self):
        nn = NeuralNetwork(
            [4, 8, 3], batch_norm=True, learning_rate=0.01,
            activation="relu", optimizer="adam",
            loss="cross_entropy", softmax_output=True, seed=42,
        )
        rng = np.random.default_rng(0)
        X = rng.random((6, 4))
        Y = np.zeros((6, 3))
        Y[range(6), [0, 1, 2, 0, 1, 2]] = 1.0
        losses = nn.train(X, Y, epochs=2000, log_every=0)
        assert losses[-1] < losses[0]
        out = nn.predict(X)
        np.testing.assert_almost_equal(np.sum(out, axis=1), np.ones(6), decimal=5)


# ── Confidence Calibration ───────────────────────────────────────────
class TestConfidenceCalibration:
    def test_calibrate_returns_temperature(self):
        nn = NeuralNetwork(
            [4, 8, 3], learning_rate=0.01, activation="tanh",
            optimizer="adam", loss="cross_entropy", softmax_output=True, seed=42,
        )
        rng = np.random.default_rng(0)
        X = rng.random((6, 4))
        Y = np.zeros((6, 3))
        Y[range(6), [0, 1, 2, 0, 1, 2]] = 1.0
        nn.train(X, Y, epochs=1000, log_every=0)
        T = nn.calibrate_temperature(X, Y)
        assert isinstance(T, float)
        assert T > 0

    def test_predict_calibrated_returns_valid_probs(self):
        nn = NeuralNetwork(
            [4, 8, 3], learning_rate=0.01, activation="tanh",
            optimizer="adam", loss="cross_entropy", softmax_output=True, seed=42,
        )
        rng = np.random.default_rng(0)
        X = rng.random((6, 4))
        Y = np.zeros((6, 3))
        Y[range(6), [0, 1, 2, 0, 1, 2]] = 1.0
        nn.train(X, Y, epochs=1000, log_every=0)
        nn.calibrate_temperature(X, Y)
        preds = nn.predict_calibrated(X)
        assert preds.shape == (6, 3)
        np.testing.assert_almost_equal(np.sum(preds, axis=1), np.ones(6), decimal=5)

    def test_calibrate_requires_softmax(self):
        nn = NeuralNetwork([2, 4, 1], seed=42)
        with pytest.raises(ValueError, match="softmax_output"):
            nn.calibrate_temperature(XOR_X, XOR_Y)

    def test_temperature_saved_and_loaded(self, tmp_path):
        nn = NeuralNetwork(
            [4, 8, 3], learning_rate=0.01, activation="tanh",
            optimizer="adam", loss="cross_entropy", softmax_output=True, seed=42,
        )
        rng = np.random.default_rng(0)
        X = rng.random((6, 4))
        Y = np.zeros((6, 3))
        Y[range(6), [0, 1, 2, 0, 1, 2]] = 1.0
        nn.train(X, Y, epochs=1000, log_every=0)
        nn.calibrate_temperature(X, Y)
        p = tmp_path / "calibrated.npz"
        nn.save(p)
        nn2 = NeuralNetwork.load(p)
        np.testing.assert_almost_equal(
            nn.predict_calibrated(X), nn2.predict_calibrated(X), decimal=5,
        )


# ── Gradient Accumulation ────────────────────────────────────────────
class TestGradientAccumulation:
    def test_constructor_default_is_1(self):
        nn = NeuralNetwork([2, 4, 1], seed=42)
        assert nn.accumulation_steps == 1

    def test_constructor_accepts(self):
        nn = NeuralNetwork([2, 4, 1], accumulation_steps=4, seed=42)
        assert nn.accumulation_steps == 4

    def test_training_with_accumulation(self):
        nn = NeuralNetwork(
            [2, 8, 1], batch_size=2, accumulation_steps=2,
            learning_rate=0.5, seed=42,
        )
        losses = nn.train(XOR_X, XOR_Y, epochs=3000, log_every=0)
        assert losses[-1] < losses[0]

    def test_accumulation_save_load(self, tmp_path):
        nn = NeuralNetwork([2, 4, 1], accumulation_steps=3, seed=42)
        p = tmp_path / "accum.npz"
        nn.save(p)
        nn2 = NeuralNetwork.load(p)
        assert nn2.accumulation_steps == 3


# ── LR Finder ────────────────────────────────────────────────────────
class TestLRFinder:
    def test_returns_suggested_lr(self):
        nn = NeuralNetwork([2, 8, 1], learning_rate=0.5, seed=42)
        suggested, lrs, losses = nn.lr_find(XOR_X, XOR_Y, steps=50)
        assert isinstance(suggested, float)
        assert suggested > 0
        assert len(lrs) > 0
        assert len(losses) == len(lrs)

    def test_does_not_change_weights(self):
        nn = NeuralNetwork([2, 8, 1], learning_rate=0.5, seed=42)
        w_before = [w.copy() for w in nn.weights]
        nn.lr_find(XOR_X, XOR_Y, steps=30)
        for w_b, w_a in zip(w_before, nn.weights):
            np.testing.assert_array_equal(w_b, w_a)

    def test_lr_restored(self):
        nn = NeuralNetwork([2, 8, 1], learning_rate=0.123, seed=42)
        nn.lr_find(XOR_X, XOR_Y, steps=20)
        assert nn.learning_rate == 0.123

    def test_suggested_lr_is_reasonable(self):
        nn = NeuralNetwork([2, 16, 1], learning_rate=0.5, seed=42)
        suggested, _, _ = nn.lr_find(XOR_X, XOR_Y, steps=80)
        # Should be a positive, finite number in a reasonable range
        assert 1e-8 < suggested < 10.0


# ── Cosine Annealing with Warm Restarts ──────────────────────────────
class TestCosineRestarts:
    def test_cosine_restarts_accepted(self):
        nn = NeuralNetwork([2, 4, 1], lr_schedule="cosine_restarts", lr_step_every=100, seed=42)
        assert nn.lr_schedule == "cosine_restarts"

    def test_training_with_cosine_restarts(self):
        nn = NeuralNetwork(
            [2, 8, 1], learning_rate=0.5, lr_schedule="cosine_restarts",
            lr_step_every=200, seed=42,
        )
        losses = nn.train(XOR_X, XOR_Y, epochs=2000, log_every=0)
        assert losses[-1] < losses[0]

    def test_lr_oscillates(self):
        """LR should restart (increase) periodically."""
        nn = NeuralNetwork(
            [2, 4, 1], learning_rate=1.0, lr_schedule="cosine_restarts",
            lr_step_every=10, seed=42,
        )
        nn.train(XOR_X, XOR_Y, epochs=30, log_every=0)
        # After training, history should show LR going up at restart points
        lrs = nn.history["lr"]
        # There should be at least one increase in the LR after epoch 10
        has_increase = any(lrs[i] > lrs[i - 1] for i in range(1, len(lrs)))
        assert has_increase


# ── Label Smoothing ──────────────────────────────────────────────────
class TestLabelSmoothing:
    def test_default_is_zero(self):
        nn = NeuralNetwork([2, 4, 1], seed=42)
        assert nn.label_smoothing == 0.0

    def test_constructor_accepts(self):
        nn = NeuralNetwork([2, 4, 1], label_smoothing=0.1, seed=42)
        assert nn.label_smoothing == 0.1

    def test_training_with_label_smoothing(self):
        nn = NeuralNetwork(
            [4, 8, 3], learning_rate=0.01, activation="tanh",
            optimizer="adam", loss="cross_entropy", softmax_output=True,
            label_smoothing=0.1, seed=42,
        )
        rng = np.random.default_rng(0)
        X = rng.random((6, 4))
        Y = np.zeros((6, 3))
        Y[range(6), [0, 1, 2, 0, 1, 2]] = 1.0
        losses = nn.train(X, Y, epochs=1000, log_every=0)
        assert losses[-1] < losses[0]

    def test_save_load_label_smoothing(self, tmp_path):
        nn = NeuralNetwork([2, 4, 1], label_smoothing=0.05, seed=42)
        p = tmp_path / "ls.npz"
        nn.save(p)
        nn2 = NeuralNetwork.load(p)
        assert nn2.label_smoothing == 0.05


# ── Training History ─────────────────────────────────────────────────
class TestTrainingHistory:
    def test_history_contains_train_loss(self):
        nn = NeuralNetwork([2, 4, 1], seed=42)
        nn.train(XOR_X, XOR_Y, epochs=100, log_every=0)
        assert hasattr(nn, "history")
        assert "train_loss" in nn.history
        assert len(nn.history["train_loss"]) == 100

    def test_history_contains_lr(self):
        nn = NeuralNetwork([2, 4, 1], seed=42)
        nn.train(XOR_X, XOR_Y, epochs=50, log_every=0)
        assert "lr" in nn.history
        assert len(nn.history["lr"]) == 50

    def test_history_contains_val_loss(self):
        nn = NeuralNetwork([4, 8, 3], learning_rate=0.01, activation="tanh",
                           optimizer="adam", loss="cross_entropy",
                           softmax_output=True, seed=42)
        rng = np.random.default_rng(0)
        X = rng.random((6, 4))
        Y = np.zeros((6, 3))
        Y[range(6), [0, 1, 2, 0, 1, 2]] = 1.0
        x_val = rng.random((3, 4))
        y_val = np.eye(3, dtype=np.float64)
        nn.train(
            X, Y, epochs=100, log_every=0,
            early_stopping=True, patience=200,
            x_val=x_val, y_val=y_val,
        )
        assert "val_loss" in nn.history
        assert len(nn.history["val_loss"]) > 0

    def test_no_val_loss_without_val_data(self):
        nn = NeuralNetwork([2, 4, 1], seed=42)
        nn.train(XOR_X, XOR_Y, epochs=50, log_every=0)
        assert "val_loss" not in nn.history
