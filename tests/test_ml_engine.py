"""Tests for ml_engine.py — ML self-growth, stabilization, and optimization."""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

import ml_engine


@pytest.fixture(autouse=True)
def _isolate_ml_engine(tmp_path, monkeypatch):
    """Redirect all ml_engine paths to tmp_path for test isolation."""
    monkeypatch.setattr(ml_engine, "ENGINE_CONFIG_PATH", tmp_path / "ml_engine_config.json")
    monkeypatch.setattr(ml_engine, "ENGINE_HISTORY_PATH", tmp_path / "ml_engine_history.json")
    monkeypatch.setattr(ml_engine, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(ml_engine, "BACKUP_DIR", tmp_path / "models" / "backups")
    (tmp_path / "models").mkdir(exist_ok=True)


class TestConfig:
    def test_default_config_created(self):
        cfg = ml_engine.load_engine_config()
        assert isinstance(cfg, dict)
        assert "auto_optimize" in cfg
        assert "min_accuracy_threshold" in cfg

    def test_save_and_load_config(self):
        cfg = ml_engine.load_engine_config()
        cfg["custom_key"] = "test_value"
        ml_engine.save_engine_config(cfg)
        loaded = ml_engine.load_engine_config()
        assert loaded["custom_key"] == "test_value"

    def test_default_optimization_configs(self):
        cfg = ml_engine.load_engine_config()
        assert "optimization_configs" in cfg
        assert len(cfg["optimization_configs"]) > 0
        for c in cfg["optimization_configs"]:
            assert "activation" in c
            assert "optimizer" in c
            assert "lr" in c


class TestHistory:
    def test_empty_history_initially(self):
        history = ml_engine.load_history()
        assert history == []

    def test_record_event(self):
        ml_engine._record_event("test_event", {"key": "value"})
        history = ml_engine.load_history()
        assert len(history) == 1
        assert history[0]["type"] == "test_event"
        assert history[0]["key"] == "value"
        assert "timestamp" in history[0]

    def test_history_capped_at_100(self):
        for i in range(110):
            ml_engine._record_event("event", {"i": i})
        history = ml_engine.load_history()
        assert len(history) <= 100


class TestAssessModel:
    def test_no_model_returns_status(self):
        result = ml_engine.assess_model()
        assert result["status"] == "no_model"

    def test_assess_with_model(self, tmp_path, monkeypatch):
        """Create a minimal trained model and assess it."""
        from neural_network import NeuralNetwork
        from vectorizer import BagOfWords

        # Train a tiny model
        questions = ["What is TCP?", "What is a firewall?"]
        answers = ["TCP is a protocol.", "A firewall filters traffic."]
        bow = BagOfWords()
        X = bow.fit_transform(questions)
        n_classes = 2
        labels = np.eye(n_classes, dtype=np.float64)
        nn = NeuralNetwork(
            [bow.vocab_size, 16, n_classes],
            learning_rate=0.01, activation="tanh",
            optimizer="adam", loss="cross_entropy",
            softmax_output=True, seed=42,
        )
        nn.train(X, labels, epochs=100, log_every=0)

        model_dir = tmp_path / "models"
        nn.save(model_dir / "knowledge.npz")
        bow.save(model_dir / "vectorizer.json")
        answer_map = {str(i): answers[i] for i in range(n_classes)}
        (model_dir / "answer_map.json").write_text(json.dumps(answer_map))

        # Patch KNOWLEDGE to match our tiny dataset
        mini_knowledge = [
            ("What is TCP?", "TCP is a protocol.", "networking"),
            ("What is a firewall?", "A firewall filters traffic.", "security"),
        ]
        monkeypatch.setattr(ml_engine, "MODEL_DIR", model_dir)
        with patch("ml_engine.KNOWLEDGE", mini_knowledge, create=True):
            # Need to patch the import inside assess_model
            with patch.dict("sys.modules", {}):
                pass
            # assess_model imports KNOWLEDGE from knowledge_base
            import knowledge_base
            orig = knowledge_base.KNOWLEDGE
            monkeypatch.setattr(knowledge_base, "KNOWLEDGE", mini_knowledge)
            result = ml_engine.assess_model()
            monkeypatch.setattr(knowledge_base, "KNOWLEDGE", orig)

        assert result["status"] == "ok"
        assert "overall_accuracy" in result
        assert "domains" in result


class TestBackupRestore:
    def test_backup_no_model(self):
        result = ml_engine._backup_current_model()
        assert result is None

    def test_backup_creates_dir(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "models"
        model_dir.mkdir(exist_ok=True)
        backup_dir = model_dir / "backups"
        monkeypatch.setattr(ml_engine, "MODEL_DIR", model_dir)
        monkeypatch.setattr(ml_engine, "BACKUP_DIR", backup_dir)

        # Create a fake model file
        (model_dir / "knowledge.npz").write_bytes(b"fake")
        (model_dir / "vectorizer.json").write_text("{}")
        (model_dir / "answer_map.json").write_text("{}")

        result = ml_engine._backup_current_model()
        assert result is not None
        assert result.exists()
        assert (result / "knowledge.npz").exists()

    def test_restore_no_backup(self):
        assert ml_engine._restore_best_backup() is False

    def test_backup_and_restore(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "models"
        model_dir.mkdir(exist_ok=True)
        backup_dir = model_dir / "backups"
        monkeypatch.setattr(ml_engine, "MODEL_DIR", model_dir)
        monkeypatch.setattr(ml_engine, "BACKUP_DIR", backup_dir)

        # Create model files with known content
        (model_dir / "knowledge.npz").write_bytes(b"original_model")
        (model_dir / "vectorizer.json").write_text('{"vocab": {}}')
        (model_dir / "answer_map.json").write_text('{"0": "test"}')

        # Backup
        ml_engine._backup_current_model()

        # Overwrite model
        (model_dir / "knowledge.npz").write_bytes(b"new_model")

        # Restore
        assert ml_engine._restore_best_backup() is True
        assert (model_dir / "knowledge.npz").read_bytes() == b"original_model"


class TestEngineStats:
    def test_get_engine_stats(self):
        stats = ml_engine.get_engine_stats()
        assert isinstance(stats, dict)
        assert "config" in stats
        assert "history_count" in stats
        assert "recent_events" in stats
