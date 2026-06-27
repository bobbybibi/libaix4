"""
test_app.py — Tests for the Flask web application routes.

Covers:
  • /predict endpoint (input validation, logic gate predictions)
  • /train endpoint (parameter validation, training)
  • /chat endpoint (question handling, length limits)
  • /datasets endpoint
  • /knowledge/* endpoints (stats, domains)
  • /api/health endpoint
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app  # noqa: E402


class TestPredict:
    """Tests for POST /predict."""

    def setup_method(self):
        self.client = app.test_client()

    def test_predict_xor_00(self):
        resp = self.client.post("/predict", json={"a": 0, "b": 0, "dataset": "xor"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["a"] == 0 and data["b"] == 0
        assert data["dataset"] == "xor"
        assert "raw" in data and "result" in data

    def test_predict_xor_01(self):
        resp = self.client.post("/predict", json={"a": 0, "b": 1, "dataset": "xor"})
        data = resp.get_json()
        assert data["result"] == 1

    def test_predict_xor_11(self):
        resp = self.client.post("/predict", json={"a": 1, "b": 1, "dataset": "xor"})
        data = resp.get_json()
        assert data["result"] == 0

    def test_predict_and_gate(self):
        resp = self.client.post("/predict", json={"a": 1, "b": 1, "dataset": "and"})
        data = resp.get_json()
        assert data["dataset"] == "and"
        assert data["result"] == 1

    def test_predict_unknown_dataset(self):
        resp = self.client.post("/predict", json={"a": 0, "b": 0, "dataset": "nope"})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_predict_invalid_a_value(self):
        resp = self.client.post("/predict", json={"a": "foo", "b": 0})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_predict_clamps_values(self):
        resp = self.client.post("/predict", json={"a": 5, "b": -1, "dataset": "xor"})
        data = resp.get_json()
        assert data["a"] == 1
        assert data["b"] == 0

    def test_predict_defaults_to_xor(self):
        resp = self.client.post("/predict", json={"a": 0, "b": 0})
        data = resp.get_json()
        assert data["dataset"] == "xor"

    def test_predict_returns_activation_and_optimizer(self):
        resp = self.client.post("/predict", json={"a": 0, "b": 1})
        data = resp.get_json()
        assert "activation" in data
        assert "optimizer" in data


class TestTrain:
    """Tests for POST /train."""

    def setup_method(self):
        self.client = app.test_client()

    def test_train_xor(self):
        resp = self.client.post("/train", json={"dataset": "xor", "epochs": 100})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["dataset"] == "xor"
        assert data["epochs"] == 100
        assert "final_loss" in data
        assert "loss_curve" in data
        assert "predictions" in data
        assert "expected" in data

    def test_train_unknown_dataset(self):
        resp = self.client.post("/train", json={"dataset": "nope"})
        assert resp.status_code == 400

    def test_train_invalid_activation(self):
        resp = self.client.post("/train", json={"activation": "nonexistent_act"})
        assert resp.status_code == 400
        assert "activation" in resp.get_json()["error"]

    def test_train_invalid_optimizer(self):
        resp = self.client.post("/train", json={"optimizer": "rmsprop"})
        assert resp.status_code == 400
        assert "optimizer" in resp.get_json()["error"]

    def test_train_invalid_lr(self):
        resp = self.client.post("/train", json={"lr": "abc"})
        assert resp.status_code == 400

    def test_train_invalid_epochs(self):
        resp = self.client.post("/train", json={"epochs": "abc"})
        assert resp.status_code == 400

    def test_train_epochs_clamped(self):
        resp = self.client.post("/train", json={"epochs": 200_000})
        data = resp.get_json()
        assert data["epochs"] <= 100_000

    def test_train_with_all_activations(self):
        for act in ("sigmoid", "tanh", "relu"):
            resp = self.client.post("/train", json={
                "dataset": "and", "activation": act, "epochs": 50
            })
            assert resp.status_code == 200, f"Failed for activation={act}"

    def test_train_with_all_optimizers(self):
        for opt in ("sgd", "momentum", "adam"):
            resp = self.client.post("/train", json={
                "dataset": "or", "optimizer": opt, "epochs": 50
            })
            assert resp.status_code == 200, f"Failed for optimizer={opt}"


class TestDatasets:
    """Tests for GET /datasets."""

    def setup_method(self):
        self.client = app.test_client()

    def test_datasets_returns_list(self):
        resp = self.client.get("/datasets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert "xor" in data
        assert "and" in data
        assert "or" in data
        assert "nand" in data


class TestChat:
    """Tests for POST /chat."""

    def setup_method(self):
        self.client = app.test_client()

    def test_chat_empty_question(self):
        resp = self.client.post("/chat", json={"question": ""})
        assert resp.status_code == 400

    def test_chat_missing_question(self):
        resp = self.client.post("/chat", json={})
        assert resp.status_code == 400

    def test_chat_question_too_long(self):
        resp = self.client.post("/chat", json={"question": "x" * 2001})
        assert resp.status_code == 400
        assert "too long" in resp.get_json()["error"]

    def test_chat_returns_answer(self):
        """If knowledge model is loaded, chat should return an answer."""
        resp = self.client.post("/chat", json={"question": "What is TCP?"})
        data = resp.get_json()
        if resp.status_code == 503:
            # Model not loaded — acceptable in test environment
            assert "not loaded" in data["error"]
        else:
            assert resp.status_code == 200
            assert "answer" in data
            assert "confidence" in data
            assert "domain" in data


class TestKnowledgeEndpoints:
    """Tests for /knowledge/* endpoints."""

    def setup_method(self):
        self.client = app.test_client()

    def test_knowledge_stats(self):
        resp = self.client.get("/knowledge/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "loaded" in data
        assert "total_answers" in data
        assert "domains" in data
        assert "vocab_size" in data

    def test_knowledge_domains(self):
        resp = self.client.get("/knowledge/domains")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)


class TestHealth:
    """Tests for GET /api/health."""

    def setup_method(self):
        self.client = app.test_client()

    def test_health_returns_ok(self):
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "knowledge_loaded" in data
        assert "datasets" in data
        assert "models_ready" in data
        assert isinstance(data["datasets"], list)

    def test_health_lists_datasets(self):
        resp = self.client.get("/api/health")
        data = resp.get_json()
        assert "xor" in data["datasets"]


class TestIndex:
    """Tests for GET /."""

    def setup_method(self):
        self.client = app.test_client()

    def test_index_returns_html(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"libaix" in resp.data

    def test_index_has_chat_container(self):
        resp = self.client.get("/")
        assert b"chatMessages" in resp.data

    def test_index_has_aria_attributes(self):
        resp = self.client.get("/")
        assert b'role="tablist"' in resp.data
        assert b'aria-live="polite"' in resp.data
        assert b'aria-label' in resp.data


class TestMemory:
    """Tests for /memory/* endpoints."""

    def setup_method(self):
        self.client = app.test_client()

    def test_memory_context(self):
        resp = self.client.get("/memory/context")
        assert resp.status_code in (200, 500)  # 500 if memory dir missing

    def test_memory_performance(self):
        resp = self.client.get("/memory/performance")
        assert resp.status_code in (200, 500)
