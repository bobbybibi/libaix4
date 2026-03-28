"""Tests for the project memory system (project_memory.py)."""

from unittest.mock import patch

import pytest

from project_memory import (
    _normalise_question,
    add_insight,
    build_startup_context,
    cache_response,
    compute_project_fingerprint,
    forget,
    get_insights,
    get_performance_history,
    get_performance_trend,
    load_memory,
    log_model_performance,
    lookup_cached_response,
    recall,
    recall_all,
    remember,
    remember_training_result,
)


@pytest.fixture(autouse=True)
def _use_temp_dirs(tmp_path):
    """Redirect all memory paths to a temp directory for test isolation."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    with (
        patch("project_memory.MEMORY_DIR", mem_dir),
        patch("project_memory.MEMORY_PATH", mem_dir / "project_memory.json"),
        patch("project_memory.CACHE_PATH", mem_dir / "response_cache.json"),
        patch("project_memory.PERF_PATH", mem_dir / "performance_log.json"),
    ):
        yield


class TestRememberRecall:
    def test_remember_and_recall(self):
        remember("test_ns", "key1", "value1")
        assert recall("test_ns", "key1") == "value1"

    def test_recall_missing_returns_default(self):
        assert recall("test_ns", "nonexistent") is None
        assert recall("test_ns", "nonexistent", "fallback") == "fallback"

    def test_forget(self):
        remember("test_ns", "key1", "hello")
        assert forget("test_ns", "key1") is True
        assert recall("test_ns", "key1") is None

    def test_forget_missing(self):
        assert forget("test_ns", "nope") is False

    def test_recall_all(self):
        remember("ns1", "a", 1)
        remember("ns1", "b", 2)
        all_vals = recall_all("ns1")
        assert all_vals == {"a": 1, "b": 2}

    def test_remember_overwrites(self):
        remember("ns", "x", 10)
        remember("ns", "x", 20)
        assert recall("ns", "x") == 20

    def test_complex_values(self):
        data = {"items": [1, 2, 3], "nested": {"key": "val"}}
        remember("ns", "complex", data)
        assert recall("ns", "complex") == data

    def test_ttl_expired(self):
        remember("ns", "temp", "value", ttl_hours=0.0001)
        # Immediate recall should work since TTL hasn't elapsed yet
        val = recall("ns", "temp")
        assert val == "value"

    def test_memory_persists_across_loads(self):
        remember("persist", "test", "data")
        # Force reload
        mem = load_memory()
        ns = mem.get("persist", {})
        assert "test" in ns


class TestResponseCache:
    def test_cache_and_lookup(self):
        cache_response("What is TCP?", "TCP is a protocol.", 0.95, "networking")
        result = lookup_cached_response("What is TCP?")
        assert result is not None
        assert result["answer"] == "TCP is a protocol."
        assert result["confidence"] == 0.95
        assert result["domain"] == "networking"

    def test_cache_normalisation(self):
        cache_response("What is DNS?", "DNS resolves names.", 0.9, "internet")
        # Should match with different casing and trailing ?
        result = lookup_cached_response("what is dns?")
        assert result is not None
        assert result["answer"] == "DNS resolves names."

    def test_cache_miss(self):
        result = lookup_cached_response("completely unknown question")
        assert result is None

    def test_hit_counter(self):
        cache_response("Test Q?", "Test A.", 0.8, "general")
        lookup_cached_response("Test Q?")
        lookup_cached_response("Test Q?")
        result = lookup_cached_response("Test Q?")
        assert result["hits"] >= 3

    def test_normalise_question(self):
        assert _normalise_question("What is TCP?") == "what is tcp"
        assert _normalise_question("  DNS  ") == "dns"
        assert _normalise_question("HELLO?") == "hello"


class TestPerformanceTracking:
    def test_log_and_retrieve(self):
        log_model_performance(0.95, 0.8, 100, 5)
        log_model_performance(0.97, 0.85, 120, 6)
        history = get_performance_history()
        assert len(history) == 2
        assert history[-1]["accuracy"] == 0.97

    def test_trend_detection(self):
        log_model_performance(0.80, 0.7, 80, 4)
        log_model_performance(0.85, 0.75, 90, 5)
        log_model_performance(0.90, 0.8, 100, 5)
        trend = get_performance_trend()
        assert trend["improving"] is True
        assert trend["latest_accuracy"] == 0.90
        assert trend["best_accuracy"] == 0.90

    def test_empty_trend(self):
        trend = get_performance_trend()
        assert trend["entries"] == 0
        assert trend["improving"] is False


class TestProjectFingerprint:
    def test_fingerprint_is_deterministic(self):
        fp1 = compute_project_fingerprint()
        fp2 = compute_project_fingerprint()
        assert fp1 == fp2

    def test_fingerprint_is_string(self):
        fp = compute_project_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 16


class TestInsights:
    def test_add_and_get_insights(self):
        add_insight("Model accuracy improved to 95%", "model")
        add_insight("New domain added: IoT", "knowledge")
        insights = get_insights()
        assert len(insights) == 2

    def test_filter_by_category(self):
        add_insight("Fact A", "model")
        add_insight("Fact B", "knowledge")
        model_insights = get_insights("model")
        assert len(model_insights) == 1
        assert model_insights[0]["message"] == "Fact A"


class TestRememberTrainingResult:
    def test_remembers_all_fields(self):
        remember_training_result(
            accuracy=0.95,
            entries=100,
            domains=5,
            config={"activation": "tanh", "optimizer": "adam", "avg_confidence": 0.85},
        )
        assert recall("project", "last_accuracy") == 0.95
        assert recall("project", "knowledge_count") == 100
        cfg = recall("project", "last_train_config")
        assert cfg["activation"] == "tanh"

    def test_logs_performance(self):
        remember_training_result(
            accuracy=0.92, entries=80, domains=4,
            config={"avg_confidence": 0.78},
        )
        history = get_performance_history()
        assert len(history) >= 1
        assert history[-1]["event"] == "training"


class TestStartupContext:
    def test_builds_context(self):
        ctx = build_startup_context()
        assert "project_changed" in ctx
        assert "performance" in ctx
        assert "cache_size" in ctx

    def test_includes_remembered_facts(self):
        remember("project", "structure", {"datasets": ["xor"]})
        ctx = build_startup_context()
        assert ctx.get("structure") == {"datasets": ["xor"]}
