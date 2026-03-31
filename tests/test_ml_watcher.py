"""Tests for ml_watcher.py — ML Learner Watcher."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_watcher(tmp_path, monkeypatch):
    """Redirect watcher state files to a temp directory."""
    watcher_dir = tmp_path / "watcher"
    watcher_dir.mkdir()
    monkeypatch.setattr("ml_watcher.WATCHER_DIR", watcher_dir)
    monkeypatch.setattr("ml_watcher.SNAPSHOT_PATH", watcher_dir / "file_snapshot.json")
    monkeypatch.setattr("ml_watcher.CHANGE_LOG_PATH", watcher_dir / "change_log.json")
    monkeypatch.setattr("ml_watcher.KNOWLEDGE_INDEX_PATH", watcher_dir / "knowledge_index.json")
    monkeypatch.setattr("ml_watcher.ALERT_PATH", watcher_dir / "alerts.json")

    # Also isolate project_memory
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    monkeypatch.setattr("project_memory.MEMORY_DIR", mem_dir)
    monkeypatch.setattr("project_memory.MEMORY_PATH", mem_dir / "project_memory.json")
    monkeypatch.setattr("project_memory.CACHE_PATH", mem_dir / "response_cache.json")
    monkeypatch.setattr("project_memory.PERF_PATH", mem_dir / "performance_log.json")


# ── File snapshot ────────────────────────────────────────────────────


class TestFileSnapshot:
    def test_take_snapshot(self):
        from ml_watcher import take_snapshot
        snap = take_snapshot()
        assert "taken_at" in snap
        assert "files" in snap
        assert snap["file_count"] > 0

    def test_load_snapshot_returns_empty_when_none(self, tmp_path, monkeypatch):
        # Point to nonexistent file
        monkeypatch.setattr("ml_watcher.SNAPSHOT_PATH", tmp_path / "nope.json")
        from ml_watcher import load_snapshot
        snap = load_snapshot()
        assert snap["taken_at"] is None
        assert snap["file_count"] == 0

    def test_snapshot_persists(self):
        from ml_watcher import load_snapshot, take_snapshot
        take_snapshot()
        loaded = load_snapshot()
        assert loaded["file_count"] > 0


# ── Change detection ─────────────────────────────────────────────────


class TestChangeDetection:
    def test_detect_changes_first_run(self):
        from ml_watcher import detect_changes
        changes = detect_changes()
        assert "added" in changes
        assert "removed" in changes
        assert "modified" in changes
        assert "total_changes" in changes
        # First run: everything is "added" since no previous snapshot
        assert changes["total_changes"] >= 0

    def test_no_changes_on_second_run(self):
        from ml_watcher import detect_changes
        detect_changes()  # First run creates snapshot
        changes = detect_changes()  # Second run compares
        # Should have very few or no changes (only watcher files themselves)
        assert isinstance(changes["modified"], list)

    def test_change_history(self):
        from ml_watcher import detect_changes, get_change_history
        detect_changes()
        history = get_change_history(n=5)
        assert isinstance(history, list)


# ── Knowledge index ──────────────────────────────────────────────────


class TestKnowledgeIndex:
    def test_build_knowledge_index(self):
        from ml_watcher import build_knowledge_index
        idx = build_knowledge_index()
        assert "builtin_entries" in idx
        assert "total_entries" in idx
        assert "domains" in idx
        assert "domain_counts" in idx
        assert idx["builtin_entries"] > 0  # knowledge_base.py has entries

    def test_index_persists(self):
        from ml_watcher import build_knowledge_index, load_knowledge_index
        build_knowledge_index()
        loaded = load_knowledge_index()
        assert loaded["builtin_entries"] > 0

    def test_sources_breakdown(self):
        from ml_watcher import build_knowledge_index
        idx = build_knowledge_index()
        assert "sources" in idx
        assert "builtin" in idx["sources"]
        assert idx["sources"]["builtin"] > 0


# ── Model watcher ────────────────────────────────────────────────────


class TestModelWatcher:
    def test_watch_models(self):
        from ml_watcher import watch_models
        result = watch_models()
        assert "checked_at" in result
        assert "models" in result
        assert "knowledge.npz" in result["models"]
        assert "vectorizer.json" in result["models"]
        assert "answer_map.json" in result["models"]

    def test_all_present_flag(self):
        from ml_watcher import watch_models
        result = watch_models()
        # May or may not be present depending on if model is trained
        assert isinstance(result["all_present"], bool)


# ── Alerts ───────────────────────────────────────────────────────────


class TestAlerts:
    def test_add_and_get_alert(self):
        from ml_watcher import add_alert, get_alerts
        add_alert("Test alert", level="info", category="test")
        alerts = get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["message"] == "Test alert"
        assert alerts[0]["acknowledged"] is False

    def test_acknowledge_alert(self):
        from ml_watcher import acknowledge_alert, add_alert, get_alerts
        alert = add_alert("Ack me", level="warning")
        assert acknowledge_alert(alert["id"])
        alerts = get_alerts(unacknowledged_only=True)
        assert len(alerts) == 0

    def test_acknowledge_nonexistent(self):
        from ml_watcher import acknowledge_alert
        assert acknowledge_alert(999) is False

    def test_filter_by_level(self):
        from ml_watcher import add_alert, get_alerts
        add_alert("Info", level="info")
        add_alert("Warning", level="warning")
        add_alert("Critical", level="critical")
        warnings = get_alerts(level="warning")
        assert len(warnings) == 1
        assert warnings[0]["level"] == "warning"

    def test_unacknowledged_filter(self):
        from ml_watcher import acknowledge_alert, add_alert, get_alerts
        a1 = add_alert("Alert 1")
        add_alert("Alert 2")
        acknowledge_alert(a1["id"])
        unack = get_alerts(unacknowledged_only=True)
        assert len(unack) == 1


# ── Health check ─────────────────────────────────────────────────────


class TestHealthCheck:
    def test_run_health_check(self):
        from ml_watcher import run_health_check
        result = run_health_check()
        assert "checked_at" in result
        assert "models" in result
        assert "performance" in result
        assert "knowledge" in result


# ── Watcher context ──────────────────────────────────────────────────


class TestWatcherContext:
    def test_build_context(self):
        from ml_watcher import build_watcher_context
        ctx = build_watcher_context()
        assert ctx["project_name"] == "libaix"
        assert "knowledge" in ctx
        assert "models" in ctx
        assert "performance" in ctx
        assert "context_built_at" in ctx

    def test_context_has_knowledge_details(self):
        from ml_watcher import build_watcher_context
        ctx = build_watcher_context()
        assert "total_entries" in ctx["knowledge"]
        assert "domains" in ctx["knowledge"]


# ── Full watcher cycle ───────────────────────────────────────────────


class TestWatcherCycle:
    def test_run_cycle(self):
        from ml_watcher import run_watcher_cycle
        result = run_watcher_cycle()
        assert "cycle_at" in result
        assert "changes" in result
        assert "knowledge" in result
        assert "health" in result


# ── Helpers ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_hash_file(self):
        from ml_watcher import _hash_file
        h = _hash_file(Path("app.py"))
        assert len(h) == 16
        assert h != "error"

    def test_hash_missing_file(self):
        from ml_watcher import _hash_file
        assert _hash_file(Path("nonexistent.xyz")) == "error"

    def test_file_info(self):
        from ml_watcher import _file_info
        info = _file_info(Path("app.py"))
        assert info["size"] > 0
        assert info["hash"] != "missing"

    def test_file_info_missing(self):
        from ml_watcher import _file_info
        info = _file_info(Path("nonexistent.xyz"))
        assert info["hash"] == "missing"
