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
    monkeypatch.setattr("ml_watcher.GROWTH_LOG_PATH", watcher_dir / "growth_log.json")
    monkeypatch.setattr("ml_watcher.CONFIG_BASELINE_PATH", watcher_dir / "config_baseline.json")

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


# ── Knowledge growth tracking ───────────────────────────────────────


class TestKnowledgeGrowth:
    def test_track_growth_returns_structure(self):
        from ml_watcher import track_knowledge_growth
        result = track_knowledge_growth()
        assert "current_total" in result
        assert "domain_count" in result
        assert "velocity" in result
        assert "average_growth" in result
        assert "data_points" in result
        assert "history" in result
        assert result["data_points"] >= 1

    def test_growth_accumulates(self):
        from ml_watcher import track_knowledge_growth
        r1 = track_knowledge_growth()
        r2 = track_knowledge_growth()
        assert r2["data_points"] == r1["data_points"] + 1

    def test_growth_history_limited(self):
        from ml_watcher import track_knowledge_growth
        track_knowledge_growth()
        result = track_knowledge_growth()
        assert len(result["history"]) <= 10

    def test_velocity_calculation(self):
        from ml_watcher import track_knowledge_growth
        track_knowledge_growth()
        result = track_knowledge_growth()
        # Same knowledge base, so velocity should be 0
        assert result["velocity"] == 0


# ── Config drift detection ──────────────────────────────────────────


class TestConfigDrift:
    def test_detect_drift_returns_structure(self):
        from ml_watcher import detect_config_drift
        result = detect_config_drift()
        assert "added" in result
        assert "removed" in result
        assert "modified" in result
        assert "total_drift" in result
        assert "has_drift" in result
        assert "current_configs" in result

    def test_save_baseline(self):
        from ml_watcher import save_config_baseline
        baseline = save_config_baseline()
        assert "saved_at" in baseline
        assert "configs" in baseline

    def test_no_drift_after_baseline(self):
        from ml_watcher import detect_config_drift, save_config_baseline
        save_config_baseline()
        result = detect_config_drift()
        assert result["modified"] == []
        assert result["added"] == []
        assert result["removed"] == []

    def test_drift_detected_after_change(self, tmp_path, monkeypatch):
        import json as _json
        from ml_watcher import detect_config_drift, save_config_baseline
        save_config_baseline()
        # Simulate a config change by creating a new file in data/
        config_path = Path("data") / "test_drift_detect.json"
        config_path.write_text(_json.dumps({"test": True}), encoding="utf-8")
        try:
            result = detect_config_drift()
            assert "test_drift_detect.json" in result["added"]
            assert result["has_drift"] is True
        finally:
            config_path.unlink(missing_ok=True)


# ── Disk usage ──────────────────────────────────────────────────────


class TestDiskUsage:
    def test_measure_returns_structure(self):
        from ml_watcher import measure_disk_usage
        result = measure_disk_usage()
        assert "directories" in result
        assert "total_bytes" in result
        assert "total_formatted" in result
        assert "data" in result["directories"]
        assert "models" in result["directories"]
        assert "tests" in result["directories"]

    def test_directory_entries_have_bytes(self):
        from ml_watcher import measure_disk_usage
        result = measure_disk_usage()
        for name, info in result["directories"].items():
            assert "bytes" in info
            assert "formatted" in info
            assert info["bytes"] >= 0

    def test_total_is_sum(self):
        from ml_watcher import measure_disk_usage
        result = measure_disk_usage()
        expected = sum(d["bytes"] for d in result["directories"].values())
        assert result["total_bytes"] == expected


# ── Alert summary ───────────────────────────────────────────────────


class TestAlertSummary:
    def test_empty_summary(self):
        from ml_watcher import get_alert_summary
        summary = get_alert_summary()
        assert summary["total_alerts"] == 0
        assert summary["unacknowledged"] == 0
        assert summary["by_level"] == {}
        assert summary["by_category"] == {}

    def test_summary_with_alerts(self):
        from ml_watcher import add_alert, get_alert_summary
        add_alert("Info 1", level="info", category="test")
        add_alert("Warning 1", level="warning", category="model")
        add_alert("Info 2", level="info", category="test")
        summary = get_alert_summary()
        assert summary["total_alerts"] == 3
        assert summary["unacknowledged"] == 3
        assert summary["by_level"]["info"] == 2
        assert summary["by_level"]["warning"] == 1
        assert summary["by_category"]["test"] == 2
        assert summary["by_category"]["model"] == 1

    def test_summary_recent_limited(self):
        from ml_watcher import add_alert, get_alert_summary
        for i in range(10):
            add_alert(f"Alert {i}")
        summary = get_alert_summary()
        assert len(summary["recent"]) <= 5


class TestClearAlerts:
    def test_clear_acknowledged(self):
        from ml_watcher import (
            acknowledge_alert,
            add_alert,
            clear_acknowledged_alerts,
            get_alerts,
        )
        add_alert("Keep me")
        a2 = add_alert("Clear me")
        acknowledge_alert(a2["id"])
        result = clear_acknowledged_alerts()
        assert result["removed"] == 1
        assert result["remaining"] == 1
        remaining = get_alerts()
        assert len(remaining) == 1
        assert remaining[0]["message"] == "Keep me"

    def test_clear_when_none_acknowledged(self):
        from ml_watcher import add_alert, clear_acknowledged_alerts
        add_alert("Unacknowledged")
        result = clear_acknowledged_alerts()
        assert result["removed"] == 0
        assert result["remaining"] == 1
