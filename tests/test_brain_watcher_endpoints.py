"""Tests for extended brain and watcher API endpoints."""

from __future__ import annotations

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Create a test client with isolated brain/watcher state."""
    # Isolate brain state
    brain_dir = tmp_path / "brain"
    brain_dir.mkdir()
    monkeypatch.setattr("libaix_brain.BRAIN_STATE_DIR", brain_dir)
    monkeypatch.setattr("libaix_brain.BRAIN_STATE_PATH", brain_dir / "brain_state.json")
    monkeypatch.setattr("libaix_brain.TASK_QUEUE_PATH", brain_dir / "task_queue.json")
    monkeypatch.setattr("libaix_brain.SESSION_LOG_PATH", brain_dir / "session_log.json")

    # Isolate watcher state
    watcher_dir = tmp_path / "watcher"
    watcher_dir.mkdir()
    monkeypatch.setattr("ml_watcher.WATCHER_DIR", watcher_dir)
    monkeypatch.setattr("ml_watcher.SNAPSHOT_PATH", watcher_dir / "file_snapshot.json")
    monkeypatch.setattr("ml_watcher.CHANGE_LOG_PATH", watcher_dir / "change_log.json")
    monkeypatch.setattr("ml_watcher.KNOWLEDGE_INDEX_PATH", watcher_dir / "knowledge_index.json")
    monkeypatch.setattr("ml_watcher.ALERT_PATH", watcher_dir / "alerts.json")
    monkeypatch.setattr("ml_watcher.GROWTH_LOG_PATH", watcher_dir / "growth_log.json")
    monkeypatch.setattr("ml_watcher.CONFIG_BASELINE_PATH", watcher_dir / "config_baseline.json")

    # Isolate project memory
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    monkeypatch.setattr("project_memory.MEMORY_DIR", mem_dir)
    monkeypatch.setattr("project_memory.MEMORY_PATH", mem_dir / "project_memory.json")
    monkeypatch.setattr("project_memory.CACHE_PATH", mem_dir / "response_cache.json")
    monkeypatch.setattr("project_memory.PERF_PATH", mem_dir / "performance_log.json")

    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Brain endpoint tests ─────────────────────────────────────────────


class TestBrainGaps:
    def test_get_gaps(self, client):
        resp = client.get("/brain/gaps")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "gaps" in data
        assert isinstance(data["gaps"], list)

    def test_gaps_have_structure(self, client):
        resp = client.get("/brain/gaps")
        data = resp.get_json()
        for gap in data["gaps"]:
            assert "title" in gap
            assert "severity" in gap


class TestBrainTasks:
    def test_get_tasks_empty(self, client):
        resp = client.get("/brain/tasks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)

    def test_get_tasks_with_agent_filter(self, client):
        resp = client.get("/brain/tasks?agent=developer")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data["tasks"], list)


class TestBrainHealth:
    def test_get_health_score(self, client):
        resp = client.get("/brain/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total" in data
        assert "components" in data
        assert "grade" in data
        assert 0 <= data["total"] <= 100


class TestBrainDependencies:
    def test_get_dependencies(self, client):
        resp = client.get("/brain/dependencies")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "graph" in data
        assert "total_modules" in data
        assert "total_edges" in data


class TestBrainComplexity:
    def test_get_complexity(self, client):
        resp = client.get("/brain/complexity")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "modules" in data
        assert "total_complexity" in data
        assert len(data["modules"]) > 0


class TestBrainQuality:
    def test_get_quality(self, client):
        resp = client.get("/brain/quality")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "modules" in data
        assert "overall_docstring_coverage" in data
        assert "total_functions" in data


class TestBrainKnowledgeGaps:
    def test_get_knowledge_gaps(self, client):
        resp = client.get("/brain/knowledge-gaps")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_entries" in data
        assert "recommendations" in data


class TestBrainImpact:
    def test_impact_known_module(self, client):
        # Ensure manifest exists
        client.post("/brain/scan")
        resp = client.get("/brain/impact/neural_network.py")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["target"] == "neural_network.py"
        assert "risk_level" in data

    def test_impact_unknown_module(self, client):
        resp = client.get("/brain/impact/nonexistent.py")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["direct_dependents"] == []


class TestBrainStale:
    def test_get_stale_default(self, client):
        resp = client.get("/brain/stale")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "stale_files" in data
        assert data["max_age_days"] == 30

    def test_get_stale_custom_days(self, client):
        resp = client.get("/brain/stale?days=7")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["max_age_days"] == 7


class TestBrainModuleSummary:
    def test_summarize_known_module(self, client):
        resp = client.get("/brain/module/app.py")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["exists"] is True
        assert data["module"] == "app.py"
        assert data["lines"] > 0

    def test_summarize_unknown_module(self, client):
        resp = client.get("/brain/module/nonexistent.py")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["exists"] is False


# ── Watcher endpoint tests ───────────────────────────────────────────


class TestWatcherGrowth:
    def test_get_growth(self, client):
        resp = client.get("/watcher/growth")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current_total" in data
        assert "velocity" in data
        assert "data_points" in data


class TestWatcherConfigDrift:
    def test_get_config_drift(self, client):
        resp = client.get("/watcher/config-drift")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "added" in data
        assert "removed" in data
        assert "modified" in data
        assert "has_drift" in data


class TestWatcherDisk:
    def test_get_disk_usage(self, client):
        resp = client.get("/watcher/disk")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "directories" in data
        assert "total_bytes" in data
        assert "total_formatted" in data


class TestWatcherAlerts:
    def test_get_alerts_empty(self, client):
        resp = client.get("/watcher/alerts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_alerts"] == 0

    def test_clear_alerts(self, client):
        resp = client.post("/watcher/alerts/clear")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "removed" in data
        assert "remaining" in data


class TestWatcherHealth:
    def test_get_health(self, client):
        resp = client.get("/watcher/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "checked_at" in data
        assert "models" in data
