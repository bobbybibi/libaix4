"""Tests for libaix_brain.py — LIBAIXBrain orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_brain(tmp_path, monkeypatch):
    """Redirect brain state files to a temp directory."""
    brain_dir = tmp_path / "brain"
    brain_dir.mkdir()
    monkeypatch.setattr("libaix_brain.BRAIN_STATE_DIR", brain_dir)
    monkeypatch.setattr("libaix_brain.BRAIN_STATE_PATH", brain_dir / "brain_state.json")
    monkeypatch.setattr("libaix_brain.TASK_QUEUE_PATH", brain_dir / "task_queue.json")
    monkeypatch.setattr("libaix_brain.SESSION_LOG_PATH", brain_dir / "session_log.json")

    # Also isolate project_memory
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    monkeypatch.setattr("project_memory.MEMORY_DIR", mem_dir)
    monkeypatch.setattr("project_memory.MEMORY_PATH", mem_dir / "project_memory.json")
    monkeypatch.setattr("project_memory.CACHE_PATH", mem_dir / "response_cache.json")
    monkeypatch.setattr("project_memory.PERF_PATH", mem_dir / "performance_log.json")


# ── Brain state persistence ──────────────────────────────────────────


class TestBrainState:
    def test_default_state(self):
        from libaix_brain import load_brain_state
        state = load_brain_state()
        assert state["auto_mode"] is False
        assert state["health_score"] == 0
        assert state["cycle_count"] == 0

    def test_save_and_load(self):
        from libaix_brain import load_brain_state, save_brain_state
        state = load_brain_state()
        state["health_score"] = 42
        save_brain_state(state)
        reloaded = load_brain_state()
        assert reloaded["health_score"] == 42

    def test_corrupt_file_returns_default(self, tmp_path, monkeypatch):
        from libaix_brain import BRAIN_STATE_PATH, load_brain_state
        BRAIN_STATE_PATH.write_text("not json", encoding="utf-8")
        state = load_brain_state()
        assert "auto_mode" in state


# ── Task queue ───────────────────────────────────────────────────────


class TestTaskQueue:
    def test_add_task(self):
        from libaix_brain import add_task, load_task_queue
        task = add_task("Test task", "Do something", agent="tester", priority=3)
        assert task["status"] == "pending"
        assert task["priority"] == 3
        queue = load_task_queue()
        assert len(queue) == 1

    def test_complete_task(self):
        from libaix_brain import add_task, complete_task, load_task_queue
        task = add_task("Complete me", "Test")
        assert complete_task(task["id"])
        queue = load_task_queue()
        assert queue[0]["status"] == "completed"
        assert queue[0]["completed_at"] is not None

    def test_complete_nonexistent_returns_false(self):
        from libaix_brain import complete_task
        assert complete_task(999) is False

    def test_get_pending_tasks_filters_by_agent(self):
        from libaix_brain import add_task, get_pending_tasks
        add_task("Task A", "desc", agent="developer")
        add_task("Task B", "desc", agent="tester")
        add_task("Task C", "desc", agent="developer")
        dev_tasks = get_pending_tasks(agent="developer")
        assert len(dev_tasks) == 2
        all_tasks = get_pending_tasks()
        assert len(all_tasks) == 3

    def test_pending_sorted_by_priority(self):
        from libaix_brain import add_task, get_pending_tasks
        add_task("Low", "desc", priority=8)
        add_task("High", "desc", priority=1)
        add_task("Med", "desc", priority=5)
        tasks = get_pending_tasks()
        priorities = [t["priority"] for t in tasks]
        assert priorities == sorted(priorities)

    def test_priority_clamped(self):
        from libaix_brain import add_task
        t1 = add_task("Too low", "d", priority=-5)
        t2 = add_task("Too high", "d", priority=99)
        assert t1["priority"] == 1
        assert t2["priority"] == 10


# ── Session log ──────────────────────────────────────────────────────


class TestSessionLog:
    def test_log_and_retrieve(self):
        from libaix_brain import get_session_log, log_session
        log_session("Test session", ["action1", "action2"])
        log = get_session_log()
        assert len(log) == 1
        assert log[0]["summary"] == "Test session"
        assert len(log[0]["actions"]) == 2


# ── Project scanner ──────────────────────────────────────────────────


class TestProjectScanner:
    def test_scan_returns_manifest(self):
        from libaix_brain import scan_project
        manifest = scan_project()
        assert "modules" in manifest
        assert "routes" in manifest
        assert "tests" in manifest
        assert "stats" in manifest
        assert manifest["stats"]["total_modules"] > 0

    def test_scan_persists_state(self):
        from libaix_brain import load_brain_state, scan_project
        scan_project()
        state = load_brain_state()
        assert state["last_scan"] is not None
        assert state["manifest"]["stats"]["total_modules"] > 0


# ── Gap analysis ─────────────────────────────────────────────────────


class TestGapAnalysis:
    def test_analyse_gaps_returns_list(self):
        from libaix_brain import analyse_gaps, scan_project
        manifest = scan_project()
        gaps = analyse_gaps(manifest)
        assert isinstance(gaps, list)
        for gap in gaps:
            assert "title" in gap
            assert "severity" in gap
            assert "category" in gap

    def test_gaps_persisted(self):
        from libaix_brain import analyse_gaps, load_brain_state, scan_project
        manifest = scan_project()
        analyse_gaps(manifest)
        state = load_brain_state()
        assert "gaps" in state


# ── Health scoring ───────────────────────────────────────────────────


class TestHealthScore:
    def test_score_returns_components(self):
        from libaix_brain import calculate_health_score, scan_project
        manifest = scan_project()
        health = calculate_health_score(manifest)
        assert "total" in health
        assert "components" in health
        assert "grade" in health
        assert 0 <= health["total"] <= 100

    def test_grade_assignment(self):
        from libaix_brain import _score_to_grade
        assert _score_to_grade(95) == "A"
        assert _score_to_grade(85) == "B"
        assert _score_to_grade(75) == "C"
        assert _score_to_grade(65) == "D"
        assert _score_to_grade(50) == "F"


# ── Task generation ──────────────────────────────────────────────────


class TestTaskGeneration:
    def test_generate_tasks_from_gaps(self):
        from libaix_brain import generate_tasks_from_gaps
        gaps = [
            {
                "title": "Missing tests",
                "description": "Need tests for X",
                "severity": "medium",
                "category": "testing",
                "agent": "tester",
            }
        ]
        tasks = generate_tasks_from_gaps(gaps)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Fix: Missing tests"

    def test_no_duplicates(self):
        from libaix_brain import generate_tasks_from_gaps
        gaps = [
            {
                "title": "Gap A",
                "description": "Desc",
                "severity": "high",
                "category": "testing",
                "agent": "tester",
            }
        ]
        generate_tasks_from_gaps(gaps)
        # Run again — should not create duplicates
        tasks = generate_tasks_from_gaps(gaps)
        assert len(tasks) == 0


# ── Session briefing ─────────────────────────────────────────────────


class TestSessionBriefing:
    def test_briefing_structure(self):
        from libaix_brain import build_session_briefing
        briefing = build_session_briefing()
        assert "project" in briefing
        assert "stats" in briefing
        assert "known_agents" in briefing
        assert "health_score" in briefing


# ── Full scan cycle ──────────────────────────────────────────────────


class TestFullScanCycle:
    def test_full_cycle_returns_results(self):
        from libaix_brain import run_full_scan_cycle
        result = run_full_scan_cycle()
        assert "manifest_stats" in result
        assert "gaps" in result
        assert "health" in result
        assert "actions" in result
        assert result["cycle"] >= 1


# ── Auto mode ────────────────────────────────────────────────────────


class TestAutoMode:
    def test_toggle_auto_mode(self):
        from libaix_brain import get_auto_mode, set_auto_mode
        assert get_auto_mode() is False
        set_auto_mode(True)
        assert get_auto_mode() is True
        set_auto_mode(False)
        assert get_auto_mode() is False


# ── Status ───────────────────────────────────────────────────────────


class TestStatus:
    def test_get_status(self):
        from libaix_brain import get_status
        status = get_status()
        assert "health_score" in status
        assert "auto_mode" in status
        assert "known_agents" in status


# ── Helpers ──────────────────────────────────────────────────────────


class TestHelpers:
    def test_count_lines(self):
        from libaix_brain import _count_lines
        assert _count_lines(Path("app.py")) > 0

    def test_count_lines_missing_file(self):
        from libaix_brain import _count_lines
        assert _count_lines(Path("nonexistent.py")) == 0

    def test_extract_functions(self):
        from libaix_brain import _extract_functions
        funcs = _extract_functions(Path("app.py"))
        assert len(funcs) > 0

    def test_extract_routes(self):
        from libaix_brain import _extract_routes
        routes = _extract_routes(Path("app.py"))
        assert len(routes) > 0
        assert any(r["path"] == "/" for r in routes)

    def test_extract_test_names(self):
        from libaix_brain import _extract_test_names
        names = _extract_test_names(Path("tests/test_neural_network.py"))
        assert len(names) > 0
        assert all(n.startswith("test_") for n in names)
