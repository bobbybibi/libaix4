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

    def test_validate_module_name_known(self):
        from libaix_brain import _validate_module_name
        assert _validate_module_name("app.py") is True
        assert _validate_module_name("neural_network.py") is True

    def test_validate_module_name_unknown(self):
        from libaix_brain import _validate_module_name
        assert _validate_module_name("nonexistent.py") is False

    def test_validate_module_name_traversal(self):
        from libaix_brain import _validate_module_name
        assert _validate_module_name("../../etc/passwd") is False
        assert _validate_module_name("../app.py") is False
        assert _validate_module_name("app/../../etc/passwd") is False


# ── Dependency graph ─────────────────────────────────────────────────


class TestDependencyGraph:
    def test_build_graph_structure(self):
        from libaix_brain import build_dependency_graph
        result = build_dependency_graph()
        assert "graph" in result
        assert "reverse_graph" in result
        assert "total_modules" in result
        assert "total_edges" in result
        assert "circular_dependencies" in result
        assert "leaf_modules" in result
        assert result["total_modules"] > 0

    def test_graph_has_known_dependencies(self):
        from libaix_brain import build_dependency_graph
        result = build_dependency_graph()
        graph = result["graph"]
        # app.py imports many modules
        assert "app.py" in graph
        assert len(graph["app.py"]) > 0

    def test_reverse_graph_consistency(self):
        from libaix_brain import build_dependency_graph
        result = build_dependency_graph()
        graph = result["graph"]
        reverse = result.get("reverse_graph", {})
        # Every edge in reverse should have corresponding forward edge
        for dep, dependents in reverse.items():
            for d in dependents:
                assert dep in graph.get(d, []), f"{d} should depend on {dep}"

    def test_leaf_modules_have_no_deps(self):
        from libaix_brain import build_dependency_graph
        result = build_dependency_graph()
        graph = result["graph"]
        for leaf in result["leaf_modules"]:
            assert graph.get(leaf, []) == []

    def test_most_depended_on_sorted(self):
        from libaix_brain import build_dependency_graph
        result = build_dependency_graph()
        counts = [m["depended_by"] for m in result["most_depended_on"]]
        assert counts == sorted(counts, reverse=True)

    def test_graph_persists_summary(self):
        from libaix_brain import build_dependency_graph, load_brain_state
        build_dependency_graph()
        state = load_brain_state()
        assert "dependency_graph" in state
        assert "total_edges" in state["dependency_graph"]


class TestExtractImports:
    def test_extract_imports_from_app(self):
        from libaix_brain import _extract_imports
        imports = _extract_imports(Path("app.py"))
        assert len(imports) > 0
        # app.py imports knowledge_base, neural_network, etc.
        assert "knowledge_base.py" in imports

    def test_extract_imports_missing_file(self):
        from libaix_brain import _extract_imports
        assert _extract_imports(Path("nonexistent.py")) == []


# ── Module complexity ────────────────────────────────────────────────


class TestModuleComplexity:
    def test_score_returns_structure(self):
        from libaix_brain import score_module_complexity
        result = score_module_complexity()
        assert "modules" in result
        assert "total_complexity" in result
        assert "average_complexity" in result
        assert "most_complex" in result
        assert "simplest" in result
        assert len(result["modules"]) > 0

    def test_modules_sorted_by_complexity(self):
        from libaix_brain import score_module_complexity
        result = score_module_complexity()
        scores = [m["complexity_score"] for m in result["modules"]]
        assert scores == sorted(scores, reverse=True)

    def test_module_entry_has_expected_fields(self):
        from libaix_brain import score_module_complexity
        result = score_module_complexity()
        mod = result["modules"][0]
        assert "module" in mod
        assert "lines" in mod
        assert "functions" in mod
        assert "classes" in mod
        assert "branches" in mod
        assert "complexity_score" in mod

    def test_complexity_non_negative(self):
        from libaix_brain import score_module_complexity
        result = score_module_complexity()
        for mod in result["modules"]:
            assert mod["complexity_score"] >= 0

    def test_count_classes(self):
        from libaix_brain import _count_classes
        # neural_network.py has at least the NeuralNetwork class
        assert _count_classes(Path("neural_network.py")) >= 1

    def test_count_branches(self):
        from libaix_brain import _count_branches
        assert _count_branches(Path("app.py")) > 0
        assert _count_branches(Path("nonexistent.py")) == 0


# ── Code quality ─────────────────────────────────────────────────────


class TestCodeQuality:
    def test_measure_returns_structure(self):
        from libaix_brain import measure_code_quality
        result = measure_code_quality()
        assert "modules" in result
        assert "overall_docstring_coverage" in result
        assert "total_todos" in result
        assert "total_functions" in result
        assert "total_documented" in result

    def test_docstring_pct_in_range(self):
        from libaix_brain import measure_code_quality
        result = measure_code_quality()
        assert 0 <= result["overall_docstring_coverage"] <= 100

    def test_each_module_has_metrics(self):
        from libaix_brain import measure_code_quality
        result = measure_code_quality()
        for mod in result["modules"]:
            assert "module" in mod
            assert "lines" in mod
            assert "docstring_pct" in mod
            assert "todos" in mod
            assert 0 <= mod["docstring_pct"] <= 100

    def test_count_todos(self):
        from libaix_brain import _count_todos
        # Missing file returns 0
        assert _count_todos(Path("nonexistent.py")) == 0
        # Any file should return >= 0
        assert _count_todos(Path("app.py")) >= 0

    def test_has_docstrings(self):
        from libaix_brain import _has_docstrings
        documented, total = _has_docstrings(Path("libaix_brain.py"))
        assert total > 0
        assert documented >= 0
        assert documented <= total


# ── Knowledge gap recommendations ───────────────────────────────────


class TestKnowledgeGaps:
    def test_recommend_returns_structure(self):
        from libaix_brain import recommend_knowledge_gaps
        result = recommend_knowledge_gaps()
        assert "total_entries" in result
        assert "domain_count" in result
        assert "domain_distribution" in result
        assert "recommendations" in result
        assert "recommendation_count" in result

    def test_recommendations_are_actionable(self):
        from libaix_brain import recommend_knowledge_gaps
        result = recommend_knowledge_gaps()
        for rec in result["recommendations"]:
            assert "type" in rec
            assert "domain" in rec
            assert "reason" in rec
            assert "priority" in rec
            assert rec["type"] in ("expand_domain", "new_domain")
            assert rec["priority"] in ("high", "medium", "low")

    def test_domain_distribution_matches_total(self):
        from libaix_brain import recommend_knowledge_gaps
        result = recommend_knowledge_gaps()
        dist_total = sum(result["domain_distribution"].values())
        assert dist_total == result["total_entries"]


# ── Impact analysis ──────────────────────────────────────────────────


class TestImpactAnalysis:
    def test_analyse_app_impact(self):
        from libaix_brain import analyse_impact, scan_project
        scan_project()  # Need manifest for routes
        result = analyse_impact("neural_network.py")
        assert result["target"] == "neural_network.py"
        assert "direct_dependents" in result
        assert "transitive_dependents" in result
        assert "affected_tests" in result
        assert "risk_level" in result
        assert result["risk_level"] in ("low", "medium", "high")

    def test_impact_unknown_module(self):
        from libaix_brain import analyse_impact
        result = analyse_impact("nonexistent.py")
        assert result["target"] == "nonexistent.py"
        assert result["direct_dependents"] == []
        assert "error" in result

    def test_impact_blocks_path_traversal(self):
        from libaix_brain import analyse_impact
        result = analyse_impact("../../etc/passwd")
        assert "error" in result
        assert result["direct_dependents"] == []

    def test_impact_has_analysis_note(self):
        from libaix_brain import analyse_impact
        result = analyse_impact("knowledge_base.py")
        assert "analysis_note" in result
        assert "knowledge_base.py" in result["analysis_note"]


# ── Stale data detection ────────────────────────────────────────────


class TestStaleData:
    def test_detect_returns_structure(self):
        from libaix_brain import detect_stale_data
        result = detect_stale_data(max_age_days=30)
        assert "max_age_days" in result
        assert "stale_files" in result
        assert "stale_count" in result
        assert "total_stale_bytes" in result
        assert result["max_age_days"] == 30

    def test_all_stale_with_zero_days(self):
        from libaix_brain import detect_stale_data
        # 0 days means everything is stale (modified before "now")
        result = detect_stale_data(max_age_days=0)
        assert isinstance(result["stale_files"], list)
        # With max_age_days=0, entries older than 0 days (i.e., all) are stale
        for sf in result["stale_files"]:
            assert "path" in sf
            assert "age_days" in sf
            assert "category" in sf

    def test_stale_sorted_by_age(self):
        from libaix_brain import detect_stale_data
        result = detect_stale_data(max_age_days=0)
        ages = [sf["age_days"] for sf in result["stale_files"]]
        assert ages == sorted(ages, reverse=True)


# ── Module summary ──────────────────────────────────────────────────


class TestModuleSummary:
    def test_summarize_existing_module(self):
        from libaix_brain import summarize_module
        result = summarize_module("app.py")
        assert result["exists"] is True
        assert result["module"] == "app.py"
        assert result["lines"] > 0
        assert result["functions"] > 0
        assert len(result["routes"]) > 0
        assert result["complexity_score"] > 0

    def test_summarize_missing_module(self):
        from libaix_brain import summarize_module
        result = summarize_module("nonexistent.py")
        assert result["exists"] is False
        assert "error" in result

    def test_summarize_blocks_path_traversal(self):
        from libaix_brain import summarize_module
        result = summarize_module("../../../etc/passwd")
        assert result["exists"] is False
        assert "error" in result

    def test_summarize_has_test_info(self):
        from libaix_brain import summarize_module
        result = summarize_module("neural_network.py")
        assert result["exists"] is True
        assert result["test_file"] is not None
        assert result["test_count"] > 0

    def test_summarize_has_imports(self):
        from libaix_brain import summarize_module
        result = summarize_module("app.py")
        assert isinstance(result["local_imports"], list)
        assert len(result["local_imports"]) > 0
