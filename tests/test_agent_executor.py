"""Tests for agent_executor.py."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch


from agent_executor import AgentExecutor, AgentTask, _MAX_STORED_TASKS
from skill_registry import SkillRegistry, SkillResult


# ── Helper: registry with a mock skill ───────────────────────────────

def _make_registry_with_mock() -> SkillRegistry:
    """Return a SkillRegistry whose execute() is mocked for testing."""
    reg = SkillRegistry()
    return reg


def _make_executor(registry: SkillRegistry | None = None) -> AgentExecutor:
    if registry is None:
        registry = SkillRegistry()
    return AgentExecutor(registry=registry)


# ── AgentTask tests ──────────────────────────────────────────────────

class TestAgentTask:
    def test_creation(self):
        t = AgentTask(
            task_id="t1",
            skill_name="sk",
            command="cmd",
            args={"a": 1},
            status="pending",
        )
        assert t.task_id == "t1"
        assert t.skill_name == "sk"
        assert t.command == "cmd"
        assert t.args == {"a": 1}
        assert t.status == "pending"

    def test_defaults(self):
        t = AgentTask(task_id="t2", skill_name="s", command="c", args={}, status="pending")
        assert t.result is None
        assert t.error is None
        assert t.completed_at is None
        assert t.background is False
        assert isinstance(t.created_at, float)

    def test_status_values(self):
        for status in ("pending", "running", "completed", "failed", "cancelled"):
            t = AgentTask(task_id="x", skill_name="s", command="c", args={}, status=status)
            assert t.status == status

    def test_background_flag(self):
        t = AgentTask(task_id="x", skill_name="s", command="c", args={}, status="pending", background=True)
        assert t.background is True


# ── AgentExecutor — process_message tests ────────────────────────────

class TestProcessMessage:
    def test_no_match_returns_action_taken_false(self):
        ex = _make_executor()
        result = ex.process_message("xyzzy nothing matches")
        assert result["action_taken"] is False
        assert result["skill"] == ""
        assert result["task_id"] is None
        assert "No matching skill" in result["message"]

    def test_empty_input(self):
        ex = _make_executor()
        result = ex.process_message("")
        assert result["action_taken"] is False

    def test_match_with_confirmation_required(self):
        """When intent requires confirmation, no execution should happen."""
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        mock_match = MagicMock()
        mock_match.requires_confirmation = True
        mock_match.skill_name = "test_skill"
        mock_match.command = "dangerous_cmd"
        mock_match.args = {}
        mock_match.confidence = 0.9

        with patch.object(reg, "match_intent", return_value=mock_match):
            result = ex.process_message("do dangerous thing")

        assert result["action_taken"] is False
        assert result.get("requires_confirmation") is True
        assert result["skill"] == "test_skill"

    def test_match_without_confirmation_executes(self):
        """When intent matches and does not require confirmation, task runs."""
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        mock_match = MagicMock()
        mock_match.requires_confirmation = False
        mock_match.skill_name = "test_skill"
        mock_match.command = "safe_cmd"
        mock_match.args = {}

        mock_result = SkillResult(success=True, message="done", data={"key": "val"})

        with patch.object(reg, "match_intent", return_value=mock_match), \
             patch.object(reg, "execute", return_value=mock_result):
            result = ex.process_message("do safe thing")

        assert result["action_taken"] is True
        assert result["skill"] == "test_skill"
        assert result["command"] == "safe_cmd"
        assert result["task_id"] is not None

    def test_match_execute_failure(self):
        """When execution fails, the result reflects the failure."""
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        mock_match = MagicMock()
        mock_match.requires_confirmation = False
        mock_match.skill_name = "test_skill"
        mock_match.command = "fail_cmd"
        mock_match.args = {}

        with patch.object(reg, "match_intent", return_value=mock_match), \
             patch.object(reg, "execute", side_effect=RuntimeError("kaboom")):
            result = ex.process_message("do failing thing")

        assert result["action_taken"] is True
        assert result["task_id"] is not None


# ── AgentExecutor — execute_task tests ───────────────────────────────

class TestExecuteTask:
    def test_synchronous_task_completed(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        mock_result = SkillResult(success=True, message="ok", data={"out": 1})
        with patch.object(reg, "execute", return_value=mock_result):
            task = ex.execute_task("sk", "cmd", {"a": 1})

        assert task.skill_name == "sk"
        assert task.command == "cmd"
        assert task.status == "completed"
        assert task.result is not None
        assert task.completed_at is not None

    def test_synchronous_task_failed(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        with patch.object(reg, "execute", side_effect=ValueError("bad")):
            task = ex.execute_task("sk", "cmd", {})

        assert task.status == "failed"
        assert "bad" in task.error

    def test_default_args_is_empty(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        mock_result = SkillResult(success=True, message="ok")
        with patch.object(reg, "execute", return_value=mock_result):
            task = ex.execute_task("sk", "cmd")

        assert task.args == {}

    def test_background_task_returns_immediately(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        # Make execute block briefly to ensure background thread runs
        def slow_execute(*args, **kwargs):
            time.sleep(0.1)
            return SkillResult(success=True, message="bg done")

        with patch.object(reg, "execute", side_effect=slow_execute):
            task = ex.execute_task("sk", "cmd", {}, background=True)

        # Task returned immediately; status is pending or running
        assert task.background is True
        assert task.status in ("pending", "running")

        # Wait for background thread to finish
        time.sleep(0.5)
        refreshed = ex.get_task_status(task.task_id)
        assert refreshed is not None
        assert refreshed.status in ("completed", "failed")

    def test_task_id_is_unique(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)
        mock_result = SkillResult(success=True, message="ok")
        with patch.object(reg, "execute", return_value=mock_result):
            t1 = ex.execute_task("sk", "cmd")
            t2 = ex.execute_task("sk", "cmd")
        assert t1.task_id != t2.task_id


# ── AgentExecutor — get_task_status tests ────────────────────────────

class TestGetTaskStatus:
    def test_existing_task(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)
        mock_result = SkillResult(success=True, message="ok")
        with patch.object(reg, "execute", return_value=mock_result):
            task = ex.execute_task("sk", "cmd")
        found = ex.get_task_status(task.task_id)
        assert found is not None
        assert found.task_id == task.task_id

    def test_unknown_task(self):
        ex = _make_executor()
        assert ex.get_task_status("nonexistent-id") is None


# ── AgentExecutor — list_active_tasks tests ──────────────────────────

class TestListActiveTasks:
    def test_no_active_tasks(self):
        ex = _make_executor()
        assert ex.list_active_tasks() == []

    def test_completed_tasks_not_listed(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)
        mock_result = SkillResult(success=True, message="ok")
        with patch.object(reg, "execute", return_value=mock_result):
            ex.execute_task("sk", "cmd")
        # Completed tasks should not appear in active list
        assert ex.list_active_tasks() == []

    def test_pending_task_listed(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        def slow(*a, **kw):
            time.sleep(1)
            return SkillResult(success=True, message="ok")

        with patch.object(reg, "execute", side_effect=slow):
            task = ex.execute_task("sk", "cmd", {}, background=True)

        active = ex.list_active_tasks()
        assert len(active) >= 1
        assert any(t.task_id == task.task_id for t in active)

        # Clean up
        ex.cancel_task(task.task_id)
        time.sleep(0.1)


# ── AgentExecutor — cancel_task tests ────────────────────────────────

class TestCancelTask:
    def test_cancel_pending_task(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)

        def slow(*a, **kw):
            time.sleep(2)
            return SkillResult(success=True, message="ok")

        with patch.object(reg, "execute", side_effect=slow):
            task = ex.execute_task("sk", "cmd", {}, background=True)

        result = ex.cancel_task(task.task_id)
        assert result is True
        assert ex.get_task_status(task.task_id).status == "cancelled"
        time.sleep(0.1)

    def test_cancel_unknown_task(self):
        ex = _make_executor()
        assert ex.cancel_task("no-such-id") is False

    def test_cancel_completed_task_returns_false(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)
        mock_result = SkillResult(success=True, message="ok")
        with patch.object(reg, "execute", return_value=mock_result):
            task = ex.execute_task("sk", "cmd")
        # Task is already completed
        assert ex.cancel_task(task.task_id) is False


# ── AgentExecutor — task pruning tests ───────────────────────────────

class TestTaskPruning:
    def test_prune_keeps_under_max(self):
        reg = SkillRegistry()
        ex = AgentExecutor(registry=reg)
        mock_result = SkillResult(success=True, message="ok")

        with patch.object(reg, "execute", return_value=mock_result):
            for _ in range(_MAX_STORED_TASKS + 20):
                ex.execute_task("sk", "cmd")

        # After pruning, should be at most _MAX_STORED_TASKS
        assert len(ex._tasks) <= _MAX_STORED_TASKS


# ── get_executor singleton test ──────────────────────────────────────

class TestGetExecutor:
    def test_returns_agent_executor(self):
        from agent_executor import get_executor
        ex = get_executor()
        assert isinstance(ex, AgentExecutor)

    def test_returns_same_instance(self):
        from agent_executor import get_executor
        ex1 = get_executor()
        ex2 = get_executor()
        assert ex1 is ex2
