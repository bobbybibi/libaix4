"""
agent_executor.py — Task executor for the libaix agent framework.

Bridges the skill registry with user-facing chat by:
  • Detecting intent from user input via the SkillRegistry
  • Executing matched skill commands (foreground or background)
  • Tracking task lifecycle (pending → running → completed/failed/cancelled)
  • Managing background tasks in daemon threads
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from skill_registry import (
    IntentMatch,
    SkillRegistry,
    SkillResult,
    get_registry,
)

# ── Constants ─────────────────────────────────────────────────────────

_MAX_STORED_TASKS: int = 100

# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class AgentTask:
    """Represents a task being executed by the agent."""

    task_id: str
    skill_name: str
    command: str
    args: dict[str, Any]
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    background: bool = False


# ── AgentExecutor ─────────────────────────────────────────────────────


class AgentExecutor:
    """Orchestrates skill execution on behalf of the user."""

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry
        self._tasks: dict[str, AgentTask] = {}
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────

    def process_message(self, user_input: str) -> dict[str, Any]:
        """Main entry point.  Detect intent and execute the matched skill.

        Returns a response dict with keys:
            action_taken, skill, command, result, message, task_id
        """
        match: IntentMatch | None = self._registry.match_intent(user_input)

        if match is None:
            return {
                "action_taken": False,
                "skill": "",
                "command": "",
                "result": None,
                "message": "No matching skill found",
                "task_id": None,
            }

        # If the match requires confirmation, surface that to the caller
        # without executing anything yet.
        if getattr(match, "requires_confirmation", False):
            return {
                "action_taken": False,
                "skill": match.skill_name,
                "command": match.command,
                "result": None,
                "message": "This action requires confirmation before execution.",
                "task_id": None,
                "requires_confirmation": True,
                "intent": {
                    "skill_name": match.skill_name,
                    "command": match.command,
                    "args": match.args if hasattr(match, "args") else {},
                    "confidence": match.confidence if hasattr(match, "confidence") else 1.0,
                },
            }

        args = match.args if hasattr(match, "args") else {}
        task = self.execute_task(
            skill_name=match.skill_name,
            command=match.command,
            args=args,
        )

        return {
            "action_taken": True,
            "skill": task.skill_name,
            "command": task.command,
            "result": task.result,
            "message": task.error if task.status == "failed" else "Task completed successfully.",
            "task_id": task.task_id,
        }

    def execute_task(
        self,
        skill_name: str,
        command: str,
        args: dict[str, Any] | None = None,
        background: bool = False,
    ) -> AgentTask:
        """Execute a specific task, optionally in the background."""
        if args is None:
            args = {}

        task = AgentTask(
            task_id=str(uuid.uuid4()),
            skill_name=skill_name,
            command=command,
            args=args,
            status="pending",
            background=background,
        )

        with self._lock:
            self._tasks[task.task_id] = task
            self._prune_tasks()

        if background:
            thread = threading.Thread(
                target=self._run_in_background,
                args=(task,),
                daemon=True,
            )
            thread.start()
            return task

        # Foreground: execute synchronously
        self._execute(task)
        return task

    def get_task_status(self, task_id: str) -> AgentTask | None:
        """Return the task for *task_id*, or ``None`` if unknown."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_active_tasks(self) -> list[AgentTask]:
        """Return all tasks whose status is ``'running'`` or ``'pending'``."""
        with self._lock:
            return [
                t for t in self._tasks.values()
                if t.status in ("pending", "running")
            ]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running or pending task.

        Returns ``True`` if the task was found and marked cancelled.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status not in ("pending", "running"):
                return False
            task.status = "cancelled"
            task.completed_at = time.time()
            return True

    # ── internal helpers ──────────────────────────────────────────────

    def _run_in_background(self, task: AgentTask) -> None:
        """Execute *task* inside a daemon thread."""
        self._execute(task)

    def _execute(self, task: AgentTask) -> None:
        """Run the skill command and update the task in-place."""
        task.status = "running"
        try:
            skill_result: SkillResult = self._registry.execute(
                task.skill_name,
                task.command,
                **task.args,
            )
            # If cancelled mid-flight, don't overwrite status
            if task.status == "cancelled":
                return
            task.status = "completed"
            task.result = skill_result.to_dict() if hasattr(skill_result, "to_dict") else {"data": skill_result.data if hasattr(skill_result, "data") else str(skill_result)}
            task.completed_at = time.time()
        except Exception as exc:  # noqa: BLE001
            if task.status == "cancelled":
                return
            task.status = "failed"
            task.error = str(exc)
            task.completed_at = time.time()

    def _prune_tasks(self) -> None:
        """Drop oldest completed tasks when the store exceeds the cap.

        Must be called while ``self._lock`` is held.
        """
        if len(self._tasks) <= _MAX_STORED_TASKS:
            return

        completed = sorted(
            (t for t in self._tasks.values() if t.status in ("completed", "failed", "cancelled")),
            key=lambda t: t.completed_at or t.created_at,
        )
        to_remove = len(self._tasks) - _MAX_STORED_TASKS
        for task in completed[:to_remove]:
            self._tasks.pop(task.task_id, None)


# ── Singleton accessor ────────────────────────────────────────────────

_executor: AgentExecutor | None = None
_executor_lock = threading.Lock()


def get_executor() -> AgentExecutor:
    """Return the global :class:`AgentExecutor` singleton."""
    global _executor  # noqa: PLW0603
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = AgentExecutor(registry=get_registry())
    return _executor
