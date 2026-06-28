"""job_queue.py — Redis/RQ job queue helpers for libaix background tasks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from redis import Redis
from rq import Queue

_QUEUE_NAME = os.environ.get("LIBAIX_QUEUE", "libaix")
_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def get_redis() -> Redis:
    return Redis.from_url(_REDIS_URL)


def get_queue() -> Queue:
    return Queue(_QUEUE_NAME, connection=get_redis())


def enqueue_research_topic(topic: str, urls: list[str] | None = None):
    return get_queue().enqueue(task_research_topic, topic, urls or [])


def enqueue_rebuild_retriever():
    return get_queue().enqueue(task_rebuild_retriever)


def enqueue_train_knowledge():
    return get_queue().enqueue(task_train_knowledge)


def task_research_topic(topic: str, urls: list[str] | None = None) -> dict:
    import app as app_module

    result = app_module._execute_research(topic, urls=urls or None)
    return {"status": "ok", "topic": topic, "result": result}


def task_rebuild_retriever() -> dict:
    import app as app_module

    ok = bool(app_module.rebuild_retriever())
    return {"status": "ok" if ok else "failed", "rebuilt": ok}


def task_train_knowledge() -> dict:
    root = Path(__file__).resolve().parent
    proc = subprocess.run(
        [sys.executable, str(root / "train_knowledge.py")],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }
