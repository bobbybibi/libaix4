#!/usr/bin/env python3
"""
local_scheduler.py — Standalone local scheduler for libaix automation.

Replaces GitHub Actions cron jobs when running offline or self-hosted.
Reads configuration from data/cron_config.json.

Usage:
    python local_scheduler.py              # Run scheduler in foreground
    python local_scheduler.py --once       # Run all jobs once and exit
    python local_scheduler.py --status     # Show scheduler status
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("libaix-scheduler")

CRON_CONFIG_PATH = Path("data/cron_config.json")
SCHEDULER_STATE_PATH = Path("data/scheduler_state.json")

_scheduler_thread: threading.Thread | None = None
_stop_event = threading.Event()


# ── Config ────────────────────────────────────────────────────────────

def _load_cron_config() -> dict:
    if CRON_CONFIG_PATH.exists():
        try:
            return json.loads(CRON_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "jobs": {
            "auto_train": {"enabled": True, "runs_per_hour": 4},
            "wiki_crawler": {"enabled": True, "runs_per_hour": 4},
            "forum_crawler": {"enabled": True, "runs_per_hour": 4},
            "ml_growth": {"enabled": True, "runs_per_hour": 1},
            "digest": {"enabled": True, "runs_per_hour": 1},
            "topic_learner": {"enabled": True, "runs_per_hour": 2},
        }
    }


def _load_state() -> dict:
    if SCHEDULER_STATE_PATH.exists():
        try:
            return json.loads(SCHEDULER_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_runs": {}, "run_counts": {}, "errors": {}, "started_at": None}


def _save_state(state: dict) -> None:
    SCHEDULER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULER_STATE_PATH.write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )


# ── Job Runners ───────────────────────────────────────────────────────

def _run_auto_train() -> dict:
    """Run knowledge model training."""
    log.info("Running auto-train…")
    try:
        from train_knowledge import main as train_main
        train_main()
        return {"status": "success"}
    except Exception as e:
        log.error("Auto-train failed: %s", e)
        return {"status": "error", "error": str(e)}


def _run_wiki_crawler() -> dict:
    """Run Wikipedia knowledge crawler."""
    log.info("Running wiki crawler…")
    try:
        from crawler import run_all_crawlers
        result = run_all_crawlers()
        entries = result.get("total_new_entries", 0)
        log.info("Wiki crawler: %d new entries", entries)
        return {"status": "success", "entries": entries}
    except Exception as e:
        log.error("Wiki crawler failed: %s", e)
        return {"status": "error", "error": str(e)}


def _run_forum_crawler() -> dict:
    """Run forum crawler (StackExchange, Reddit, HN, DEV.to)."""
    log.info("Running forum crawler…")
    try:
        from forum_crawler import crawl_forums
        result = crawl_forums()
        entries = result.get("total_entries", 0)
        log.info("Forum crawler: %d new entries", entries)
        return {"status": "success", "entries": entries}
    except Exception as e:
        log.error("Forum crawler failed: %s", e)
        return {"status": "error", "error": str(e)}


def _run_ml_growth() -> dict:
    """Run ML self-growth cycle."""
    log.info("Running ML growth cycle…")
    try:
        from ml_engine import run_growth_cycle
        result = run_growth_cycle()
        log.info("ML growth: %s", result.get("status", "unknown"))
        return result
    except Exception as e:
        log.error("ML growth failed: %s", e)
        return {"status": "error", "error": str(e)}


def _run_digest() -> dict:
    """Run digestive mode — process existing data."""
    log.info("Running digest cycle…")
    try:
        from digest_engine import run_digest_cycle
        result = run_digest_cycle()
        log.info("Digest: %s", result.get("status", "unknown"))
        return result
    except Exception as e:
        log.error("Digest failed: %s", e)
        return {"status": "error", "error": str(e)}


def _run_topic_learner() -> dict:
    """Auto-learn from priority topics."""
    log.info("Running topic learner…")
    try:
        # Load learning topics
        topics_path = Path("data/learning_topics.json")
        if not topics_path.exists():
            return {"status": "no_topics"}
        config = json.loads(topics_path.read_text(encoding="utf-8"))
        topics = [t for t in config.get("topics", []) if t.get("enabled", True)]
        if not topics:
            return {"status": "no_enabled_topics"}

        # Sort by priority: high first, then medium, then low
        priority_order = {"high": 0, "medium": 1, "low": 2}
        topics.sort(key=lambda t: priority_order.get(t.get("priority", "medium"), 1))

        # Pick the top priority topic
        topic = topics[0]
        topic_name = topic["name"]
        keywords = topic.get("keywords", [])

        from crawler import crawl_single_topic
        from forum_crawler import crawl_single_forum_topic

        wiki_result = crawl_single_topic(topic_name, keywords, max_articles=5)
        forum_result = crawl_single_forum_topic(
            topic_name, keywords, max_per_source=5,
            sources=["stackexchange", "reddit"],
        )

        total = 0
        if wiki_result.get("status") == "success":
            total += wiki_result.get("entries", 0)
        if forum_result.get("status") == "success":
            total += forum_result.get("entries", 0)

        log.info("Topic learner: %d entries for '%s'", total, topic_name)
        return {"status": "success", "topic": topic_name, "entries": total}
    except Exception as e:
        log.error("Topic learner failed: %s", e)
        return {"status": "error", "error": str(e)}


JOB_RUNNERS = {
    "auto_train": _run_auto_train,
    "wiki_crawler": _run_wiki_crawler,
    "forum_crawler": _run_forum_crawler,
    "ml_growth": _run_ml_growth,
    "digest": _run_digest,
    "topic_learner": _run_topic_learner,
}


# ── Scheduler Logic ──────────────────────────────────────────────────

def _should_run(job_name: str, runs_per_hour: int, state: dict) -> bool:
    """Check if a job should run based on its configured frequency."""
    if runs_per_hour <= 0:
        return False
    interval_seconds = 3600 / runs_per_hour
    last_run_str = state.get("last_runs", {}).get(job_name)
    if not last_run_str:
        return True
    try:
        last_run = datetime.fromisoformat(last_run_str)
        elapsed = (datetime.now(timezone.utc) - last_run).total_seconds()
        return elapsed >= interval_seconds
    except Exception:
        return True


def run_once() -> dict:
    """Run all enabled jobs once."""
    config = _load_cron_config()
    state = _load_state()
    results = {}

    for job_name, runner in JOB_RUNNERS.items():
        job_cfg = config.get("jobs", {}).get(job_name, {})
        if not job_cfg.get("enabled", False):
            results[job_name] = {"status": "disabled"}
            continue

        log.info("Running job: %s", job_name)
        try:
            result = runner()
            results[job_name] = result
            state.setdefault("last_runs", {})[job_name] = datetime.now(
                timezone.utc
            ).isoformat()
            state.setdefault("run_counts", {})[job_name] = (
                state.get("run_counts", {}).get(job_name, 0) + 1
            )
        except Exception as e:
            results[job_name] = {"status": "error", "error": str(e)}
            state.setdefault("errors", {})[job_name] = str(e)

    _save_state(state)
    return results


def _scheduler_loop() -> None:
    """Main scheduler loop — runs in a thread."""
    state = _load_state()
    state["started_at"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)
    log.info("Scheduler started")

    while not _stop_event.is_set():
        config = _load_cron_config()
        state = _load_state()

        for job_name, runner in JOB_RUNNERS.items():
            if _stop_event.is_set():
                break
            job_cfg = config.get("jobs", {}).get(job_name, {})
            if not job_cfg.get("enabled", False):
                continue

            runs_per_hour = job_cfg.get("runs_per_hour", 1)
            if _should_run(job_name, runs_per_hour, state):
                try:
                    result = runner()
                    state.setdefault("last_runs", {})[job_name] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    state.setdefault("run_counts", {})[job_name] = (
                        state.get("run_counts", {}).get(job_name, 0) + 1
                    )
                    if result.get("status") == "error":
                        state.setdefault("errors", {})[job_name] = result.get(
                            "error", "unknown"
                        )
                except Exception as e:
                    log.error("Job %s failed: %s", job_name, e)
                    state.setdefault("errors", {})[job_name] = str(e)

                _save_state(state)

        # Sleep 60 seconds between checks
        _stop_event.wait(60)

    log.info("Scheduler stopped")


def start_scheduler() -> None:
    """Start scheduler in a background thread."""
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    _scheduler_thread.start()


def stop_scheduler() -> None:
    """Stop the background scheduler."""
    _stop_event.set()
    if _scheduler_thread:
        _scheduler_thread.join(timeout=5)


def get_scheduler_status() -> dict:
    """Return current scheduler status."""
    state = _load_state()
    running = _scheduler_thread is not None and _scheduler_thread.is_alive()
    config = _load_cron_config()

    jobs_status = {}
    for job_name in JOB_RUNNERS:
        job_cfg = config.get("jobs", {}).get(job_name, {})
        jobs_status[job_name] = {
            "enabled": job_cfg.get("enabled", False),
            "runs_per_hour": job_cfg.get("runs_per_hour", 0),
            "last_run": state.get("last_runs", {}).get(job_name),
            "total_runs": state.get("run_counts", {}).get(job_name, 0),
            "last_error": state.get("errors", {}).get(job_name),
        }

    return {
        "running": running,
        "mode": "local",
        "started_at": state.get("started_at"),
        "jobs": jobs_status,
    }


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="libaix local scheduler")
    parser.add_argument("--once", action="store_true", help="Run all jobs once and exit")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    args = parser.parse_args()

    if args.status:
        status = get_scheduler_status()
        print(json.dumps(status, indent=2, default=str))
        return

    if args.once:
        results = run_once()
        for name, result in results.items():
            print(f"  {name}: {result.get('status', 'unknown')}")
        return

    # Run scheduler in foreground
    print("Starting libaix local scheduler… Press Ctrl+C to stop.")
    print("Jobs will run based on data/cron_config.json settings.")
    print()

    _scheduler_loop_foreground()


def _scheduler_loop_foreground() -> None:
    """Run scheduler in foreground (blocking)."""
    try:
        start_scheduler()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler…")
        stop_scheduler()
        print("Done.")


if __name__ == "__main__":
    main()
