"""
project_memory.py — Persistent context and memory system for libaix.

Provides a key-value store that persists across restarts so the system
can remember project analysis results, model performance history,
user interaction patterns, and configuration state without
re-analysing everything on each boot.

Features:
  • Namespaced memory (project, model, queries, config)
  • Automatic timestamping and TTL-based expiry
  • Response cache for the chat endpoint
  • Model performance tracking over time
  • Project structure fingerprinting (detects code changes)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

MEMORY_DIR = Path("data/memory")
MEMORY_PATH = MEMORY_DIR / "project_memory.json"
CACHE_PATH = MEMORY_DIR / "response_cache.json"
PERF_PATH = MEMORY_DIR / "performance_log.json"

MAX_CACHE_ENTRIES = 500
MAX_PERF_ENTRIES = 200


# ── Core memory store ────────────────────────────────────────────────


def _ensure_dir() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def load_memory() -> dict:
    """Load the full memory store from disk."""
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return _default_memory()
    return _default_memory()


def save_memory(mem: dict) -> None:
    """Persist the memory store to disk."""
    _ensure_dir()
    mem["_updated_at"] = datetime.now(timezone.utc).isoformat()
    MEMORY_PATH.write_text(
        json.dumps(mem, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _default_memory() -> dict:
    return {
        "_created_at": datetime.now(timezone.utc).isoformat(),
        "_updated_at": datetime.now(timezone.utc).isoformat(),
        "project": {},
        "model": {},
        "config": {},
        "insights": [],
    }


# ── Namespaced getters / setters ─────────────────────────────────────


def remember(namespace: str, key: str, value: object, ttl_hours: float | None = None) -> None:
    """Store a value under *namespace*.*key* with optional TTL."""
    mem = load_memory()
    if namespace not in mem:
        mem[namespace] = {}
    entry = {
        "value": value,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }
    if ttl_hours is not None:
        entry["expires_at"] = (
            datetime.now(timezone.utc).isoformat()  # placeholder — checked on recall
        )
        entry["ttl_hours"] = ttl_hours
    mem[namespace][key] = entry
    save_memory(mem)


def recall(namespace: str, key: str, default: object = None) -> object:
    """Retrieve a previously stored value. Returns *default* if missing or expired."""
    mem = load_memory()
    ns = mem.get(namespace, {})
    entry = ns.get(key)
    if entry is None:
        return default
    # Check TTL
    if "ttl_hours" in entry:
        stored = datetime.fromisoformat(entry["stored_at"])
        elapsed = (datetime.now(timezone.utc) - stored).total_seconds() / 3600
        if elapsed > entry["ttl_hours"]:
            # Expired — remove and return default
            del ns[key]
            save_memory(mem)
            return default
    return entry.get("value", default)


def forget(namespace: str, key: str) -> bool:
    """Remove a stored value. Returns True if it existed."""
    mem = load_memory()
    ns = mem.get(namespace, {})
    if key in ns:
        del ns[key]
        save_memory(mem)
        return True
    return False


def recall_all(namespace: str) -> dict:
    """Return all non-expired entries in a namespace as ``{key: value}``."""
    mem = load_memory()
    ns = mem.get(namespace, {})
    result = {}
    now = datetime.now(timezone.utc)
    for key, entry in list(ns.items()):
        if not isinstance(entry, dict) or "value" not in entry:
            continue
        if "ttl_hours" in entry:
            stored = datetime.fromisoformat(entry["stored_at"])
            if (now - stored).total_seconds() / 3600 > entry["ttl_hours"]:
                continue
        result[key] = entry["value"]
    return result


# ── Response cache ───────────────────────────────────────────────────


def load_response_cache() -> dict[str, dict]:
    """Load the question→answer response cache."""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_response_cache(cache: dict[str, dict]) -> None:
    """Persist the response cache, pruning if too large."""
    _ensure_dir()
    # Prune oldest entries if over limit
    if len(cache) > MAX_CACHE_ENTRIES:
        items = sorted(cache.items(), key=lambda kv: kv[1].get("cached_at", ""))
        cache = dict(items[-MAX_CACHE_ENTRIES:])
    CACHE_PATH.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def cache_response(question: str, answer: str, confidence: float, domain: str) -> None:
    """Cache a chat response for faster future lookups."""
    cache = load_response_cache()
    key = _normalise_question(question)
    cache[key] = {
        "answer": answer,
        "confidence": confidence,
        "domain": domain,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "hits": cache.get(key, {}).get("hits", 0) + 1,
    }
    save_response_cache(cache)


def lookup_cached_response(question: str) -> dict | None:
    """Look up a cached response. Returns None on miss."""
    cache = load_response_cache()
    key = _normalise_question(question)
    entry = cache.get(key)
    if entry is None:
        return None
    # Bump hit count
    entry["hits"] = entry.get("hits", 0) + 1
    cache[key] = entry
    save_response_cache(cache)
    return entry


def _normalise_question(q: str) -> str:
    """Normalise a question for cache key matching."""
    return q.lower().strip().rstrip("?").strip()


# ── Model performance tracking ───────────────────────────────────────


def log_model_performance(
    accuracy: float,
    confidence: float,
    entries: int,
    domains: int,
    event: str = "assessment",
) -> None:
    """Append a performance snapshot to the log."""
    _ensure_dir()
    log = _load_perf_log()
    log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "accuracy": round(accuracy, 4),
        "confidence": round(confidence, 4),
        "entries": entries,
        "domains": domains,
    })
    # Trim to max size
    log = log[-MAX_PERF_ENTRIES:]
    PERF_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def get_performance_history() -> list[dict]:
    """Return the full performance log."""
    return _load_perf_log()


def get_performance_trend(n: int = 10) -> dict:
    """Return summary stats for the last *n* performance snapshots."""
    log = _load_perf_log()
    recent = log[-n:] if log else []
    if not recent:
        return {"entries": 0, "improving": False, "trend": []}
    accuracies = [e["accuracy"] for e in recent]
    improving = len(accuracies) >= 2 and accuracies[-1] > accuracies[0]
    return {
        "entries": len(recent),
        "latest_accuracy": accuracies[-1],
        "best_accuracy": max(accuracies),
        "avg_accuracy": round(sum(accuracies) / len(accuracies), 4),
        "improving": improving,
        "trend": recent,
    }


def _load_perf_log() -> list[dict]:
    if PERF_PATH.exists():
        try:
            return json.loads(PERF_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


# ── Project fingerprinting ───────────────────────────────────────────

_TRACKED_EXTENSIONS = {".py", ".json", ".html", ".txt", ".md"}
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}


def compute_project_fingerprint(root: str | Path | None = None) -> str:
    """Hash key source files to detect code changes since last analysis."""
    root = Path(root) if root else Path(".")
    hasher = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if any(part in _IGNORED_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix in _TRACKED_EXTENSIONS:
            try:
                hasher.update(p.read_bytes())
            except OSError:
                continue
    return hasher.hexdigest()[:16]


def has_project_changed() -> bool:
    """Return True if the project files have changed since the last remembered fingerprint."""
    stored = recall("project", "fingerprint")
    current = compute_project_fingerprint()
    return stored != current


def update_project_fingerprint() -> str:
    """Compute and store the current project fingerprint."""
    fp = compute_project_fingerprint()
    remember("project", "fingerprint", fp)
    return fp


# ── Startup context builder ──────────────────────────────────────────


def build_startup_context() -> dict:
    """Build a context summary for the system on startup.

    Returns a dict with everything the system needs to know without
    re-analysing the entire project from scratch.
    """
    context: dict = {
        "project_changed": has_project_changed(),
        "performance": get_performance_trend(),
        "cache_size": len(load_response_cache()),
    }

    # Pull remembered facts
    for key in ("structure", "last_train_config", "last_accuracy", "knowledge_count"):
        val = recall("project", key)
        if val is not None:
            context[key] = val

    model_facts = recall_all("model")
    if model_facts:
        context["model"] = model_facts

    config_facts = recall_all("config")
    if config_facts:
        context["config"] = config_facts

    return context


def remember_training_result(
    accuracy: float,
    entries: int,
    domains: int,
    config: dict,
) -> None:
    """Convenience: remember key training results in one call."""
    remember("project", "last_accuracy", accuracy)
    remember("project", "knowledge_count", entries)
    remember("project", "last_train_config", config)
    log_model_performance(
        accuracy=accuracy,
        confidence=config.get("avg_confidence", 0),
        entries=entries,
        domains=domains,
        event="training",
    )


# ── Insights (auto-discoveries) ─────────────────────────────────────


def add_insight(message: str, category: str = "general") -> None:
    """Store an auto-discovered insight about the project."""
    mem = load_memory()
    insights = mem.get("insights", [])
    insights.append({
        "message": message,
        "category": category,
        "discovered_at": datetime.now(timezone.utc).isoformat(),
    })
    # Keep last 50 insights
    mem["insights"] = insights[-50:]
    save_memory(mem)


def get_insights(category: str | None = None) -> list[dict]:
    """Return stored insights, optionally filtered by category."""
    mem = load_memory()
    insights = mem.get("insights", [])
    if category:
        return [i for i in insights if i.get("category") == category]
    return insights
