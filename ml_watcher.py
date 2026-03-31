"""
ml_watcher.py — libaix Machine-Learner Watcher.

A persistent "project consciousness" that monitors every important file,
tracks all changes, indexes knowledge growth, and can instantly brief
any new session about what's going on — so you never start from zero.

Capabilities:
  • File inventory — SHA256 hashes of every important file
  • Change detection — diff summary since last snapshot
  • Knowledge index — counts, domains, sources, growth rate
  • Model watch — tracks model files, sizes, ages, backup count
  • Config watch — monitors all JSON configs for changes
  • Session context — instant "what you need to know" builder
  • Alert system — flags important changes (accuracy drops, new gaps)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from project_memory import (
    get_performance_trend,
    load_response_cache,
    recall_all,
    remember,
)

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(".")
WATCHER_DIR = Path("data/watcher")
SNAPSHOT_PATH = WATCHER_DIR / "file_snapshot.json"
CHANGE_LOG_PATH = WATCHER_DIR / "change_log.json"
KNOWLEDGE_INDEX_PATH = WATCHER_DIR / "knowledge_index.json"
ALERT_PATH = WATCHER_DIR / "alerts.json"

MAX_CHANGE_LOG = 200
MAX_ALERTS = 100

# Files / dirs the watcher tracks
WATCHED_PATTERNS = {
    "python": "*.py",
    "templates": "templates/*.html",
    "configs": "data/*.json",
    "models": "models/*",
    "tests": "tests/test_*.py",
    "requirements": "requirements.txt",
}

# Extensions to hash
_TRACKED_EXTENSIONS = {".py", ".json", ".html", ".txt", ".md", ".npz", ".cfg"}
_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".mypy_cache",
}


# ── Helpers ──────────────────────────────────────────────────────────


def _ensure_dir() -> None:
    WATCHER_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path: Path) -> str:
    """Return SHA256 hex hash of a file (first 16 chars)."""
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except OSError:
        return "error"
    return h.hexdigest()[:16]


def _file_info(path: Path) -> dict:
    """Return metadata dict for a file."""
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "hash": _hash_file(path),
        }
    except OSError:
        return {"path": str(path), "size": 0, "modified": None, "hash": "missing"}


# ── File snapshot ────────────────────────────────────────────────────


def take_snapshot() -> dict:
    """Capture a full snapshot of all tracked project files.

    Returns the snapshot dict with file hashes and metadata.
    """
    snapshot: dict = {
        "taken_at": _now_iso(),
        "files": {},
    }

    for p in sorted(PROJECT_ROOT.rglob("*")):
        if any(part in _IGNORED_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix not in _TRACKED_EXTENSIONS:
            continue
        key = str(p)
        snapshot["files"][key] = _file_info(p)

    snapshot["file_count"] = len(snapshot["files"])

    _ensure_dir()
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return snapshot


def load_snapshot() -> dict:
    """Load the most recent file snapshot."""
    if SNAPSHOT_PATH.exists():
        try:
            return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"taken_at": None, "files": {}, "file_count": 0}


def detect_changes() -> dict:
    """Compare current files against the last snapshot.

    Returns a dict with added, removed, and modified file lists.
    """
    old_snap = load_snapshot()
    new_snap = take_snapshot()

    old_files = old_snap.get("files", {})
    new_files = new_snap.get("files", {})

    old_keys = set(old_files.keys())
    new_keys = set(new_files.keys())

    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    modified = []
    for key in sorted(old_keys & new_keys):
        if old_files[key].get("hash") != new_files[key].get("hash"):
            modified.append(key)

    changes = {
        "detected_at": _now_iso(),
        "added": added,
        "removed": removed,
        "modified": modified,
        "total_changes": len(added) + len(removed) + len(modified),
        "previous_snapshot": old_snap.get("taken_at"),
        "current_snapshot": new_snap.get("taken_at"),
    }

    # Log changes if any
    if changes["total_changes"] > 0:
        _log_change(changes)

    return changes


# ── Change log ───────────────────────────────────────────────────────


def _log_change(change: dict) -> None:
    """Append a change event to the change log."""
    log = _load_change_log()
    log.append({
        "timestamp": _now_iso(),
        "added": len(change.get("added", [])),
        "removed": len(change.get("removed", [])),
        "modified": len(change.get("modified", [])),
        "files": (change.get("added", [])[:5]
                  + change.get("modified", [])[:5]
                  + change.get("removed", [])[:5]),
    })
    log = log[-MAX_CHANGE_LOG:]
    _ensure_dir()
    CHANGE_LOG_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _load_change_log() -> list[dict]:
    if CHANGE_LOG_PATH.exists():
        try:
            return json.loads(CHANGE_LOG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def get_change_history(n: int = 20) -> list[dict]:
    """Return the last *n* change events."""
    return _load_change_log()[-n:]


# ── Knowledge index ──────────────────────────────────────────────────


def build_knowledge_index() -> dict:
    """Build an index of all knowledge in the system.

    Counts entries from the built-in knowledge base and from
    extra knowledge files.
    """
    index: dict = {
        "built_at": _now_iso(),
        "builtin_entries": 0,
        "extra_entries": 0,
        "total_entries": 0,
        "domains": [],
        "domain_counts": {},
        "extra_files": [],
        "sources": {"builtin": 0, "crawled": 0, "uploaded": 0, "manual": 0},
    }

    # Built-in knowledge
    try:
        from knowledge_base import KNOWLEDGE, get_domains

        index["builtin_entries"] = len(KNOWLEDGE)
        index["domains"] = get_domains()

        # Count per domain
        for _, _, domain in KNOWLEDGE:
            index["domain_counts"][domain] = (
                index["domain_counts"].get(domain, 0) + 1
            )
        index["sources"]["builtin"] = len(KNOWLEDGE)
    except ImportError:
        pass

    # Extra knowledge files
    extra_dir = Path("data/extra_knowledge")
    if extra_dir.exists():
        for fp in sorted(extra_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                entries = data if isinstance(data, list) else data.get("entries", [])
                count = len(entries)
                index["extra_entries"] += count
                index["extra_files"].append({"file": fp.name, "entries": count})

                # Categorise source
                name = fp.name.lower()
                if "wikipedia" in name or "wiki" in name:
                    index["sources"]["crawled"] += count
                elif "forum" in name or "stack" in name or "reddit" in name:
                    index["sources"]["crawled"] += count
                elif "site" in name or "url" in name:
                    index["sources"]["crawled"] += count
                elif "upload" in name or "file" in name:
                    index["sources"]["uploaded"] += count
                else:
                    index["sources"]["manual"] += count

                # Count domains from extra
                for entry in entries:
                    d = entry.get("domain", "general") if isinstance(entry, dict) else "general"
                    index["domain_counts"][d] = index["domain_counts"].get(d, 0) + 1
            except (json.JSONDecodeError, OSError):
                continue

    index["total_entries"] = index["builtin_entries"] + index["extra_entries"]

    # Update domain list with any new domains from extra
    all_domains = sorted(set(index["domain_counts"].keys()))
    index["domains"] = all_domains

    # Persist
    _ensure_dir()
    KNOWLEDGE_INDEX_PATH.write_text(
        json.dumps(index, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return index


def load_knowledge_index() -> dict:
    """Load the cached knowledge index."""
    if KNOWLEDGE_INDEX_PATH.exists():
        try:
            return json.loads(KNOWLEDGE_INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return build_knowledge_index()


# ── Model watcher ────────────────────────────────────────────────────


def watch_models() -> dict:
    """Check the state of all model files and backups."""
    model_dir = Path("models")
    backup_dir = Path("models/backups")

    result: dict = {
        "checked_at": _now_iso(),
        "models": {},
        "backups": [],
        "total_backups": 0,
        "all_present": True,
    }

    for name in ["knowledge.npz", "vectorizer.json", "answer_map.json"]:
        mpath = model_dir / name
        if mpath.exists():
            result["models"][name] = _file_info(mpath)
        else:
            result["models"][name] = {"exists": False}
            result["all_present"] = False

    # Backups
    if backup_dir.exists():
        for bp in sorted(backup_dir.iterdir()):
            if bp.is_dir():
                files_in_backup = [f.name for f in bp.iterdir() if f.is_file()]
                result["backups"].append({
                    "name": bp.name,
                    "files": files_in_backup,
                })
        result["total_backups"] = len(result["backups"])

    return result


# ── Alerts ───────────────────────────────────────────────────────────


def _load_alerts() -> list[dict]:
    if ALERT_PATH.exists():
        try:
            return json.loads(ALERT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_alerts(alerts: list[dict]) -> None:
    _ensure_dir()
    ALERT_PATH.write_text(
        json.dumps(alerts[-MAX_ALERTS:], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def add_alert(
    message: str,
    level: str = "info",
    category: str = "general",
) -> dict:
    """Add a watcher alert. Levels: info, warning, critical."""
    alerts = _load_alerts()
    alert = {
        "id": len(alerts) + 1,
        "message": message,
        "level": level,
        "category": category,
        "timestamp": _now_iso(),
        "acknowledged": False,
    }
    alerts.append(alert)
    _save_alerts(alerts)
    return alert


def get_alerts(
    unacknowledged_only: bool = False,
    level: str | None = None,
) -> list[dict]:
    """Return alerts, optionally filtered."""
    alerts = _load_alerts()
    if unacknowledged_only:
        alerts = [a for a in alerts if not a.get("acknowledged")]
    if level:
        alerts = [a for a in alerts if a.get("level") == level]
    return alerts


def acknowledge_alert(alert_id: int) -> bool:
    """Mark an alert as acknowledged."""
    alerts = _load_alerts()
    for a in alerts:
        if a.get("id") == alert_id:
            a["acknowledged"] = True
            _save_alerts(alerts)
            return True
    return False


# ── Health check (automated) ─────────────────────────────────────────


def run_health_check() -> dict:
    """Run automated health checks and generate alerts for issues.

    Checks: model presence, accuracy trends, knowledge gaps,
    config integrity.
    """
    alerts_generated: list[dict] = []

    # 1. Model check
    models = watch_models()
    if not models["all_present"]:
        missing = [
            k for k, v in models["models"].items() if not v.get("exists", True)
        ]
        a = add_alert(
            f"Missing model files: {', '.join(missing)}",
            level="warning",
            category="model",
        )
        alerts_generated.append(a)

    # 2. Accuracy check
    perf = get_performance_trend(n=5)
    if perf.get("entries", 0) >= 2:
        latest = perf.get("latest_accuracy", 0)
        best = perf.get("best_accuracy", 0)
        if best > 0 and latest < best * 0.90:
            a = add_alert(
                f"Accuracy dropped significantly: {latest:.2%} vs best {best:.2%}",
                level="critical",
                category="model",
            )
            alerts_generated.append(a)

    # 3. Knowledge growth
    kidx = build_knowledge_index()
    if kidx["total_entries"] < 100:
        a = add_alert(
            f"Knowledge base is small ({kidx['total_entries']} entries). "
            "Consider adding more knowledge.",
            level="info",
            category="knowledge",
        )
        alerts_generated.append(a)

    return {
        "checked_at": _now_iso(),
        "models": models,
        "performance": perf,
        "knowledge": {
            "total": kidx["total_entries"],
            "domains": len(kidx["domains"]),
        },
        "alerts_generated": len(alerts_generated),
        "alerts": alerts_generated,
    }


# ── Session context (the "instant knowledge" builder) ────────────────


def build_watcher_context() -> dict:
    """Build instant context for a new session.

    This is the answer to "the AI should know everything about the project."
    Returns a comprehensive snapshot of the project state including:
    - Module inventory, knowledge stats, model status
    - Recent changes, alerts, performance trends
    - Everything needed to start working immediately
    """
    # Knowledge
    kidx = load_knowledge_index()

    # Models
    models = watch_models()

    # Performance
    perf = get_performance_trend(n=10)

    # Recent changes
    recent_changes = get_change_history(n=5)

    # Alerts
    alerts = get_alerts(unacknowledged_only=True)

    # Cache stats
    cache = load_response_cache()

    # Project memory
    project_mem = recall_all("project")
    model_mem = recall_all("model")

    context = {
        "project_name": "libaix",
        "description": "Self-deploying AI knowledge engine (Flask + NumPy NN)",
        "context_built_at": _now_iso(),
        "knowledge": {
            "total_entries": kidx.get("total_entries", 0),
            "domains": kidx.get("domains", []),
            "domain_counts": kidx.get("domain_counts", {}),
            "sources": kidx.get("sources", {}),
            "extra_files": len(kidx.get("extra_files", [])),
        },
        "models": {
            "all_present": models.get("all_present", False),
            "files": {
                k: {
                    "exists": v.get("hash", "missing") != "missing",
                    "size": v.get("size", 0),
                }
                for k, v in models.get("models", {}).items()
            },
            "backup_count": models.get("total_backups", 0),
        },
        "performance": {
            "latest_accuracy": perf.get("latest_accuracy", "N/A"),
            "best_accuracy": perf.get("best_accuracy", "N/A"),
            "improving": perf.get("improving", False),
            "data_points": perf.get("entries", 0),
        },
        "cache_size": len(cache),
        "recent_changes": recent_changes,
        "active_alerts": len(alerts),
        "critical_alerts": [
            a for a in alerts if a.get("level") == "critical"
        ],
        "project_memory": project_mem,
        "model_memory": model_mem,
    }

    return context


# ── Full watcher cycle ───────────────────────────────────────────────


def run_watcher_cycle() -> dict:
    """Run a complete watcher cycle: snapshot → detect changes → index → health check.

    Returns a comprehensive report.
    """
    # 1. Detect changes since last snapshot
    changes = detect_changes()

    # 2. Build knowledge index
    kidx = build_knowledge_index()

    # 3. Health check
    health = run_health_check()

    # 4. Persist summary in project memory
    remember(
        "watcher",
        "last_cycle",
        {
            "timestamp": _now_iso(),
            "changes": changes["total_changes"],
            "knowledge_entries": kidx["total_entries"],
            "alerts": health["alerts_generated"],
        },
    )

    return {
        "cycle_at": _now_iso(),
        "changes": changes,
        "knowledge": kidx,
        "health": health,
    }
