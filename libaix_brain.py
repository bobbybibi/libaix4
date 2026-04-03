"""
libaix_brain.py — LIBAIXBrain: autonomous project orchestrator.

The "brain" of the libaix project. Scans the entire codebase, builds a
live manifest of every module, feature, config file, and data asset,
then identifies gaps, generates improvement tasks, and can run in
**full-auto mode** to continuously drive project quality forward.

Capabilities:
  • Project manifest — complete inventory of modules, routes, tests, data
  • Feature gap analysis — identify missing tests, docs, endpoints
  • Task planning — generate prioritised task lists for subagents
  • Health scoring — overall project quality score (0–100)
  • Auto-mode — iteratively assess → plan → delegate → verify
  • Session briefing — instant context dump for new sessions
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from project_memory import (
    add_insight,
    get_performance_trend,
    load_response_cache,
    recall_all,
    remember,
)

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(".")
BRAIN_STATE_DIR = Path("data/brain")
BRAIN_STATE_PATH = BRAIN_STATE_DIR / "brain_state.json"
TASK_QUEUE_PATH = BRAIN_STATE_DIR / "task_queue.json"
SESSION_LOG_PATH = BRAIN_STATE_DIR / "session_log.json"

MAX_TASK_QUEUE = 200
MAX_SESSION_LOG = 50

# Modules the brain knows about
CORE_MODULES = [
    "neural_network.py",
    "vectorizer.py",
    "knowledge_base.py",
    "train.py",
    "train_knowledge.py",
    "app.py",
    "admin.py",
    "admin_chatbot.py",
    "project_memory.py",
    "ml_engine.py",
    "crawler.py",
    "forum_crawler.py",
    "site_crawler.py",
    "file_processor.py",
    "digest_engine.py",
    "local_scheduler.py",
    "libaix_brain.py",
    "ml_watcher.py",
]

TEST_DIR = Path("tests")
TEMPLATE_DIR = Path("templates")
DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# Subagent registry (matches .github/agents/*.agent.md)
KNOWN_AGENTS = {
    "developer": "Feature implementation, new code",
    "tester": "Write and run tests",
    "reviewer": "Code review and quality audit",
    "researcher": "Read-only codebase exploration",
    "deployer": "Git operations, deployment, server management",
    "libaix-brain": "Orchestrator — project scanning, task planning, auto-mode",
    "ml-watcher": "ML monitor — tracks models, data, knowledge changes",
}


# ── Helpers ──────────────────────────────────────────────────────────


def _ensure_dir() -> None:
    BRAIN_STATE_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_lines(path: Path) -> int:
    """Count lines in a text file, returning 0 on error."""
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError:
        return 0


def _extract_functions(path: Path) -> list[str]:
    """Return a list of top-level function/method names defined in *path*."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return re.findall(r"^(?:def|class)\s+(\w+)", text, re.MULTILINE)


def _extract_routes(path: Path) -> list[dict]:
    """Extract Flask route definitions from a file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    routes: list[dict] = []
    for m in re.finditer(
        r'@\w+\.route\(\s*["\']([^"\']+)["\'].*?\)\s*\ndef\s+(\w+)',
        text,
        re.DOTALL,
    ):
        routes.append({"path": m.group(1), "handler": m.group(2)})
    return routes


def _extract_test_names(path: Path) -> list[str]:
    """Extract test function names from a pytest file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return re.findall(r"def\s+(test_\w+)", text)


# ── Brain state persistence ──────────────────────────────────────────


def load_brain_state() -> dict:
    """Load the brain's persistent state from disk."""
    if BRAIN_STATE_PATH.exists():
        try:
            return json.loads(BRAIN_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return _default_brain_state()


def save_brain_state(state: dict) -> None:
    """Persist brain state to disk."""
    _ensure_dir()
    state["_updated_at"] = _now_iso()
    BRAIN_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _default_brain_state() -> dict:
    return {
        "_created_at": _now_iso(),
        "_updated_at": _now_iso(),
        "last_scan": None,
        "manifest": {},
        "health_score": 0,
        "gaps": [],
        "auto_mode": False,
        "cycle_count": 0,
    }


# ── Task queue ───────────────────────────────────────────────────────


def load_task_queue() -> list[dict]:
    """Load the pending task queue."""
    if TASK_QUEUE_PATH.exists():
        try:
            return json.loads(TASK_QUEUE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_task_queue(queue: list[dict]) -> None:
    """Persist the task queue."""
    _ensure_dir()
    TASK_QUEUE_PATH.write_text(
        json.dumps(queue[-MAX_TASK_QUEUE:], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def add_task(
    title: str,
    description: str,
    agent: str = "developer",
    priority: int = 5,
    category: str = "feature",
) -> dict:
    """Add a task to the queue. Priority 1 = highest, 10 = lowest."""
    queue = load_task_queue()
    task = {
        "id": len(queue) + 1,
        "title": title,
        "description": description,
        "agent": agent,
        "priority": max(1, min(10, priority)),
        "category": category,
        "status": "pending",
        "created_at": _now_iso(),
        "completed_at": None,
    }
    queue.append(task)
    save_task_queue(queue)
    return task


def complete_task(task_id: int) -> bool:
    """Mark a task as completed."""
    queue = load_task_queue()
    for task in queue:
        if task["id"] == task_id and task["status"] == "pending":
            task["status"] = "completed"
            task["completed_at"] = _now_iso()
            save_task_queue(queue)
            return True
    return False


def get_pending_tasks(agent: str | None = None) -> list[dict]:
    """Return pending tasks, optionally filtered by agent, sorted by priority."""
    queue = load_task_queue()
    pending = [t for t in queue if t["status"] == "pending"]
    if agent:
        pending = [t for t in pending if t["agent"] == agent]
    return sorted(pending, key=lambda t: t["priority"])


# ── Session log ──────────────────────────────────────────────────────


def _load_session_log() -> list[dict]:
    if SESSION_LOG_PATH.exists():
        try:
            return json.loads(SESSION_LOG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def log_session(summary: str, actions: list[str] | None = None) -> None:
    """Record a brain session for audit trail."""
    _ensure_dir()
    log = _load_session_log()
    log.append({
        "timestamp": _now_iso(),
        "summary": summary,
        "actions": actions or [],
    })
    log = log[-MAX_SESSION_LOG:]
    SESSION_LOG_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def get_session_log() -> list[dict]:
    """Return the session audit log."""
    return _load_session_log()


# ── Project scanner ──────────────────────────────────────────────────


def scan_project() -> dict:
    """Build a complete manifest of the project's current state.

    Returns a dict with modules, routes, tests, data files, models,
    templates, configs, and statistics.
    """
    manifest: dict = {
        "scanned_at": _now_iso(),
        "modules": {},
        "routes": [],
        "tests": {},
        "data_files": [],
        "model_files": [],
        "templates": [],
        "configs": [],
        "agents": list(KNOWN_AGENTS.keys()),
        "stats": {},
    }

    # Scan Python modules
    total_lines = 0
    total_functions = 0
    for mod_name in CORE_MODULES:
        mod_path = PROJECT_ROOT / mod_name
        if mod_path.exists():
            lines = _count_lines(mod_path)
            funcs = _extract_functions(mod_path)
            total_lines += lines
            total_functions += len(funcs)
            manifest["modules"][mod_name] = {
                "lines": lines,
                "functions": len(funcs),
                "function_names": funcs[:30],  # Cap at 30 names
                "exists": True,
            }
        else:
            manifest["modules"][mod_name] = {"exists": False}

    # Scan routes from app.py and admin.py
    for route_file in ["app.py", "admin.py"]:
        rpath = PROJECT_ROOT / route_file
        if rpath.exists():
            routes = _extract_routes(rpath)
            for r in routes:
                r["file"] = route_file
            manifest["routes"].extend(routes)

    # Scan tests
    total_tests = 0
    if TEST_DIR.exists():
        for tp in sorted(TEST_DIR.glob("test_*.py")):
            test_names = _extract_test_names(tp)
            total_tests += len(test_names)
            manifest["tests"][tp.name] = {
                "count": len(test_names),
                "names": test_names[:50],
            }

    # Scan data files
    if DATA_DIR.exists():
        for dp in sorted(DATA_DIR.rglob("*")):
            if dp.is_file():
                manifest["data_files"].append(str(dp))

    # Scan model files
    if MODEL_DIR.exists():
        for mp in sorted(MODEL_DIR.rglob("*")):
            if mp.is_file():
                manifest["model_files"].append(str(mp))

    # Scan templates
    if TEMPLATE_DIR.exists():
        for tp in sorted(TEMPLATE_DIR.rglob("*")):
            if tp.is_file():
                manifest["templates"].append(str(tp))

    # Scan config files in data/
    if DATA_DIR.exists():
        for cp in sorted(DATA_DIR.glob("*.json")):
            manifest["configs"].append(cp.name)

    # Stats
    manifest["stats"] = {
        "total_python_lines": total_lines,
        "total_functions": total_functions,
        "total_modules": sum(
            1 for m in manifest["modules"].values() if m.get("exists")
        ),
        "total_routes": len(manifest["routes"]),
        "total_tests": total_tests,
        "total_test_files": len(manifest["tests"]),
        "total_data_files": len(manifest["data_files"]),
        "total_model_files": len(manifest["model_files"]),
        "total_templates": len(manifest["templates"]),
        "total_agents": len(KNOWN_AGENTS),
    }

    # Persist
    state = load_brain_state()
    state["last_scan"] = _now_iso()
    state["manifest"] = manifest
    save_brain_state(state)

    return manifest


# ── Gap analysis ─────────────────────────────────────────────────────


def analyse_gaps(manifest: dict | None = None) -> list[dict]:
    """Identify missing features, tests, and quality issues.

    Returns a list of gap dicts with title, description, severity, category.
    """
    if manifest is None:
        state = load_brain_state()
        manifest = state.get("manifest", {})
        if not manifest:
            manifest = scan_project()

    gaps: list[dict] = []
    modules = manifest.get("modules", {})
    tests = manifest.get("tests", {})
    test_file_names = set(tests.keys())

    # 1. Modules without dedicated test files
    for mod_name, info in modules.items():
        if not info.get("exists"):
            continue
        if mod_name in ("app.py", "admin.py"):
            continue  # These have separate test files
        expected_test = f"test_{mod_name.replace('.py', '')}.py"
        if expected_test not in test_file_names:
            gaps.append({
                "title": f"Missing tests for {mod_name}",
                "description": f"Module {mod_name} ({info.get('lines', 0)} lines, "
                f"{info.get('functions', 0)} functions) has no dedicated test file "
                f"'{expected_test}'.",
                "severity": "medium",
                "category": "testing",
                "agent": "tester",
            })

    # 2. Low test count relative to functions
    for test_file, tinfo in tests.items():
        mod_name = test_file.replace("test_", "").replace(".py", "") + ".py"
        mod_info = modules.get(mod_name, {})
        func_count = mod_info.get("functions", 0)
        test_count = tinfo.get("count", 0)
        if func_count > 0 and test_count < func_count // 2:
            gaps.append({
                "title": f"Low test coverage in {test_file}",
                "description": f"{test_file} has {test_count} tests but {mod_name} "
                f"has {func_count} functions. Consider adding more tests.",
                "severity": "low",
                "category": "testing",
                "agent": "tester",
            })

    # 3. Model files missing
    model_files = manifest.get("model_files", [])
    model_names = [os.path.basename(f) for f in model_files]
    for required in ["knowledge.npz", "vectorizer.json", "answer_map.json"]:
        if required not in model_names:
            gaps.append({
                "title": f"Missing model file: {required}",
                "description": f"The trained model file '{required}' is not present. "
                "Run 'python train_knowledge.py' to generate it.",
                "severity": "high",
                "category": "model",
                "agent": "developer",
            })

    # 4. Check for important data configs
    configs = set(manifest.get("configs", []))
    for required_cfg in [
        "crawler_config.json",
        "forum_config.json",
        "cron_config.json",
        "ml_engine_config.json",
    ]:
        if required_cfg not in configs:
            gaps.append({
                "title": f"Missing config: {required_cfg}",
                "description": f"Configuration file 'data/{required_cfg}' is missing. "
                "It may be auto-generated on first use.",
                "severity": "low",
                "category": "config",
                "agent": "developer",
            })

    # 5. Performance regression check
    perf = get_performance_trend(n=5)
    if perf.get("entries", 0) >= 2 and not perf.get("improving", True):
        latest_acc = perf.get("latest_accuracy", 0)
        best_acc = perf.get("best_accuracy", 0)
        if best_acc > 0 and latest_acc < best_acc * 0.95:
            gaps.append({
                "title": "Model accuracy regression detected",
                "description": f"Latest accuracy ({latest_acc:.2%}) is below "
                f"best ({best_acc:.2%}). Consider running ML stabilize.",
                "severity": "high",
                "category": "model",
                "agent": "developer",
            })

    # 6. Response cache health
    cache = load_response_cache()
    if len(cache) > 400:
        gaps.append({
            "title": "Response cache nearing limit",
            "description": f"Cache has {len(cache)}/{500} entries. "
            "Old entries will be pruned automatically.",
            "severity": "low",
            "category": "performance",
            "agent": "developer",
        })

    # Store gaps
    state = load_brain_state()
    state["gaps"] = gaps
    save_brain_state(state)

    return gaps


# ── Health scoring ───────────────────────────────────────────────────


def calculate_health_score(manifest: dict | None = None) -> dict:
    """Calculate an overall project health score (0–100).

    Components:
      • Module completeness (20 points)
      • Test coverage breadth (25 points)
      • Model readiness (20 points)
      • Data/config completeness (15 points)
      • Performance trend (20 points)
    """
    if manifest is None:
        state = load_brain_state()
        manifest = state.get("manifest", {})
        if not manifest:
            manifest = scan_project()

    scores: dict[str, float] = {}
    stats = manifest.get("stats", {})

    # Module completeness (20 points)
    total_mods = stats.get("total_modules", 0)
    expected_mods = len(CORE_MODULES)
    scores["modules"] = min(20, round(20 * total_mods / max(1, expected_mods), 1))

    # Test coverage breadth (25 points)
    test_files = stats.get("total_test_files", 0)
    testable_modules = sum(
        1
        for m, info in manifest.get("modules", {}).items()
        if info.get("exists") and m not in ("app.py", "admin.py")
    )
    coverage_ratio = test_files / max(1, testable_modules)
    scores["tests"] = min(25, round(25 * min(1.0, coverage_ratio), 1))

    # Model readiness (20 points)
    model_files = [os.path.basename(f) for f in manifest.get("model_files", [])]
    required_models = {"knowledge.npz", "vectorizer.json", "answer_map.json"}
    present = len(required_models & set(model_files))
    scores["models"] = round(20 * present / len(required_models), 1)

    # Data/config completeness (15 points)
    configs = set(manifest.get("configs", []))
    required_configs = {
        "crawler_config.json",
        "forum_config.json",
        "cron_config.json",
        "ml_engine_config.json",
    }
    config_score = len(required_configs & configs) / len(required_configs)
    scores["configs"] = round(15 * config_score, 1)

    # Performance trend (20 points)
    perf = get_performance_trend(n=5)
    perf_score = 0.0
    if perf.get("entries", 0) > 0:
        acc = perf.get("latest_accuracy", 0)
        perf_score = min(20, round(20 * min(1.0, acc / 0.95), 1))
        if perf.get("improving"):
            perf_score = min(20, perf_score + 2)
    scores["performance"] = perf_score

    total = round(sum(scores.values()), 1)

    result = {
        "total": total,
        "max": 100,
        "components": scores,
        "grade": _score_to_grade(total),
        "assessed_at": _now_iso(),
    }

    # Persist
    state = load_brain_state()
    state["health_score"] = total
    save_brain_state(state)
    remember("project", "health_score", total)

    return result


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# ── Task generation ──────────────────────────────────────────────────


def generate_tasks_from_gaps(gaps: list[dict] | None = None) -> list[dict]:
    """Convert gap analysis results into actionable tasks in the queue."""
    if gaps is None:
        gaps = analyse_gaps()

    created: list[dict] = []
    existing_titles = {t["title"] for t in load_task_queue()}

    priority_map = {"high": 2, "medium": 5, "low": 8}

    for gap in gaps:
        title = f"Fix: {gap['title']}"
        if title in existing_titles:
            continue
        task = add_task(
            title=title,
            description=gap["description"],
            agent=gap.get("agent", "developer"),
            priority=priority_map.get(gap.get("severity", "medium"), 5),
            category=gap.get("category", "improvement"),
        )
        created.append(task)

    return created


# ── Session briefing ─────────────────────────────────────────────────


def build_session_briefing() -> dict:
    """Build a comprehensive context briefing for a new session.

    This is the solution to "the AI doesn't know about the project":
    call this at the start of any session to get full context.
    """
    state = load_brain_state()
    manifest = state.get("manifest", {})
    if not manifest:
        manifest = scan_project()

    stats = manifest.get("stats", {})
    gaps = state.get("gaps", [])
    pending = get_pending_tasks()

    # Pull from project memory
    project_mem = recall_all("project")
    model_mem = recall_all("model")

    perf = get_performance_trend(n=5)

    briefing = {
        "project": "libaix — self-deploying AI knowledge engine",
        "tech_stack": "Python 3.10+, Flask, NumPy (pure, no ML frameworks)",
        "stats": {
            "python_lines": stats.get("total_python_lines", 0),
            "modules": stats.get("total_modules", 0),
            "routes": stats.get("total_routes", 0),
            "tests": stats.get("total_tests", 0),
            "test_files": stats.get("total_test_files", 0),
            "agents": stats.get("total_agents", 0),
        },
        "health_score": state.get("health_score", 0),
        "gaps_count": len(gaps),
        "high_priority_gaps": [g for g in gaps if g.get("severity") == "high"],
        "pending_tasks": len(pending),
        "top_tasks": pending[:5],
        "performance": perf,
        "project_memory": project_mem,
        "model_memory": model_mem,
        "known_agents": KNOWN_AGENTS,
        "last_scan": state.get("last_scan"),
        "auto_mode": state.get("auto_mode", False),
        "cycle_count": state.get("cycle_count", 0),
        "briefed_at": _now_iso(),
    }

    log_session("Session briefing generated", ["build_session_briefing"])
    return briefing


# ── Full-auto mode ───────────────────────────────────────────────────


def run_full_scan_cycle() -> dict:
    """Run a complete brain cycle: scan → analyse → score → plan.

    This is what "full auto" mode runs. Returns a summary of findings
    and any new tasks created.
    """
    actions: list[str] = []

    # Step 1: Scan
    manifest = scan_project()
    actions.append(f"Scanned project: {manifest['stats']['total_modules']} modules, "
                   f"{manifest['stats']['total_routes']} routes, "
                   f"{manifest['stats']['total_tests']} tests")

    # Step 2: Analyse gaps
    gaps = analyse_gaps(manifest)
    actions.append(f"Found {len(gaps)} gaps")

    # Step 3: Health score
    health = calculate_health_score(manifest)
    actions.append(f"Health score: {health['total']}/100 (grade {health['grade']})")

    # Step 4: Generate tasks
    new_tasks = generate_tasks_from_gaps(gaps)
    actions.append(f"Created {len(new_tasks)} new tasks")

    # Step 5: Record insight
    add_insight(
        f"Brain cycle #{load_brain_state().get('cycle_count', 0)}: "
        f"health={health['total']}/100, gaps={len(gaps)}, new_tasks={len(new_tasks)}",
        category="brain_cycle",
    )

    # Increment cycle count
    state = load_brain_state()
    state["cycle_count"] = state.get("cycle_count", 0) + 1
    save_brain_state(state)

    result = {
        "manifest_stats": manifest["stats"],
        "gaps": gaps,
        "health": health,
        "new_tasks": new_tasks,
        "actions": actions,
        "cycle": state["cycle_count"],
    }

    log_session("Full scan cycle completed", actions)
    return result


def set_auto_mode(enabled: bool) -> dict:
    """Enable or disable full-auto mode."""
    state = load_brain_state()
    state["auto_mode"] = enabled
    save_brain_state(state)
    remember("project", "brain_auto_mode", enabled)
    return {"auto_mode": enabled, "updated_at": _now_iso()}


def get_auto_mode() -> bool:
    """Check if auto-mode is enabled."""
    state = load_brain_state()
    return state.get("auto_mode", False)


# ── Quick status ─────────────────────────────────────────────────────


def get_status() -> dict:
    """Return a compact status summary of the brain."""
    state = load_brain_state()
    pending = get_pending_tasks()
    return {
        "health_score": state.get("health_score", 0),
        "last_scan": state.get("last_scan"),
        "auto_mode": state.get("auto_mode", False),
        "cycle_count": state.get("cycle_count", 0),
        "pending_tasks": len(pending),
        "gaps_count": len(state.get("gaps", [])),
        "known_agents": list(KNOWN_AGENTS.keys()),
    }


# ── Dependency graph ─────────────────────────────────────────────────


def _extract_imports(path: Path) -> list[str]:
    """Extract local project imports from a Python file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    imports: list[str] = []
    all_modules = {m.replace(".py", "") for m in CORE_MODULES}
    for m in re.finditer(
        r"^(?:from|import)\s+([\w.]+)", text, re.MULTILINE
    ):
        mod_name = m.group(1).split(".")[0]
        if mod_name in all_modules:
            imports.append(mod_name + ".py")
    return sorted(set(imports))


def build_dependency_graph() -> dict:
    """Build a module-level dependency graph for the project.

    Returns a dict mapping each module to its local imports, plus
    summary statistics (edges count, most-depended-on modules,
    most-dependent modules, circular deps).
    """
    graph: dict[str, list[str]] = {}
    for mod_name in CORE_MODULES:
        mod_path = PROJECT_ROOT / mod_name
        if mod_path.exists():
            deps = _extract_imports(mod_path)
            # Exclude self-imports
            graph[mod_name] = [d for d in deps if d != mod_name]

    # Reverse graph: who depends on me?
    reverse: dict[str, list[str]] = {m: [] for m in graph}
    for mod, deps in graph.items():
        for dep in deps:
            if dep in reverse:
                reverse[dep].append(mod)

    # Detect circular dependencies (direct A↔B cycles)
    circular: list[list[str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for mod, deps in graph.items():
        for dep in deps:
            if mod in graph.get(dep, []):
                pair = tuple(sorted([mod, dep]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    circular.append(list(pair))

    # Stats
    total_edges = sum(len(deps) for deps in graph.values())
    most_depended = sorted(
        reverse.items(), key=lambda kv: len(kv[1]), reverse=True
    )[:5]
    most_dependent = sorted(
        graph.items(), key=lambda kv: len(kv[1]), reverse=True
    )[:5]

    # Leaf modules (no local deps)
    leaf_modules = [m for m, deps in graph.items() if not deps]

    result = {
        "graph": graph,
        "reverse_graph": {k: v for k, v in reverse.items() if v},
        "total_modules": len(graph),
        "total_edges": total_edges,
        "most_depended_on": [
            {"module": m, "depended_by": len(d)} for m, d in most_depended
        ],
        "most_dependent": [
            {"module": m, "depends_on": len(d)} for m, d in most_dependent
        ],
        "leaf_modules": leaf_modules,
        "circular_dependencies": circular,
        "built_at": _now_iso(),
    }

    # Persist
    state = load_brain_state()
    state["dependency_graph"] = {
        "total_edges": total_edges,
        "circular": len(circular),
        "built_at": result["built_at"],
    }
    save_brain_state(state)

    return result


# ── Module complexity scoring ────────────────────────────────────────


def _count_classes(path: Path) -> int:
    """Count class definitions in a Python file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    return len(re.findall(r"^class\s+\w+", text, re.MULTILINE))


def _count_branches(path: Path) -> int:
    """Count branching statements (if/elif/for/while/try/except)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    return len(re.findall(
        r"^\s+(?:if|elif|for|while|try|except)\s", text, re.MULTILINE
    ))


def score_module_complexity() -> dict:
    """Score each module's complexity based on lines, functions, classes,
    and branching statements. Returns sorted by complexity (highest first).
    """
    results: list[dict] = []
    for mod_name in CORE_MODULES:
        mod_path = PROJECT_ROOT / mod_name
        if not mod_path.exists():
            continue
        lines = _count_lines(mod_path)
        funcs = len(_extract_functions(mod_path))
        classes = _count_classes(mod_path)
        branches = _count_branches(mod_path)

        # Weighted complexity score
        score = (
            lines * 0.1         # 1 point per 10 lines
            + funcs * 2.0       # 2 points per function
            + classes * 5.0     # 5 points per class
            + branches * 1.5    # 1.5 points per branch
        )

        results.append({
            "module": mod_name,
            "lines": lines,
            "functions": funcs,
            "classes": classes,
            "branches": branches,
            "complexity_score": round(score, 1),
        })

    results.sort(key=lambda r: r["complexity_score"], reverse=True)

    total_score = sum(r["complexity_score"] for r in results)
    avg_score = round(total_score / max(1, len(results)), 1)

    return {
        "modules": results,
        "total_complexity": round(total_score, 1),
        "average_complexity": avg_score,
        "most_complex": results[0]["module"] if results else None,
        "simplest": results[-1]["module"] if results else None,
        "assessed_at": _now_iso(),
    }


# ── Code quality metrics ─────────────────────────────────────────────


def _count_todos(path: Path) -> int:
    """Count TODO/FIXME/HACK/XXX comments in a file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    return len(re.findall(r"#\s*(?:TODO|FIXME|HACK|XXX)\b", text, re.IGNORECASE))


def _has_docstrings(path: Path) -> tuple[int, int]:
    """Return (functions_with_docstrings, total_functions) in a file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0, 0
    func_defs = list(re.finditer(r"^([ \t]*)def\s+\w+\s*\(", text, re.MULTILINE))
    total = len(func_defs)
    documented = 0
    lines = text.splitlines()
    for m in func_defs:
        line_num = text[:m.start()].count("\n")
        # Look for docstring within the next 3 lines after `def ...:`
        for offset in range(1, 4):
            idx = line_num + offset
            if idx < len(lines):
                stripped = lines[idx].strip()
                if stripped.startswith(('"""', "'''", 'r"""', "r'''")):
                    documented += 1
                    break
                if stripped and not stripped.startswith("#"):
                    break
    return documented, total


def measure_code_quality() -> dict:
    """Measure code quality metrics across all modules.

    Checks docstring coverage, TODO counts, average function length,
    and import hygiene.
    """
    module_metrics: list[dict] = []
    total_todos = 0
    total_documented = 0
    total_functions = 0

    for mod_name in CORE_MODULES:
        mod_path = PROJECT_ROOT / mod_name
        if not mod_path.exists():
            continue
        lines = _count_lines(mod_path)
        todos = _count_todos(mod_path)
        total_todos += todos
        documented, funcs = _has_docstrings(mod_path)
        total_documented += documented
        total_functions += funcs
        avg_func_len = round(lines / max(1, funcs), 1) if funcs else 0

        module_metrics.append({
            "module": mod_name,
            "lines": lines,
            "functions": funcs,
            "documented_functions": documented,
            "docstring_pct": round(100 * documented / max(1, funcs), 1),
            "todos": todos,
            "avg_function_length": avg_func_len,
        })

    overall_docstring_pct = round(
        100 * total_documented / max(1, total_functions), 1
    )

    return {
        "modules": module_metrics,
        "overall_docstring_coverage": overall_docstring_pct,
        "total_todos": total_todos,
        "total_functions": total_functions,
        "total_documented": total_documented,
        "assessed_at": _now_iso(),
    }


# ── Knowledge gap recommendations ───────────────────────────────────


def recommend_knowledge_gaps() -> dict:
    """Analyse the knowledge base and recommend areas to expand.

    Looks at domain balance, entry counts, and identifies underserved
    topics that could benefit from more Q&A pairs.
    """
    try:
        from knowledge_base import KNOWLEDGE
    except ImportError:
        return {"error": "knowledge_base not available", "recommendations": []}

    # Count per domain
    domain_counts: dict[str, int] = {}
    for _, _, domain in KNOWLEDGE:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Include extra knowledge
    extra_dir = Path("data/extra_knowledge")
    extra_domains: dict[str, int] = {}
    if extra_dir.exists():
        for fp in sorted(extra_dir.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                entries = data if isinstance(data, list) else data.get("entries", [])
                for entry in entries:
                    d = entry.get("domain", "general") if isinstance(entry, dict) else "general"
                    extra_domains[d] = extra_domains.get(d, 0) + 1
            except (json.JSONDecodeError, OSError):
                continue

    # Merge counts
    all_domains: dict[str, int] = dict(domain_counts)
    for d, c in extra_domains.items():
        all_domains[d] = all_domains.get(d, 0) + c

    total = sum(all_domains.values())
    avg_per_domain = total / max(1, len(all_domains))

    recommendations: list[dict] = []

    # Under-represented domains
    for domain, count in sorted(all_domains.items(), key=lambda kv: kv[1]):
        if count < avg_per_domain * 0.5:
            recommendations.append({
                "type": "expand_domain",
                "domain": domain,
                "current_entries": count,
                "suggested_target": int(avg_per_domain),
                "reason": f"Domain '{domain}' has only {count} entries "
                          f"(average is {avg_per_domain:.0f}). Consider adding more.",
                "priority": "high" if count < avg_per_domain * 0.25 else "medium",
            })

    # Missing common domains (suggest new topics)
    suggested_domains = [
        "cloud_computing", "devops", "database", "programming",
        "operating_systems", "web_development", "cryptography",
    ]
    existing = set(all_domains.keys())
    for sd in suggested_domains:
        if sd not in existing:
            recommendations.append({
                "type": "new_domain",
                "domain": sd,
                "current_entries": 0,
                "suggested_target": 15,
                "reason": f"Domain '{sd}' is not covered yet. "
                          "Consider adding knowledge entries.",
                "priority": "low",
            })

    return {
        "total_entries": total,
        "domain_count": len(all_domains),
        "domain_distribution": all_domains,
        "average_per_domain": round(avg_per_domain, 1),
        "recommendations": recommendations,
        "recommendation_count": len(recommendations),
        "assessed_at": _now_iso(),
    }


# ── Cross-module impact analysis ─────────────────────────────────────


def analyse_impact(target_module: str) -> dict:
    """Analyse what would be affected if *target_module* changes.

    Uses the dependency graph to find direct and transitive dependents,
    tests that cover the module, and routes it serves.
    """
    dep_graph = build_dependency_graph()
    reverse = dep_graph.get("reverse_graph", {})

    # Direct dependents
    direct = reverse.get(target_module, [])

    # Transitive dependents (BFS)
    transitive: list[str] = []
    visited: set[str] = set()
    queue = list(direct)
    while queue:
        mod = queue.pop(0)
        if mod in visited:
            continue
        visited.add(mod)
        transitive.append(mod)
        queue.extend(reverse.get(mod, []))

    # Find related tests
    test_mapping: dict[str, list[str]] = {}
    mod_base = target_module.replace(".py", "")
    if TEST_DIR.exists():
        for tp in sorted(TEST_DIR.glob("test_*.py")):
            try:
                text = tp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # Check if test file imports or references the target module
            if mod_base in text:
                test_names = _extract_test_names(tp)
                test_mapping[tp.name] = test_names

    # Find affected routes
    state = load_brain_state()
    manifest = state.get("manifest", {})
    affected_routes = [
        r for r in manifest.get("routes", [])
        if r.get("file") == target_module
    ]

    # Risk assessment
    risk = "low"
    if len(transitive) > 3:
        risk = "high"
    elif len(transitive) > 1 or len(affected_routes) > 2:
        risk = "medium"

    return {
        "target": target_module,
        "direct_dependents": direct,
        "transitive_dependents": transitive,
        "affected_tests": test_mapping,
        "affected_routes": affected_routes,
        "risk_level": risk,
        "analysis_note": (
            f"Changing {target_module} directly affects {len(direct)} module(s) "
            f"and transitively affects {len(transitive)} module(s). "
            f"{len(test_mapping)} test file(s) cover this module."
        ),
        "analysed_at": _now_iso(),
    }


# ── Stale data detection ────────────────────────────────────────────


def detect_stale_data(max_age_days: int = 30) -> dict:
    """Detect stale data files, configs, and model backups.

    Returns files that haven't been modified in *max_age_days* days.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    stale: list[dict] = []

    scan_dirs = [
        ("data", DATA_DIR),
        ("models", MODEL_DIR),
    ]

    for category, dir_path in scan_dirs:
        if not dir_path.exists():
            continue
        for fp in sorted(dir_path.rglob("*")):
            if not fp.is_file():
                continue
            try:
                mtime = datetime.fromtimestamp(
                    fp.stat().st_mtime, tz=timezone.utc
                )
                if mtime < cutoff:
                    age_days = (now - mtime).days
                    stale.append({
                        "path": str(fp),
                        "category": category,
                        "size": fp.stat().st_size,
                        "last_modified": mtime.isoformat(),
                        "age_days": age_days,
                    })
            except OSError:
                continue

    stale.sort(key=lambda s: s["age_days"], reverse=True)

    return {
        "max_age_days": max_age_days,
        "stale_files": stale,
        "stale_count": len(stale),
        "total_stale_bytes": sum(s["size"] for s in stale),
        "checked_at": _now_iso(),
    }


# ── Module summary (compact) ────────────────────────────────────────


def summarize_module(module_name: str) -> dict:
    """Generate a compact summary of a single module.

    Includes line count, functions, classes, imports, routes (if any),
    test coverage, and complexity score.
    """
    mod_path = PROJECT_ROOT / module_name
    if not mod_path.exists():
        return {"error": f"Module '{module_name}' not found", "exists": False}

    lines = _count_lines(mod_path)
    funcs = _extract_functions(mod_path)
    classes = _count_classes(mod_path)
    branches = _count_branches(mod_path)
    imports = _extract_imports(mod_path)
    todos = _count_todos(mod_path)
    documented, total_funcs = _has_docstrings(mod_path)
    routes = _extract_routes(mod_path)

    # Related test file
    mod_base = module_name.replace(".py", "")
    test_file = TEST_DIR / f"test_{mod_base}.py"
    test_count = 0
    test_names: list[str] = []
    if test_file.exists():
        test_names = _extract_test_names(test_file)
        test_count = len(test_names)

    complexity = round(
        lines * 0.1 + len(funcs) * 2.0 + classes * 5.0 + branches * 1.5, 1
    )

    return {
        "module": module_name,
        "exists": True,
        "lines": lines,
        "functions": len(funcs),
        "function_names": funcs,
        "classes": classes,
        "branches": branches,
        "local_imports": imports,
        "routes": routes,
        "todos": todos,
        "docstring_coverage": round(100 * documented / max(1, total_funcs), 1),
        "test_file": test_file.name if test_file.exists() else None,
        "test_count": test_count,
        "complexity_score": complexity,
        "summarized_at": _now_iso(),
    }
