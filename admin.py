"""
admin.py — Admin backend blueprint for libaix knowledge management.

Provides:
  • Authenticated admin dashboard
  • File upload with auto-extraction (PDF/TXT/CSV/…) + file deletion after extraction
  • Text paste → knowledge extraction
  • Wikipedia crawler management (multi-topic, add/remove/toggle)
  • AI learning prompts ("Learn about <topic>")
  • Knowledge browser
  • One-click retrain
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import (
    Blueprint,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from crawler import (
    crawl_single_topic,
    load_config,
    run_all_crawlers,
    save_config,
)
from file_processor import classify_domain, process_file, process_pasted_text
from forum_crawler import (
    crawl_single_forum_topic,
    get_learning_stats,
    load_forum_config,
    log_learning_event,
    run_all_forum_crawlers,
    save_forum_config,
)
from knowledge_base import KNOWLEDGE, get_domains
from ml_engine import (
    assess_growth,
    assess_model,
    get_engine_stats,
    load_engine_config,
    optimize_model,
    run_growth_cycle,
    save_engine_config,
    stabilize_model,
)
from site_crawler import add_site_job, clear_site_jobs, get_site_crawl_stats

# ── Config ────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("data/uploads")
EXTRA_KNOWLEDGE_DIR = Path("data/extra_knowledge")
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".csv", ".log", ".conf",
    ".cfg", ".ini", ".json", ".xml", ".html",
}

# Credentials — override via env vars ADMIN_USER / ADMIN_PASS
_admin_user = os.environ.get("ADMIN_USER", "kakababa")
_admin_pass = os.environ.get("ADMIN_PASS", "Nepidaras25!!??")
ADMIN_CREDENTIALS = {
    "username": _admin_user,
    "password_hash": generate_password_hash(_admin_pass),
}
del _admin_pass  # scrub plaintext from process memory

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


# ── Auth helper ───────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin.login"))
        return f(*args, **kwargs)
    return decorated


# ── Auth routes ───────────────────────────────────────────────────────

@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if (
            username == ADMIN_CREDENTIALS["username"]
            and check_password_hash(ADMIN_CREDENTIALS["password_hash"], password)
        ):
            session["admin_logged_in"] = True
            session["admin_user"] = username
            return redirect(url_for("admin.dashboard"))
        flash("Invalid credentials", "error")
    return render_template("admin_login.html")


@admin_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("admin.login"))


# ── Dashboard ─────────────────────────────────────────────────────────

@admin_bp.route("/")
@login_required
def dashboard():
    extra_count = _count_extra_knowledge()
    config = load_config()
    return render_template(
        "admin_dashboard.html",
        knowledge_count=len(KNOWLEDGE),
        extra_count=extra_count,
        total_count=len(KNOWLEDGE) + extra_count,
        domains=get_domains() + _get_extra_domains(),
        crawler_config=config,
    )


@admin_bp.route("/api/stats")
@login_required
def api_stats():
    extra_count = _count_extra_knowledge()
    config = load_config()
    forum_config = load_forum_config()
    site_stats = get_site_crawl_stats()
    engine_stats = get_engine_stats()
    learning_stats = get_learning_stats()
    cron_config = _load_cron_config()

    # Source breakdown from extra knowledge files
    source_breakdown = _get_source_breakdown()

    return jsonify({
        "builtin_entries": len(KNOWLEDGE),
        "extra_entries": extra_count,
        "total_entries": len(KNOWLEDGE) + extra_count,
        "domains": get_domains() + _get_extra_domains(),
        "extra_files": _list_extra_files(),
        "source_breakdown": source_breakdown,
        "crawler": {
            "topics": config.get("topics", []),
            "last_crawl": config.get("last_crawl"),
        },
        "forum_crawler": {
            "topics": forum_config.get("topics", []),
            "last_crawl": forum_config.get("last_crawl"),
            "stats": forum_config.get("stats", {}),
        },
        "site_crawler": site_stats,
        "ml_engine": engine_stats,
        "learning": learning_stats,
        "cron": cron_config,
    })


# ── File upload ───────────────────────────────────────────────────────

@admin_bp.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    domain_hint = request.form.get("domain", "")
    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported format: {suffix}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filepath = UPLOAD_DIR / filename
    file.save(filepath)

    try:
        entries, preview = process_file(filepath, domain_hint)
        if entries:
            save_path = _save_knowledge(entries, f"upload_{filename}")
            return jsonify({
                "status": "success",
                "file": filename,
                "entries_extracted": len(entries),
                "preview": preview,
                "samples": entries[:5],
                "saved_to": str(save_path),
                "message": f"Extracted {len(entries)} entries. File deleted.",
            })
        return jsonify({
            "status": "warning",
            "file": filename,
            "entries_extracted": 0,
            "preview": preview,
            "message": "No entries could be extracted from this file.",
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500
    finally:
        # ALWAYS delete the upload — space preservation
        if filepath.exists():
            filepath.unlink()


# ── Text paste ────────────────────────────────────────────────────────

@admin_bp.route("/paste", methods=["POST"])
@login_required
def paste_text():
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "No text provided"}), 400

    entries = process_pasted_text(data["text"], data.get("domain", ""))
    if entries:
        save_path = _save_knowledge(entries, "paste")
        return jsonify({
            "status": "success",
            "entries_extracted": len(entries),
            "samples": entries[:5],
            "saved_to": str(save_path),
        })
    return jsonify({
        "status": "warning",
        "entries_extracted": 0,
        "message": "No entries extracted. Try pasting structured content with definitions.",
    })


# ── Manual Q&A entry ─────────────────────────────────────────────────

@admin_bp.route("/add-entry", methods=["POST"])
@login_required
def add_entry():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    domain = (data.get("domain") or "").strip() or classify_domain(answer)
    if not question or not answer:
        return jsonify({"error": "Both question and answer are required"}), 400

    entries = [{"question": question, "answer": answer, "domain": domain}]
    save_path = _save_knowledge(entries, "manual")
    return jsonify({"status": "success", "entry": entries[0], "saved_to": str(save_path)})


# ── Crawler management ────────────────────────────────────────────────

@admin_bp.route("/crawler/topics")
@login_required
def get_crawler_topics():
    return jsonify(load_config().get("topics", []))


@admin_bp.route("/crawler/add-topic", methods=["POST"])
@login_required
def add_crawler_topic():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Topic name required"}), 400

    config = load_config()
    topics = config.get("topics", [])
    if any(t["name"].lower() == data["name"].lower() for t in topics):
        return jsonify({"error": "Topic already exists"}), 400

    topics.append({
        "name": data["name"],
        "keywords": [k.strip() for k in data.get("keywords", []) if k.strip()],
        "enabled": True,
        "max_articles": int(data.get("max_articles", 8)),
    })
    config["topics"] = topics
    save_config(config)
    return jsonify({"status": "success", "topics": topics})


@admin_bp.route("/crawler/remove-topic", methods=["POST"])
@login_required
def remove_crawler_topic():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Topic name required"}), 400
    config = load_config()
    config["topics"] = [t for t in config.get("topics", []) if t["name"] != data["name"]]
    save_config(config)
    return jsonify({"status": "success", "topics": config["topics"]})


@admin_bp.route("/crawler/toggle-topic", methods=["POST"])
@login_required
def toggle_crawler_topic():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Topic name required"}), 400
    config = load_config()
    for t in config.get("topics", []):
        if t["name"] == data["name"]:
            t["enabled"] = not t.get("enabled", True)
            break
    save_config(config)
    return jsonify({"status": "success", "topics": config["topics"]})


@admin_bp.route("/crawler/run", methods=["POST"])
@login_required
def run_crawler():
    results = run_all_crawlers()
    return jsonify(results)


@admin_bp.route("/crawler/run-topic", methods=["POST"])
@login_required
def run_single_crawler():
    data = request.get_json()
    if not data or not data.get("topic"):
        return jsonify({"error": "Topic required"}), 400
    return jsonify(crawl_single_topic(
        data["topic"],
        data.get("keywords", []),
        int(data.get("max_articles", 10)),
    ))


# ── AI learning prompts ──────────────────────────────────────────────

@admin_bp.route("/learn", methods=["POST"])
@login_required
def learn_prompt():
    """Parse an admin learning command and crawl ALL sources for the topic."""
    data = request.get_json()
    if not data or not data.get("prompt"):
        return jsonify({"error": "Prompt required"}), 400

    topic = _parse_learning_prompt(data["prompt"])
    if not topic:
        return jsonify({
            "error": "Could not parse prompt. Try: 'Learn about <topic>'"
        }), 400

    keywords = [k.strip() for k in data.get("keywords", []) if k.strip()]
    results: dict[str, dict] = {}
    total_entries = 0

    # 1) Wikipedia (single crawl — not repeated frequently)
    wiki_result = crawl_single_topic(topic, keywords, max_articles=10)
    results["wikipedia"] = wiki_result
    if wiki_result.get("status") == "success":
        total_entries += wiki_result.get("entries", 0)
        log_learning_event("wikipedia", topic, wiki_result.get("entries", 0))

    # 2) Forums (Reddit + StackExchange + HN + DEV.to)
    forum_result = crawl_single_forum_topic(
        topic, keywords, max_per_source=10,
        sources=["stackexchange", "reddit", "hackernews", "devto"],
    )
    results["forums"] = forum_result
    if forum_result.get("status") == "success":
        total_entries += forum_result.get("entries", 0)

    # Auto-add to crawler config for continuous learning
    config = load_config()
    topics = config.get("topics", [])
    if not any(t["name"].lower() == topic.lower() for t in topics):
        topics.append({
            "name": topic,
            "keywords": keywords,
            "enabled": True,
            "max_articles": 8,
        })
        config["topics"] = topics
        save_config(config)

    # Auto-add to forum config too
    forum_config = load_forum_config()
    forum_topics = forum_config.get("topics", [])
    if not any(t["name"].lower() == topic.lower() for t in forum_topics):
        forum_topics.append({
            "name": topic,
            "keywords": keywords,
            "enabled": True,
            "max_per_source": 10,
            "sources": ["stackexchange", "reddit", "hackernews", "devto"],
        })
        forum_config["topics"] = forum_topics
        save_forum_config(forum_config)

    # Collect all samples
    all_samples = []
    if wiki_result.get("samples"):
        all_samples.extend(wiki_result["samples"][:2])
    if forum_result.get("samples"):
        all_samples.extend(forum_result["samples"][:2])

    return jsonify({
        "status": "success" if total_entries > 0 else "no_results",
        "entries": total_entries,
        "sources": results,
        "source_breakdown": forum_result.get("source_breakdown", {}),
        "samples": all_samples,
        "message": (
            f"Learned {total_entries} facts about '{topic}' from multiple sources. "
            "Topic added to all crawlers for continuous learning."
        ) if total_entries > 0 else f"No results found for '{topic}'.",
    })


# ── Retrain ───────────────────────────────────────────────────────────

@admin_bp.route("/retrain", methods=["POST"])
@login_required
def retrain():
    try:
        from train_knowledge import train as _train
        model, bow, answer_map = _train(
            activation="tanh", optimizer="adam",
            lr=0.01, epochs=5000, hidden=256,
            augment=True, verbose=False,
        )
        return jsonify({
            "status": "success",
            "entries": len(answer_map),
            "vocab": bow.vocab_size,
            "message": f"Retrained on {len(answer_map)} answers, {bow.vocab_size} vocab words.",
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


# ── Knowledge browser ────────────────────────────────────────────────

@admin_bp.route("/knowledge")
@login_required
def browse_knowledge():
    domain_filter = request.args.get("domain", "")
    entries: list[dict] = []

    for q, a, d in KNOWLEDGE:
        if not domain_filter or d == domain_filter:
            entries.append({"question": q, "answer": a, "domain": d, "source": "builtin"})

    for fp in _iter_extra_files():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            for e in data:
                if not domain_filter or e.get("domain") == domain_filter:
                    entries.append({
                        "question": e["question"],
                        "answer": e["answer"],
                        "domain": e.get("domain", "general"),
                        "source": f"extra:{fp.name}",
                    })
        except Exception:
            continue

    return jsonify({"entries": entries, "total": len(entries)})


# ── Helpers ───────────────────────────────────────────────────────────

# ── Forum Crawler management ─────────────────────────────────────────

@admin_bp.route("/forum/topics")
@login_required
def get_forum_topics():
    config = load_forum_config()
    return jsonify({
        "topics": config.get("topics", []),
        "stats": config.get("stats", {}),
    })


@admin_bp.route("/forum/add-topic", methods=["POST"])
@login_required
def add_forum_topic():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Topic name required"}), 400
    config = load_forum_config()
    topics = config.get("topics", [])
    if any(t["name"].lower() == data["name"].lower() for t in topics):
        return jsonify({"error": "Topic already exists"}), 400
    topics.append({
        "name": data["name"],
        "keywords": [k.strip() for k in data.get("keywords", []) if k.strip()],
        "enabled": True,
        "max_per_source": int(data.get("max_per_source", 10)),
        "sources": data.get("sources", ["stackexchange", "reddit"]),
    })
    config["topics"] = topics
    save_forum_config(config)
    return jsonify({"status": "success", "topics": topics})


@admin_bp.route("/forum/remove-topic", methods=["POST"])
@login_required
def remove_forum_topic():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Topic name required"}), 400
    config = load_forum_config()
    config["topics"] = [t for t in config.get("topics", []) if t["name"] != data["name"]]
    save_forum_config(config)
    return jsonify({"status": "success", "topics": config["topics"]})


@admin_bp.route("/forum/toggle-topic", methods=["POST"])
@login_required
def toggle_forum_topic():
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Topic name required"}), 400
    config = load_forum_config()
    for t in config.get("topics", []):
        if t["name"] == data["name"]:
            t["enabled"] = not t.get("enabled", True)
            break
    save_forum_config(config)
    return jsonify({"status": "success", "topics": config["topics"]})


@admin_bp.route("/forum/run", methods=["POST"])
@login_required
def run_forum_crawlers():
    results = run_all_forum_crawlers()
    return jsonify(results)


@admin_bp.route("/forum/run-topic", methods=["POST"])
@login_required
def run_single_forum():
    data = request.get_json()
    if not data or not data.get("topic"):
        return jsonify({"error": "Topic required"}), 400
    return jsonify(crawl_single_forum_topic(
        data["topic"],
        data.get("keywords", []),
        int(data.get("max_per_source", 10)),
        data.get("sources"),
    ))


# ── Site Crawler (URL crawler) ───────────────────────────────────────

@admin_bp.route("/site-crawl", methods=["POST"])
@login_required
def site_crawl():
    data = request.get_json()
    if not data or not data.get("url") or not data.get("topic"):
        return jsonify({"error": "URL and topic are required"}), 400
    url = data["url"].strip()
    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        return jsonify({"error": "URL must start with http:// or https://"}), 400
    result = add_site_job(
        url=url,
        topic=data["topic"],
        keywords=[k.strip() for k in data.get("keywords", []) if k.strip()],
        max_pages=min(int(data.get("max_pages", 20)), 50),
        max_depth=min(int(data.get("max_depth", 2)), 3),
    )
    return jsonify(result)


@admin_bp.route("/site-crawl/stats")
@login_required
def site_crawl_stats():
    return jsonify(get_site_crawl_stats())


@admin_bp.route("/site-crawl/clear", methods=["POST"])
@login_required
def site_crawl_clear():
    clear_site_jobs()
    return jsonify({"status": "success", "message": "Job history cleared"})


# ── ML Engine controls ───────────────────────────────────────────────

@admin_bp.route("/ml/stats")
@login_required
def ml_stats():
    return jsonify(get_engine_stats())


@admin_bp.route("/ml/assess", methods=["POST"])
@login_required
def ml_assess():
    return jsonify(assess_model())


@admin_bp.route("/ml/optimize", methods=["POST"])
@login_required
def ml_optimize():
    result = optimize_model()
    return jsonify(result)


@admin_bp.route("/ml/stabilize", methods=["POST"])
@login_required
def ml_stabilize():
    return jsonify(stabilize_model())


@admin_bp.route("/ml/growth-check", methods=["POST"])
@login_required
def ml_growth_check():
    return jsonify(assess_growth())


@admin_bp.route("/ml/growth-cycle", methods=["POST"])
@login_required
def ml_growth_cycle():
    return jsonify(run_growth_cycle())


@admin_bp.route("/ml/config", methods=["GET", "POST"])
@login_required
def ml_config():
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        config = load_engine_config()
        for key in ["auto_optimize", "auto_stabilize", "auto_grow"]:
            if key in data:
                config[key] = bool(data[key])
        if "min_accuracy_threshold" in data:
            config["min_accuracy_threshold"] = max(0.5, min(1.0, float(data["min_accuracy_threshold"])))
        if "confidence_threshold" in data:
            config["confidence_threshold"] = max(0.1, min(1.0, float(data["confidence_threshold"])))
        if "growth_targets" in data:
            targets = config.get("growth_targets", {})
            gt = data["growth_targets"]
            if "min_entries" in gt:
                targets["min_entries"] = int(gt["min_entries"])
            if "min_domains" in gt:
                targets["min_domains"] = int(gt["min_domains"])
            if "target_accuracy" in gt:
                targets["target_accuracy"] = max(0.5, min(1.0, float(gt["target_accuracy"])))
            config["growth_targets"] = targets
        save_engine_config(config)
        return jsonify({"status": "success", "config": config})
    return jsonify(load_engine_config())


# ── Cron Job Controls ─────────────────────────────────────────────────

CRON_CONFIG_PATH = Path("data/cron_config.json")

# Redline/max safe values for cron schedules
CRON_REDLINES = {
    "auto_train": {
        "label": "Auto-Train",
        "max_safe_per_hour": 4,
        "max_absolute_per_hour": 12,
        "note": "8 parallel configs per run = 8x multiplier. >4/hr risks GitHub throttling.",
    },
    "wiki_crawler": {
        "label": "Wikipedia Crawler",
        "max_safe_per_hour": 2,
        "max_absolute_per_hour": 4,
        "note": "Wikipedia rate limits: 200 req/s but polite crawling means 1-2/hr max.",
    },
    "forum_crawler": {
        "label": "Forum Crawler (Reddit/SE/HN)",
        "max_safe_per_hour": 4,
        "max_absolute_per_hour": 8,
        "note": "SE: 300 req/day. Reddit: 60 req/min. HN: unlimited. >4/hr may hit SE limit.",
    },
    "ml_growth": {
        "label": "ML Growth Cycle",
        "max_safe_per_hour": 1,
        "max_absolute_per_hour": 2,
        "note": "Each cycle trains multiple models. Very CPU-heavy. 1/hr is optimal.",
    },
}


@admin_bp.route("/cron/config", methods=["GET", "POST"])
@login_required
def cron_config_endpoint():
    if request.method == "POST":
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        config = _load_cron_config()
        for wf_key in ["auto_train", "wiki_crawler", "forum_crawler", "ml_growth"]:
            if wf_key in data:
                wf_data = data[wf_key]
                if "enabled" in wf_data:
                    config["workflows"][wf_key]["enabled"] = bool(wf_data["enabled"])
                if "runs_per_hour" in wf_data:
                    limit = CRON_REDLINES[wf_key]["max_absolute_per_hour"]
                    config["workflows"][wf_key]["runs_per_hour"] = max(0, min(limit, int(wf_data["runs_per_hour"])))
        config["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_cron_config(config)
        return jsonify({"status": "success", "config": config, "redlines": CRON_REDLINES})
    config = _load_cron_config()
    return jsonify({"config": config, "redlines": CRON_REDLINES})


@admin_bp.route("/learning/stats")
@login_required
def learning_stats_endpoint():
    return jsonify(get_learning_stats())


# ── Helpers ───────────────────────────────────────────────────────────

def _save_knowledge(entries: list[dict], source: str) -> Path:
    EXTRA_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", source.lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = EXTRA_KNOWLEDGE_DIR / f"{safe}_{ts}.json"
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_cron_config() -> dict:
    if CRON_CONFIG_PATH.exists():
        try:
            return json.loads(CRON_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _default_cron_config()


def _save_cron_config(config: dict) -> None:
    CRON_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CRON_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _default_cron_config() -> dict:
    return {
        "workflows": {
            "auto_train": {"enabled": True, "runs_per_hour": 4},
            "wiki_crawler": {"enabled": True, "runs_per_hour": 4},
            "forum_crawler": {"enabled": True, "runs_per_hour": 4},
            "ml_growth": {"enabled": True, "runs_per_hour": 1},
        },
        "updated_at": None,
    }


def _get_source_breakdown() -> dict:
    """Count extra knowledge entries by source type."""
    breakdown: dict[str, int] = {}
    for fp in _iter_extra_files():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            for e in data:
                src = e.get("source", "unknown")
                # Normalize source to category
                if "wikipedia" in src or "wiki" in fp.name:
                    cat = "wikipedia"
                elif "stackexchange" in src or "serverfault" in src:
                    cat = "stackexchange"
                elif "reddit" in src:
                    cat = "reddit"
                elif "hackernews" in src:
                    cat = "hackernews"
                elif "devto" in src:
                    cat = "devto"
                elif "forum" in fp.name:
                    cat = "forums"
                elif "site_" in fp.name:
                    cat = "site_crawler"
                elif "upload" in fp.name:
                    cat = "uploads"
                elif "paste" in fp.name:
                    cat = "paste"
                elif "manual" in fp.name:
                    cat = "manual"
                else:
                    cat = "other"
                breakdown[cat] = breakdown.get(cat, 0) + 1
        except Exception:
            continue
    return breakdown


def _count_extra_knowledge() -> int:
    total = 0
    for fp in _iter_extra_files():
        try:
            total += len(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return total


def _list_extra_files() -> list[dict]:
    out: list[dict] = []
    for fp in _iter_extra_files():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            out.append({
                "name": fp.name,
                "entries": len(data),
                "size_kb": round(fp.stat().st_size / 1024, 1),
            })
        except Exception:
            continue
    return out


def _iter_extra_files():
    if EXTRA_KNOWLEDGE_DIR.exists():
        yield from sorted(EXTRA_KNOWLEDGE_DIR.glob("*.json"))


def _get_extra_domains() -> list[str]:
    domains: set[str] = set()
    for fp in _iter_extra_files():
        try:
            for e in json.loads(fp.read_text(encoding="utf-8")):
                if "domain" in e:
                    domains.add(e["domain"])
        except Exception:
            continue
    return sorted(domains)


def _parse_learning_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    prompt_lower = prompt.lower()
    prefixes = [
        "learn about", "research", "study", "find information about",
        "teach yourself about", "gather knowledge on", "crawl for",
        "learn", "fetch data about", "get information on", "find out about",
    ]
    for pfx in sorted(prefixes, key=len, reverse=True):
        if prompt_lower.startswith(pfx):
            topic = prompt[len(pfx):].strip()
            if topic:
                return topic
    return prompt if len(prompt) > 3 else ""
