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
from admin_chatbot import get_chat_history, process_message

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
            domain_breakdown = _entries_domain_breakdown(entries)
            topics = _entries_topic_summary(entries)
            return jsonify({
                "status": "success",
                "file": filename,
                "entries_extracted": len(entries),
                "preview": preview,
                "samples": entries[:5],
                "saved_to": str(save_path),
                "domain_breakdown": domain_breakdown,
                "topics": topics,
                "message": f"Extracted {len(entries)} entries from {filename}. Saved and file deleted.",
            })
        return jsonify({
            "status": "warning",
            "file": filename,
            "entries_extracted": 0,
            "preview": preview,
            "message": "No Q&A entries could be extracted. The text may lack definitions (\"X is a Y\") or structured content. Try pasting as plain text instead.",
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

    text = data["text"]
    text_len = len(text)
    entries = process_pasted_text(text, data.get("domain", ""))
    if entries:
        save_path = _save_knowledge(entries, "paste")
        domain_breakdown = _entries_domain_breakdown(entries)
        topics = _entries_topic_summary(entries)
        return jsonify({
            "status": "success",
            "entries_extracted": len(entries),
            "text_length": text_len,
            "samples": entries[:5],
            "saved_to": str(save_path),
            "domain_breakdown": domain_breakdown,
            "topics": topics,
            "message": f"Extracted {len(entries)} entries from {text_len} characters of pasted text.",
        })
    return jsonify({
        "status": "warning",
        "entries_extracted": 0,
        "text_length": text_len,
        "message": (
            "No Q&A entries could be extracted. Tips:\n"
            "• Include definitions like 'X is a Y' or 'X provides Y'\n"
            "• Use sentences longer than 30 characters\n"
            "• Paste structured content (articles, docs, wiki text)"
        ),
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


# ── Learned Topics ────────────────────────────────────────────────────

@admin_bp.route("/learned")
@login_required
def learned_topics():
    """Return a structured list of all learned knowledge grouped by source/topic."""
    files_info: list[dict] = []
    topic_map: dict[str, dict] = {}

    for fp in _iter_extra_files():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if not data:
                continue

            # Parse source type and topic from filename
            name = fp.name
            source_type, topic = _parse_knowledge_filename(name)
            domains: dict[str, int] = {}
            for e in data:
                d = e.get("domain", "general")
                domains[d] = domains.get(d, 0) + 1

            ts_str = _extract_timestamp_from_filename(name)
            stat = fp.stat()

            file_info = {
                "filename": name,
                "source_type": source_type,
                "topic": topic,
                "entries": len(data),
                "domains": domains,
                "top_domain": max(domains, key=domains.get) if domains else "general",
                "size_kb": round(stat.st_size / 1024, 1),
                "timestamp": ts_str,
                "sample_questions": [e["question"] for e in data[:3]],
            }
            files_info.append(file_info)

            # Aggregate by topic
            if topic not in topic_map:
                topic_map[topic] = {
                    "topic": topic,
                    "total_entries": 0,
                    "sources": [],
                    "domains": {},
                    "file_count": 0,
                    "latest": ts_str,
                }
            tm = topic_map[topic]
            tm["total_entries"] += len(data)
            tm["file_count"] += 1
            if source_type not in tm["sources"]:
                tm["sources"].append(source_type)
            for d, c in domains.items():
                tm["domains"][d] = tm["domains"].get(d, 0) + c
            if ts_str and (not tm["latest"] or ts_str > tm["latest"]):
                tm["latest"] = ts_str

        except Exception:
            continue

    # Sort: topics by total entries descending
    topics_sorted = sorted(topic_map.values(), key=lambda t: -t["total_entries"])

    return jsonify({
        "files": sorted(files_info, key=lambda f: f.get("timestamp", ""), reverse=True),
        "topics": topics_sorted,
        "summary": {
            "total_files": len(files_info),
            "total_entries": sum(f["entries"] for f in files_info),
            "total_topics": len(topic_map),
            "source_types": sorted(set(f["source_type"] for f in files_info)),
        },
    })


# ── Local Scheduler Status ───────────────────────────────────────────

@admin_bp.route("/scheduler/status")
@login_required
def scheduler_status():
    """Return local scheduler status if running."""
    try:
        from local_scheduler import get_scheduler_status
        return jsonify(get_scheduler_status())
    except ImportError:
        return jsonify({
            "running": False,
            "mode": "github_actions",
            "message": "Local scheduler not loaded. Using GitHub Actions for automation.",
        })


@admin_bp.route("/scheduler/toggle", methods=["POST"])
@login_required
def scheduler_toggle():
    """Start or stop the local scheduler."""
    try:
        from local_scheduler import start_scheduler, stop_scheduler, get_scheduler_status
        data = request.get_json() or {}
        if data.get("action") == "stop":
            stop_scheduler()
        else:
            start_scheduler()
        return jsonify(get_scheduler_status())
    except ImportError:
        return jsonify({"error": "Local scheduler module not available"}), 500


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


# ── Admin Chatbot ─────────────────────────────────────────────────────

@admin_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    """Process a chatbot message and execute any actions."""
    data = request.get_json()
    if not data or not data.get("message"):
        return jsonify({"error": "Message required"}), 400

    result = process_message(data["message"])
    actions = result.get("actions", [])
    queries = result.get("queries", [])
    action_results: list[dict] = []

    # Execute actions
    for act in actions:
        atype = act.get("type", "")
        try:
            if atype == "site_crawl":
                r = add_site_job(
                    url=act["url"],
                    topic=act.get("topic", "general"),
                    keywords=act.get("keywords", []),
                    max_pages=min(int(act.get("max_pages", 30)), 50),
                    max_depth=min(int(act.get("max_depth", 2)), 3),
                )
                action_results.append({"type": "site_crawl", "url": act["url"], "result": r})
            elif atype == "learn":
                wiki_r = crawl_single_topic(act["topic"], act.get("keywords", []), max_articles=10)
                forum_r = crawl_single_forum_topic(
                    act["topic"], act.get("keywords", []), max_per_source=10,
                    sources=["stackexchange", "reddit", "hackernews", "devto"],
                )
                total = (wiki_r.get("entries", 0) if wiki_r.get("status") == "success" else 0) + \
                        (forum_r.get("entries", 0) if forum_r.get("status") == "success" else 0)
                action_results.append({
                    "type": "learn", "topic": act["topic"],
                    "total_entries": total,
                    "wikipedia": wiki_r.get("entries", 0),
                    "forums": forum_r.get("entries", 0),
                })
            elif atype == "forum_crawl":
                r = crawl_single_forum_topic(
                    act["topic"], act.get("keywords", []), max_per_source=10,
                    sources=["stackexchange", "reddit", "hackernews", "devto"],
                )
                action_results.append({"type": "forum_crawl", "topic": act["topic"], "result": r})
            elif atype == "retrain":
                from train_knowledge import train as _train
                model, bow, answer_map = _train(
                    activation="tanh", optimizer="adam",
                    lr=0.01, epochs=5000, hidden=256,
                    augment=True, verbose=False,
                )
                action_results.append({
                    "type": "retrain", "entries": len(answer_map),
                    "vocab": bow.vocab_size,
                })
            elif atype == "optimize":
                r = optimize_model()
                action_results.append({"type": "optimize", "result": r})
            elif atype == "stabilize":
                r = stabilize_model()
                action_results.append({"type": "stabilize", "result": r})
            elif atype == "assess":
                r = assess_model()
                action_results.append({"type": "assess", "result": r})
            elif atype == "growth_cycle":
                r = run_growth_cycle()
                action_results.append({"type": "growth_cycle", "result": r})
            elif atype == "ml_config":
                config = load_engine_config()
                key = act["key"]
                if "." in key:
                    parts = key.split(".", 1)
                    config.setdefault(parts[0], {})[parts[1]] = act["value"]
                else:
                    config[key] = act["value"]
                save_engine_config(config)
                action_results.append({"type": "ml_config", "key": key, "value": act["value"]})
            elif atype == "cron":
                cron_cfg = _load_cron_config()
                wf = act["workflow"]
                cron_cfg["workflows"].setdefault(wf, {})
                cron_cfg["workflows"][wf][act["key"]] = act["value"]
                cron_cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_cron_config(cron_cfg)
                action_results.append({"type": "cron", "workflow": wf, "key": act["key"], "value": act["value"]})
        except Exception as exc:
            action_results.append({"type": atype, "error": str(exc)})

    # Fetch queried data for status responses
    query_data: dict = {}
    for q in queries:
        try:
            if q == "stats":
                extra = _count_extra_knowledge()
                query_data["stats"] = {
                    "builtin_entries": len(KNOWLEDGE),
                    "extra_entries": extra,
                    "total_entries": len(KNOWLEDGE) + extra,
                    "domains": get_domains() + _get_extra_domains(),
                    "source_breakdown": _get_source_breakdown(),
                }
            elif q == "ml_stats":
                query_data["ml_stats"] = get_engine_stats()
            elif q == "learning":
                query_data["learning"] = get_learning_stats()
            elif q == "cron":
                query_data["cron"] = _load_cron_config()
            elif q == "sources":
                query_data["sources"] = _get_source_breakdown()
        except Exception:
            pass

    # Build follow-up reply with results
    followup = _build_chat_followup(result, action_results, query_data)

    return jsonify({
        "reply": result["reply"],
        "followup": followup,
        "intent": result.get("intent"),
        "action_results": action_results,
        "query_data": query_data,
    })


@admin_bp.route("/chat/history")
@login_required
def chat_history():
    return jsonify({"history": get_chat_history()})


def _build_chat_followup(result: dict, actions: list[dict], query_data: dict) -> str:
    """Build a human-readable followup from action/query results."""
    parts: list[str] = []

    for ar in actions:
        atype = ar.get("type", "")
        if ar.get("error"):
            parts.append(f"❌ {atype}: {ar['error']}")
            continue

        if atype == "site_crawl":
            r = ar.get("result", {})
            entries = r.get("entries", 0)
            stats = r.get("stats", {})
            parts.append(
                f"🌐 Crawled **{ar.get('url', '?')}**: "
                f"{entries} entries extracted "
                f"({stats.get('pages_crawled', 0)} pages crawled, "
                f"{stats.get('pages_relevant', 0)} relevant)"
            )
        elif atype == "learn":
            parts.append(
                f"🎓 Learned about **{ar.get('topic', '?')}**: "
                f"{ar.get('total_entries', 0)} total entries "
                f"(Wikipedia: {ar.get('wikipedia', 0)}, "
                f"Forums: {ar.get('forums', 0)})"
            )
        elif atype == "forum_crawl":
            r = ar.get("result", {})
            parts.append(
                f"💬 Forum search for **{ar.get('topic', '?')}**: "
                f"{r.get('entries', 0)} entries"
            )
        elif atype == "retrain":
            parts.append(
                f"🔄 Model retrained: {ar.get('entries', 0)} answers, "
                f"{ar.get('vocab', 0)} vocab words"
            )
        elif atype == "optimize":
            r = ar.get("result", {})
            parts.append(
                f"⚡ Optimization: {r.get('status', 'done')} — "
                f"best accuracy {(r.get('best_accuracy', 0)*100):.1f}%"
            )
        elif atype == "stabilize":
            r = ar.get("result", {})
            parts.append(f"🛡️ Stability: {r.get('message', r.get('status', 'done'))}")
        elif atype == "assess":
            r = ar.get("result", {})
            parts.append(
                f"📊 Assessment: accuracy {(r.get('overall_accuracy', 0)*100):.1f}%, "
                f"confidence {(r.get('avg_confidence', 0)*100):.1f}%"
            )
        elif atype == "growth_cycle":
            r = ar.get("result", {})
            parts.append(
                f"🔄 Growth cycle: final accuracy {(r.get('final_accuracy', 0)*100):.1f}%, "
                f"improvement {(r.get('improvement', 0)*100):+.2f}%"
            )
        elif atype == "ml_config":
            parts.append(f"⚙️ Set `{ar.get('key')}` = `{ar.get('value')}`")
        elif atype == "cron":
            parts.append(f"⏱️ Cron `{ar.get('workflow')}`.`{ar.get('key')}` = `{ar.get('value')}`")

    # Status query results
    if "stats" in query_data:
        s = query_data["stats"]
        parts.append(
            f"\n📊 **Knowledge Base**\n"
            f"• Total entries: **{s['total_entries']}** "
            f"(built-in: {s['builtin_entries']}, learned: {s['extra_entries']})\n"
            f"• Domains: **{len(s['domains'])}** — {', '.join(s['domains'][:10])}"
        )
        sb = s.get("source_breakdown", {})
        if sb:
            parts.append("• Sources: " + ", ".join(f"{k}: {v}" for k, v in sorted(sb.items(), key=lambda x: -x[1])))

    if "ml_stats" in query_data:
        ml = query_data["ml_stats"]
        cfg = ml.get("config", {})
        trend = ml.get("accuracy_trend", [])
        last_acc = f"{trend[-1]['accuracy']*100:.1f}%" if trend else "unknown"
        parts.append(
            f"\n🧠 **Model**\n"
            f"• Accuracy: **{last_acc}**\n"
            f"• Growth cycles: {ml.get('cycle_count', 0)}\n"
            f"• Auto-optimize: {'✅' if cfg.get('auto_optimize') else '❌'} | "
            f"Auto-stabilize: {'✅' if cfg.get('auto_stabilize') else '❌'} | "
            f"Auto-grow: {'✅' if cfg.get('auto_grow') else '❌'}"
        )

    if "learning" in query_data:
        ls = query_data["learning"]
        parts.append(
            f"\n📈 **Learning**\n"
            f"• Velocity: **{ls.get('learning_velocity', 0):.1f}** entries/hr\n"
            f"• Last 24h: {ls.get('entries_last_24h', 0)} entries\n"
            f"• Total events: {ls.get('total_events', 0)}"
        )

    if "cron" in query_data:
        c = query_data["cron"]
        wfs = c.get("workflows", {})
        lines = []
        for k, v in wfs.items():
            status = "✅" if v.get("enabled") else "❌"
            lines.append(f"  {status} {k}: {v.get('runs_per_hour', 0)}/hr")
        parts.append(f"\n⏱️ **Cron Jobs**\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else ""


# ── 
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


def _entries_domain_breakdown(entries: list[dict]) -> dict[str, int]:
    """Count entries per domain."""
    breakdown: dict[str, int] = {}
    for e in entries:
        d = e.get("domain", "general")
        breakdown[d] = breakdown.get(d, 0) + 1
    return breakdown


def _entries_topic_summary(entries: list[dict]) -> list[str]:
    """Extract unique topics/subjects from Q&A entries."""
    topics: set[str] = set()
    for e in entries:
        q = e.get("question", "")
        # Extract subject from common question patterns
        for prefix in ("What is ", "What are ", "Tell me about ", "What does "):
            if q.startswith(prefix):
                subj = q[len(prefix):].rstrip("?").strip()
                if 3 <= len(subj) <= 80:
                    topics.add(subj)
                break
        # Also use domain as a topic
        d = e.get("domain", "")
        if d and d != "general":
            topics.add(d.replace("_", " ").title())
    return sorted(topics)[:50]


def _parse_knowledge_filename(name: str) -> tuple[str, str]:
    """Parse source type and topic from a knowledge filename."""
    import re
    name_no_ext = name.rsplit(".", 1)[0]

    if name_no_ext.startswith("upload_"):
        return "upload", name_no_ext.replace("upload_", "").split("_2")[0] or "uploaded file"
    if name_no_ext.startswith("paste_"):
        return "paste", "pasted text"
    if name_no_ext.startswith("site_"):
        return "site_crawler", name_no_ext.replace("site_", "").split("_2")[0] or "crawled site"
    if name_no_ext.startswith("forum_"):
        return "forum_crawler", name_no_ext.replace("forum_", "").split("_2")[0] or "forum"

    # Try to extract topic from middle of filename
    parts = re.split(r"[_\-]", name_no_ext)
    # Filter out timestamps and generic parts
    meaningful = [p for p in parts if len(p) > 2 and not p.isdigit()]
    topic = " ".join(meaningful[:3]) if meaningful else name_no_ext
    return "other", topic


def _extract_timestamp_from_filename(name: str) -> str:
    """Extract a timestamp string from filename like 'paste_20260325_070122.json'."""
    import re
    m = re.search(r"(\d{8})[_-]?(\d{6})?", name)
    if m:
        date_str = m.group(1)
        time_str = m.group(2) or "000000"
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
    return ""


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
