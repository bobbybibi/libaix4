"""
admin_chatbot.py — Natural-language admin chatbot for libaix.

Parses free-form text commands and routes them to the appropriate admin
functions: URL crawling, topic learning, parameter tuning, status queries,
model operations.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── URL pattern ───────────────────────────────────────────────────────
_URL_RE = re.compile(r'https?://[^\s<>"\'`,;)\]]+', re.IGNORECASE)

# ── Chat history (in-memory, last 50) ────────────────────────────────
_chat_history: list[dict] = []
_HISTORY_FILE = Path("data/admin_chat_history.json")
_MAX_HISTORY = 100


def _load_history() -> list[dict]:
    global _chat_history
    if _HISTORY_FILE.exists():
        try:
            _chat_history = json.loads(
                _HISTORY_FILE.read_text(encoding="utf-8")
            )[-_MAX_HISTORY:]
        except Exception:
            _chat_history = []
    return _chat_history


def _save_history() -> None:
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_FILE.write_text(
        json.dumps(_chat_history[-_MAX_HISTORY:], indent=2, default=str),
        encoding="utf-8",
    )


def get_chat_history() -> list[dict]:
    if not _chat_history:
        _load_history()
    return _chat_history[-50:]


def _record(role: str, text: str) -> None:
    _chat_history.append({
        "role": role,
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _save_history()


# ── Command classification ────────────────────────────────────────────

_CRAWL_TRIGGERS = [
    "crawl", "scrape", "go to", "visit", "fetch", "grab", "pull from",
    "get data from", "extract from", "spider", "index",
]
_LEARN_TRIGGERS = [
    "learn about", "research", "study", "find information",
    "teach yourself", "gather knowledge", "learn",
]
_STATUS_TRIGGERS = [
    "status", "stats", "how many", "show me", "what is",
    "what's", "how is", "tell me about", "report",
    "accuracy", "entries", "domains", "knowledge",
]
_TUNE_TRIGGERS = [
    "set", "change", "update", "configure", "adjust",
    "enable", "disable", "turn on", "turn off", "toggle",
    "increase", "decrease", "raise", "lower",
]
_ACTION_TRIGGERS = [
    "retrain", "train", "optimize", "stabilize", "assess",
    "growth cycle", "run growth", "check growth",
]


def classify_intent(msg: str) -> str:
    """Return one of: crawl, learn, status, tune, action, help."""
    lower = msg.lower().strip()

    # Explicit help
    if lower in ("help", "?", "commands", "what can you do"):
        return "help"

    # Tune checks first — "enable/disable/set" are strong signals
    for t in _TUNE_TRIGGERS:
        if lower.startswith(t):
            # But not if it contains a URL
            if _URL_RE.search(msg):
                break
            return "tune"

    # URLs almost always mean crawl
    if _URL_RE.search(msg):
        return "crawl"

    for t in _CRAWL_TRIGGERS:
        if t in lower:
            # "disable" + "crawler" is tune, not crawl
            if any(w in lower for w in ["disable", "enable", "turn", "stop", "set"]):
                return "tune"
            return "crawl"
    for t in _ACTION_TRIGGERS:
        if t in lower:
            return "action"
    for t in _LEARN_TRIGGERS:
        if lower.startswith(t) or f" {t} " in f" {lower} ":
            return "learn"
    for t in _STATUS_TRIGGERS:
        if t in lower:
            return "status"

    # Default: treat as learn if it's a bare topic phrase
    if len(lower.split()) <= 5 and not lower.endswith("?"):
        return "learn"
    return "status"


# ── Parsing helpers ───────────────────────────────────────────────────

def _extract_urls(msg: str) -> list[str]:
    return _URL_RE.findall(msg)


def _extract_topic(msg: str, urls: list[str]) -> str:
    """Best-effort extraction of the topic from the message."""
    text = msg
    for u in urls:
        text = text.replace(u, "")

    # Strip command verbs
    lower = text.lower().strip()
    for prefix in sorted(_CRAWL_TRIGGERS + _LEARN_TRIGGERS, key=len, reverse=True):
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            lower = text.lower()
            break

    # Strip filler words
    for filler in [
        "about", "for", "on", "all topics", "everything about",
        "all", "topics", "information", "data", "and", "the",
        "related to", "regarding",
    ]:
        if lower.startswith(filler + " "):
            text = text[len(filler):].strip()
            lower = text.lower()

    # Strip trailing punctuation
    text = text.strip(" .,;:!?")
    return text if len(text) > 2 else ""


def _extract_keywords(msg: str) -> list[str]:
    """Look for explicit keyword hints like 'keywords: x, y, z'."""
    m = re.search(r'keywords?\s*[:=]\s*(.+?)(?:\.|$)', msg, re.I)
    if m:
        return [k.strip() for k in m.group(1).split(",") if k.strip()]
    return []


def _extract_number(msg: str, param: str) -> float | None:
    """Extract a numeric value after a parameter name."""
    m = re.search(rf'{param}\s*(?:to|=|:)?\s*([\d.]+)', msg, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


# ── Command handlers ──────────────────────────────────────────────────

def handle_crawl(msg: str) -> dict[str, Any]:
    """Handle URL crawling / scraping commands."""
    urls = _extract_urls(msg)
    topic = _extract_topic(msg, urls)
    keywords = _extract_keywords(msg)

    if not urls and not topic:
        return {
            "reply": (
                "I need a URL or topic to crawl. Try:\n"
                "• `crawl https://example.com for networking`\n"
                "• `scrape https://reddit.com/r/netsec for firewall rules`\n"
                "• `go to https://docs.example.com and get wifi security info`"
            ),
            "action": None,
        }

    actions: list[dict] = []

    # Site crawl for each URL
    for url in urls:
        actions.append({
            "type": "site_crawl",
            "url": url,
            "topic": topic or "general networking",
            "keywords": keywords,
            "max_pages": 30,
            "max_depth": 2,
        })

    # If there's a topic but no URL, trigger multi-source learning
    if topic and not urls:
        actions.append({
            "type": "learn",
            "topic": topic,
            "keywords": keywords,
        })

    # If URL + topic, also queue a forum search for the topic
    if topic and urls:
        actions.append({
            "type": "forum_crawl",
            "topic": topic,
            "keywords": keywords,
        })

    return {
        "reply": _build_crawl_reply(urls, topic, keywords),
        "actions": actions,
    }


def _build_crawl_reply(urls: list[str], topic: str, keywords: list[str]) -> str:
    parts = ["Got it! Here's what I'll do:\n"]
    for i, url in enumerate(urls, 1):
        parts.append(f"{i}. Crawl **{url}**" + (f" for `{topic}`" if topic else ""))
    if topic and urls:
        parts.append(f"{len(urls)+1}. Also search forums (Reddit, StackExchange, HN) for `{topic}`")
    if topic and not urls:
        parts.append(f"1. Search all sources (Wikipedia, Reddit, StackExchange, HN, DEV.to) for `{topic}`")
    if keywords:
        parts.append(f"\nKeywords: {', '.join(keywords)}")
    parts.append("\nExecuting now...")
    return "\n".join(parts)


def handle_learn(msg: str) -> dict[str, Any]:
    """Handle learning / research commands."""
    topic = _extract_topic(msg, [])
    keywords = _extract_keywords(msg)

    if not topic:
        return {
            "reply": (
                "What should I learn about? Try:\n"
                "• `learn about OSPF routing`\n"
                "• `research zero trust architecture`\n"
                "• `study WPA3 encryption`"
            ),
            "actions": [],
        }

    return {
        "reply": (
            f"Learning about **{topic}**...\n"
            f"Searching: Wikipedia, Reddit, StackExchange, Hacker News, DEV.to\n"
            + (f"Keywords: {', '.join(keywords)}\n" if keywords else "")
            + "Executing now..."
        ),
        "actions": [{"type": "learn", "topic": topic, "keywords": keywords}],
    }


def handle_status(msg: str) -> dict[str, Any]:
    """Handle status / stats queries. Returns a request to fetch live data."""
    lower = msg.lower()

    queries = []
    if any(w in lower for w in ["accuracy", "model", "performance"]):
        queries.append("ml_stats")
    if any(w in lower for w in ["entries", "knowledge", "how many", "total"]):
        queries.append("stats")
    if any(w in lower for w in ["domain", "topics"]):
        queries.append("stats")
    if any(w in lower for w in ["learn", "velocity", "rate", "growth"]):
        queries.append("learning")
    if any(w in lower for w in ["cron", "schedule", "workflow", "automation"]):
        queries.append("cron")
    if any(w in lower for w in ["crawl", "source", "where"]):
        queries.append("sources")

    if not queries:
        queries = ["stats", "ml_stats", "learning"]

    return {
        "reply": "Let me check that for you...",
        "queries": queries,
    }


def handle_tune(msg: str) -> dict[str, Any]:
    """Handle parameter tuning commands."""
    lower = msg.lower()
    changes: list[dict] = []

    # Toggle auto features
    for feature in ["auto_optimize", "auto_stabilize", "auto_grow"]:
        readable = feature.replace("_", "-")
        if readable in lower or feature in lower:
            if any(w in lower for w in ["enable", "turn on", "activate"]):
                changes.append({"type": "ml_config", "key": feature, "value": True})
            elif any(w in lower for w in ["disable", "turn off", "deactivate"]):
                changes.append({"type": "ml_config", "key": feature, "value": False})

    # Numeric thresholds
    for param, key in [
        ("accuracy threshold", "min_accuracy_threshold"),
        ("min accuracy", "min_accuracy_threshold"),
        ("confidence threshold", "confidence_threshold"),
        ("confidence", "confidence_threshold"),
        ("target accuracy", "growth_targets.target_accuracy"),
    ]:
        val = _extract_number(msg, param)
        if val is not None:
            # Normalize: if user says "95" they mean 0.95
            if val > 1.0:
                val = val / 100.0
            changes.append({"type": "ml_config", "key": key, "value": val})

    # Cron controls
    for workflow in ["auto_train", "wiki_crawler", "forum_crawler", "ml_growth"]:
        readable = workflow.replace("_", " ")
        if readable in lower or workflow in lower:
            if any(w in lower for w in ["enable", "turn on"]):
                changes.append({"type": "cron", "workflow": workflow, "key": "enabled", "value": True})
            elif any(w in lower for w in ["disable", "turn off", "stop"]):
                changes.append({"type": "cron", "workflow": workflow, "key": "enabled", "value": False})
            rph = _extract_number(msg, readable)
            if rph is None:
                rph = _extract_number(msg, "per hour")
            if rph is not None:
                changes.append({"type": "cron", "workflow": workflow, "key": "runs_per_hour", "value": int(rph)})

    if not changes:
        return {
            "reply": (
                "I can tune these parameters:\n"
                "• `set accuracy threshold to 0.95`\n"
                "• `enable auto-optimize` / `disable auto-stabilize`\n"
                "• `set forum crawler to 2 per hour`\n"
                "• `disable wiki crawler`\n"
                "• `set confidence threshold to 0.7`\n"
                "• `turn off auto train`"
            ),
            "actions": [],
        }

    return {
        "reply": f"Applying {len(changes)} change(s)...",
        "actions": changes,
    }


def handle_action(msg: str) -> dict[str, Any]:
    """Handle model action commands (retrain, optimize, etc.)."""
    lower = msg.lower()

    if "retrain" in lower or ("train" in lower and "model" in lower):
        return {
            "reply": "Retraining the model now... this may take a minute.",
            "actions": [{"type": "retrain"}],
        }
    if "optimize" in lower:
        return {
            "reply": "Starting model optimization with multiple hyperparameter configs... this may take several minutes.",
            "actions": [{"type": "optimize"}],
        }
    if "stabilize" in lower:
        return {
            "reply": "Checking model stability...",
            "actions": [{"type": "stabilize"}],
        }
    if "growth cycle" in lower or "run growth" in lower:
        return {
            "reply": "Running full growth cycle (assess → stabilize → optimize)... this may take several minutes.",
            "actions": [{"type": "growth_cycle"}],
        }
    if "assess" in lower or "check" in lower:
        return {
            "reply": "Assessing model quality...",
            "actions": [{"type": "assess"}],
        }

    return {
        "reply": (
            "Available actions:\n"
            "• `retrain the model`\n"
            "• `optimize the model`\n"
            "• `stabilize the model`\n"
            "• `assess the model`\n"
            "• `run a growth cycle`"
        ),
        "actions": [],
    }


# ── Main entry point ─────────────────────────────────────────────────

def process_message(msg: str) -> dict[str, Any]:
    """
    Parse a user message and return a response dict:
      {
        "reply": str,          # Text reply to show
        "intent": str,         # classified intent
        "actions": [...],      # actions to execute (optional)
        "queries": [...],      # data to fetch (optional, for status)
      }
    """
    msg = msg.strip()
    if not msg:
        return {"reply": "Send me a message! I can crawl URLs, learn topics, tune parameters, and more.", "intent": "help"}

    _record("user", msg)

    # Help
    lower = msg.lower()
    if lower in ("help", "?", "commands", "what can you do"):
        reply = (
            "Here's what I can do:\n\n"
            "**🌐 Crawl URLs**\n"
            "• `crawl https://example.com for wifi security`\n"
            "• `go to https://docs.site.com and scrape all networking topics`\n"
            "• `scrape https://reddit.com/r/netsec for firewall rules`\n\n"
            "**🎓 Learn Topics**\n"
            "• `learn about OSPF routing`\n"
            "• `research zero trust architecture`\n\n"
            "**📊 Check Status**\n"
            "• `how many entries do we have?`\n"
            "• `what's the model accuracy?`\n"
            "• `show me learning stats`\n\n"
            "**⚙️ Tune Parameters**\n"
            "• `set accuracy threshold to 0.95`\n"
            "• `enable auto-optimize`\n"
            "• `set forum crawler to 2 per hour`\n\n"
            "**🔄 Actions**\n"
            "• `retrain the model`\n"
            "• `optimize the model`\n"
            "• `run a growth cycle`"
        )
        _record("assistant", reply)
        return {"reply": reply, "intent": "help"}

    intent = classify_intent(msg)
    handler = {
        "crawl": handle_crawl,
        "learn": handle_learn,
        "status": handle_status,
        "tune": handle_tune,
        "action": handle_action,
    }.get(intent, handle_status)

    result = handler(msg)
    result["intent"] = intent
    _record("assistant", result["reply"])
    return result
