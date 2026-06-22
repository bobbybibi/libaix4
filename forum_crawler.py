"""
forum_crawler.py — Crawl free public forums, Q&A sites, and tech APIs for knowledge.

Sources (all free, no auth needed):
  • StackExchange API — ServerFault, SuperUser, NetworkEngineering, Security, Unix
  • Reddit public JSON API — r/networking, r/netsec, r/sysadmin, r/homelab, etc.
  • Hacker News Algolia API — top tech discussions
  • DEV.to API — developer articles on networking/security topics
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

from file_processor import classify_domain, dedupe_new_entries, generate_qa_from_text

USER_AGENT = (
    "libaix-crawler/1.0 (educational neural network project; "
    "github.com/lindapot-art/libaix)"
)
EXTRA_KNOWLEDGE_DIR = Path("data/extra_knowledge")
FORUM_CONFIG_PATH = Path("data/forum_config.json")
LEARNING_LOG_PATH = Path("data/learning_log.json")
CRAWL_DELAY = 2.0  # be polite to public APIs

_STRIP_HTML = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return unescape(_STRIP_HTML.sub("", text)).strip()


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _http_get_json(url: str, timeout: int = 30) -> dict:
    return json.loads(_http_get(url, timeout))


# ── StackExchange API (free, no key needed for 300 req/day) ──────────

STACK_SITES = {
    "serverfault": "serverfault.com",
    "superuser": "superuser.com",
    "networkengineering": "networkengineering.stackexchange.com",
    "security": "security.stackexchange.com",
    "unix": "unix.stackexchange.com",
}

SE_API = "https://api.stackexchange.com/2.3"


def crawl_stackexchange(
    query: str,
    site: str = "serverfault",
    max_questions: int = 15,
) -> list[dict[str, str]]:
    """Search StackExchange for answered questions on a topic."""
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    site_domain = STACK_SITES.get(site, site)

    # Search for questions
    params = urllib.parse.urlencode({
        "order": "desc",
        "sort": "relevance",
        "q": query,
        "site": site_domain,
        "filter": "withbody",
        "accepted": "True",
        "pagesize": str(min(max_questions, 30)),
    })
    url = f"{SE_API}/search/advanced?{params}"

    try:
        data = _http_get_json(url)
    except Exception:
        return entries

    time.sleep(CRAWL_DELAY)

    for item in data.get("items", [])[:max_questions]:
        title = _strip_html(item.get("title", ""))
        body = _strip_html(item.get("body", ""))
        if not title or not body:
            continue

        ql = title.lower()
        if ql in seen:
            continue
        seen.add(ql)

        # Question title → Question, accepted answer body → Answer
        domain = classify_domain(f"{title} {body}")
        answer_text = _truncate(body, 500)

        entries.append({
            "question": title if title.endswith("?") else f"{title}?",
            "answer": answer_text,
            "domain": domain,
            "source": f"stackexchange:{site}:{item.get('question_id', '')}",
        })

        # Also extract sub-QA from the body
        sub_entries = generate_qa_from_text(body)
        for se in sub_entries:
            sel = se["question"].lower()
            if sel not in seen:
                seen.add(sel)
                se["source"] = f"stackexchange:{site}:{item.get('question_id', '')}"
                entries.append(se)

    return entries


# ── Reddit public JSON API (no auth needed for public subreddits) ────

REDDIT_SUBREDDITS = [
    "networking", "netsec", "sysadmin", "homelab",
    "wifi", "cybersecurity", "ccna", "ITCareerQuestions",
    "AskNetsec", "CompTIA", "hacking", "linuxadmin",
]


def crawl_reddit(
    query: str,
    subreddit: str = "networking",
    max_posts: int = 15,
) -> list[dict[str, str]]:
    """Search a subreddit for informational posts."""
    entries: list[dict[str, str]] = []
    seen: set[str] = set()

    params = urllib.parse.urlencode({
        "q": query,
        "restrict_sr": "1",
        "sort": "relevance",
        "t": "all",
        "limit": str(min(max_posts, 25)),
    })
    url = f"https://www.reddit.com/r/{subreddit}/search.json?{params}"

    try:
        data = _http_get_json(url)
    except Exception:
        return entries

    time.sleep(CRAWL_DELAY)

    for child in data.get("data", {}).get("children", [])[:max_posts]:
        post = child.get("data", {})
        title = post.get("title", "").strip()
        selftext = post.get("selftext", "").strip()

        if not title or post.get("over_18") or post.get("quarantine"):
            continue
        if len(selftext) < 50:
            continue

        ql = title.lower()
        if ql in seen:
            continue
        seen.add(ql)

        domain = classify_domain(f"{title} {selftext}")
        answer_text = _truncate(selftext, 500)

        question = title if title.endswith("?") else f"What is known about {title}?"
        entries.append({
            "question": question,
            "answer": answer_text,
            "domain": domain,
            "source": f"reddit:r/{subreddit}:{post.get('id', '')}",
        })

        sub_entries = generate_qa_from_text(selftext)
        for se in sub_entries:
            sel = se["question"].lower()
            if sel not in seen:
                seen.add(sel)
                se["source"] = f"reddit:r/{subreddit}"
                entries.append(se)

    return entries


# ── Combined forum crawl ─────────────────────────────────────────────

# ── Hacker News (Algolia API — free, no auth) ────────────────────────

HN_API = "https://hn.algolia.com/api/v1"


def crawl_hackernews(
    query: str,
    max_posts: int = 15,
) -> list[dict[str, str]]:
    """Search Hacker News discussions for informational content."""
    entries: list[dict[str, str]] = []
    seen: set[str] = set()

    params = urllib.parse.urlencode({
        "query": query,
        "tags": "story",
        "hitsPerPage": str(min(max_posts, 30)),
    })
    url = f"{HN_API}/search?{params}"

    try:
        data = _http_get_json(url)
    except Exception:
        return entries

    time.sleep(CRAWL_DELAY)

    for hit in data.get("hits", [])[:max_posts]:
        title = hit.get("title", "").strip()
        story_text = _strip_html(hit.get("story_text") or "")
        # Use comment text if no story body
        if not story_text or len(story_text) < 50:
            # Fetch top comments for context
            try:
                item_url = f"{HN_API}/items/{hit.get('objectID', '')}"
                item_data = _http_get_json(item_url)
                children = item_data.get("children", [])
                comment_texts = []
                for child in children[:5]:
                    txt = _strip_html(child.get("text") or "")
                    if len(txt) > 30:
                        comment_texts.append(txt)
                story_text = " ".join(comment_texts)
                time.sleep(1.0)
            except Exception:
                pass

        if not title or len(story_text) < 50:
            continue

        ql = title.lower()
        if ql in seen:
            continue
        seen.add(ql)

        domain = classify_domain(f"{title} {story_text}")
        answer_text = _truncate(story_text, 500)
        question = title if title.endswith("?") else f"What is discussed about {title}?"

        entries.append({
            "question": question,
            "answer": answer_text,
            "domain": domain,
            "source": f"hackernews:{hit.get('objectID', '')}",
        })

        sub_entries = generate_qa_from_text(story_text)
        for se in sub_entries:
            sel = se["question"].lower()
            if sel not in seen:
                seen.add(sel)
                se["source"] = "hackernews"
                entries.append(se)

    return entries


# ── DEV.to API (free, no auth) ───────────────────────────────────────

DEVTO_API = "https://dev.to/api"


def crawl_devto(
    query: str,
    max_articles: int = 10,
) -> list[dict[str, str]]:
    """Search DEV.to for tech articles on a topic."""
    entries: list[dict[str, str]] = []
    seen: set[str] = set()

    params = urllib.parse.urlencode({
        "per_page": str(min(max_articles, 25)),
    })
    # Search by tag works well for tech topics
    tag = re.sub(r"[^a-z0-9]", "", query.lower().replace(" ", ""))
    url = f"{DEVTO_API}/articles?tag={tag}&{params}"

    try:
        articles = json.loads(_http_get(url))
    except Exception:
        # Fallback: search by query
        try:
            params2 = urllib.parse.urlencode({
                "per_page": str(min(max_articles, 25)),
                "tag": query.split()[0].lower() if query.split() else query,
            })
            url2 = f"{DEVTO_API}/articles?{params2}"
            articles = json.loads(_http_get(url2))
        except Exception:
            return entries

    time.sleep(CRAWL_DELAY)

    for article in (articles if isinstance(articles, list) else [])[:max_articles]:
        title = article.get("title", "").strip()
        description = article.get("description", "").strip()
        body_md = article.get("body_markdown", "")

        if not title:
            continue

        ql = title.lower()
        if ql in seen:
            continue
        seen.add(ql)

        # Use description or first part of body
        content = description or _strip_html(body_md)[:500] if body_md else ""
        if len(content) < 30:
            continue

        domain = classify_domain(f"{title} {content}")
        answer_text = _truncate(content, 500)
        question = title if title.endswith("?") else f"What does this article explain: {title}?"

        entries.append({
            "question": question,
            "answer": answer_text,
            "domain": domain,
            "source": f"devto:{article.get('id', '')}",
        })

    return entries


# ── Learning log (tracks all crawl activities) ───────────────────────

def log_learning_event(
    source: str,
    topic: str,
    entries_count: int,
    details: dict | None = None,
) -> None:
    """Record a learning event for live stats tracking."""
    log = _load_learning_log()
    log.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "topic": topic,
        "entries": entries_count,
        **(details or {}),
    })
    # Keep last 500 events
    log = log[-500:]
    LEARNING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNING_LOG_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _load_learning_log() -> list[dict]:
    if LEARNING_LOG_PATH.exists():
        try:
            return json.loads(LEARNING_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def get_learning_stats() -> dict:
    """Get comprehensive learning statistics for admin dashboard."""
    log = _load_learning_log()

    # Per-source stats
    source_stats: dict[str, dict] = {}
    total_entries = 0
    for event in log:
        src = event.get("source", "unknown")
        if src not in source_stats:
            source_stats[src] = {"total_entries": 0, "crawl_count": 0, "last_crawl": None}
        source_stats[src]["total_entries"] += event.get("entries", 0)
        source_stats[src]["crawl_count"] += 1
        source_stats[src]["last_crawl"] = event.get("timestamp")
        total_entries += event.get("entries", 0)

    # Per-topic stats
    topic_stats: dict[str, dict] = {}
    for event in log:
        topic = event.get("topic", "unknown")
        if topic not in topic_stats:
            topic_stats[topic] = {"total_entries": 0, "crawl_count": 0, "sources": set()}
        topic_stats[topic]["total_entries"] += event.get("entries", 0)
        topic_stats[topic]["crawl_count"] += 1
        topic_stats[topic]["sources"].add(event.get("source", "unknown"))
    # Convert sets to lists for JSON
    for v in topic_stats.values():
        v["sources"] = sorted(v["sources"])

    # Recent activity (last 20)
    recent = log[-20:]

    # Learning velocity (entries per hour over last 24h)
    now = datetime.now(timezone.utc)
    last_24h = [e for e in log if _parse_ts(e.get("timestamp", "")) and
                (now - _parse_ts(e["timestamp"])).total_seconds() < 86400]
    entries_24h = sum(e.get("entries", 0) for e in last_24h)
    velocity = entries_24h / 24.0 if last_24h else 0

    return {
        "total_entries_learned": total_entries,
        "total_events": len(log),
        "source_stats": source_stats,
        "topic_stats": topic_stats,
        "recent_activity": recent,
        "learning_velocity": round(velocity, 1),
        "entries_last_24h": entries_24h,
        "events_last_24h": len(last_24h),
    }


def _parse_ts(ts_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


# ── Combined forum crawl ─────────────────────────────────────────────

def crawl_forums(
    topic: str,
    keywords: list[str] | None = None,
    max_per_source: int = 10,
    sources: list[str] | None = None,
) -> list[dict[str, str]]:
    """Crawl multiple forum sources for a topic."""
    keywords = keywords or []
    all_entries: list[dict[str, str]] = []
    seen: set[str] = set()

    search_queries = [topic] + keywords[:3]
    enabled_sources = sources or ["stackexchange", "reddit", "hackernews", "devto"]

    for query in search_queries:
        # StackExchange sites
        if "stackexchange" in enabled_sources:
            for site in ["serverfault", "networkengineering", "security"]:
                try:
                    results = crawl_stackexchange(query, site, max_per_source)
                    for e in results:
                        ql = e["question"].lower()
                        if ql not in seen:
                            seen.add(ql)
                            all_entries.append(e)
                except Exception:
                    continue
                time.sleep(CRAWL_DELAY)

        # Reddit
        if "reddit" in enabled_sources:
            for sub in ["networking", "netsec", "sysadmin", "cybersecurity", "homelab"]:
                try:
                    results = crawl_reddit(query, sub, max_per_source)
                    for e in results:
                        ql = e["question"].lower()
                        if ql not in seen:
                            seen.add(ql)
                            all_entries.append(e)
                except Exception:
                    continue
                time.sleep(CRAWL_DELAY)

        # Hacker News
        if "hackernews" in enabled_sources:
            try:
                results = crawl_hackernews(query, max_per_source)
                for e in results:
                    ql = e["question"].lower()
                    if ql not in seen:
                        seen.add(ql)
                        all_entries.append(e)
            except Exception:
                pass
            time.sleep(CRAWL_DELAY)

        # DEV.to
        if "devto" in enabled_sources:
            try:
                results = crawl_devto(query, max_per_source)
                for e in results:
                    ql = e["question"].lower()
                    if ql not in seen:
                        seen.add(ql)
                        all_entries.append(e)
            except Exception:
                pass
            time.sleep(CRAWL_DELAY)

    return all_entries


# ── Config management ────────────────────────────────────────────────

def load_forum_config() -> dict:
    if FORUM_CONFIG_PATH.exists():
        return json.loads(FORUM_CONFIG_PATH.read_text(encoding="utf-8"))
    return _default_forum_config()


def save_forum_config(config: dict) -> None:
    FORUM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORUM_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _default_forum_config() -> dict:
    return {
        "topics": [
            {
                "name": "Wi-Fi Security",
                "keywords": ["WPA3", "802.1X", "wireless security"],
                "enabled": True,
                "max_per_source": 10,
                "sources": ["stackexchange", "reddit", "hackernews", "devto"],
            },
            {
                "name": "Network Troubleshooting",
                "keywords": ["packet loss", "latency", "DNS resolution"],
                "enabled": True,
                "max_per_source": 10,
                "sources": ["stackexchange", "reddit", "hackernews", "devto"],
            },
        ],
        "last_crawl": None,
        "stats": {},
    }


def save_forum_knowledge(entries: list[dict], topic_name: str) -> Path | None:
    """Persist new forum entries, skipping ones already saved (None if none new)."""
    EXTRA_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    new_entries = dedupe_new_entries(entries, EXTRA_KNOWLEDGE_DIR)
    if entries and not new_entries:
        return None
    safe = re.sub(r"[^\w\-]", "_", topic_name.lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fp = EXTRA_KNOWLEDGE_DIR / f"forum_{safe}_{ts}.json"
    fp.write_text(
        json.dumps(new_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return fp


def run_all_forum_crawlers() -> dict:
    """Run all enabled forum crawler topics."""
    config = load_forum_config()
    results: dict[str, dict] = {}
    total_new = 0

    for topic in config.get("topics", []):
        if not topic.get("enabled", True):
            results[topic["name"]] = {"status": "disabled", "entries": 0}
            continue
        try:
            entries = crawl_forums(
                topic["name"],
                topic.get("keywords", []),
                topic.get("max_per_source", 10),
                topic.get("sources", ["stackexchange", "reddit", "hackernews", "devto"]),
            )
            if entries:
                fp = save_forum_knowledge(entries, topic["name"])
                if fp is not None:
                    results[topic["name"]] = {
                        "status": "success",
                        "entries": len(entries),
                        "file": str(fp),
                    }
                    total_new += len(entries)
                    # Count entries by source
                    source_counts = {}
                    for e in entries:
                        src = (e.get("source", "unknown").split(":")[0])
                        source_counts[src] = source_counts.get(src, 0) + 1
                    log_learning_event("forum_crawler", topic["name"], len(entries),
                                       {"source_breakdown": source_counts})
                else:
                    results[topic["name"]] = {"status": "no_new", "entries": 0}
            else:
                results[topic["name"]] = {"status": "no_results", "entries": 0}
        except Exception as exc:
            results[topic["name"]] = {"status": f"error: {exc}", "entries": 0}

    # Update stats
    stats = config.get("stats", {})
    for name, result in results.items():
        if name not in stats:
            stats[name] = {"total_crawled": 0, "crawl_count": 0}
        stats[name]["total_crawled"] += result.get("entries", 0)
        stats[name]["crawl_count"] += 1
        stats[name]["last_crawl"] = datetime.now(timezone.utc).isoformat()

    config["last_crawl"] = datetime.now(timezone.utc).isoformat()
    config["stats"] = stats
    save_forum_config(config)

    return {"topics": results, "total_new_entries": total_new, "stats": stats}


def crawl_single_forum_topic(
    topic_name: str,
    keywords: list[str] | None = None,
    max_per_source: int = 10,
    sources: list[str] | None = None,
) -> dict:
    """One-shot forum crawl for a single topic."""
    entries = crawl_forums(topic_name, keywords, max_per_source, sources)
    if entries:
        fp = save_forum_knowledge(entries, topic_name)
        if fp is None:
            return {"status": "no_new", "entries": 0}

        # Count entries by source
        source_counts: dict[str, int] = {}
        for e in entries:
            src = e.get("source", "unknown").split(":")[0]
            source_counts[src] = source_counts.get(src, 0) + 1

        # Update stats
        config = load_forum_config()
        stats = config.get("stats", {})
        if topic_name not in stats:
            stats[topic_name] = {"total_crawled": 0, "crawl_count": 0}
        stats[topic_name]["total_crawled"] += len(entries)
        stats[topic_name]["crawl_count"] += 1
        stats[topic_name]["last_crawl"] = datetime.now(timezone.utc).isoformat()
        config["stats"] = stats
        save_forum_config(config)

        log_learning_event("forum_crawler", topic_name, len(entries),
                           {"source_breakdown": source_counts})

        return {
            "status": "success",
            "entries": len(entries),
            "file": str(fp),
            "samples": entries[:3],
            "source_breakdown": source_counts,
        }
    return {"status": "no_results", "entries": 0}


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text.strip()
    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    if last_period > max_len // 3:
        return truncated[: last_period + 1].strip()
    return truncated.strip() + "."
