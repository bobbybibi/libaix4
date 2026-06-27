"""
crawler.py — Web knowledge crawler using Wikipedia's public API.

Fetches educational content on configurable topics and converts it
to Q&A entries for the libaix knowledge base.  Only uses Wikipedia's
free API — legal, ethical, and rate-limited.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from file_processor import classify_domain, dedupe_new_entries, generate_qa_from_text

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "libaix-crawler/1.0 (educational neural network project; github.com/lindapot-art/libaix)"
CONFIG_PATH = Path("data/crawler_config.json")
EXTRA_KNOWLEDGE_DIR = Path("data/extra_knowledge")
CRAWL_DELAY = 1.0  # seconds between API calls (respect rate limits)


# ── Wikipedia helpers ─────────────────────────────────────────────────

def _wiki_request(params: dict) -> dict:
    params["format"] = "json"
    qs = urllib.parse.urlencode(params)
    url = f"{WIKI_API}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def search_wikipedia(query: str, limit: int = 10) -> list[dict]:
    data = _wiki_request({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
    })
    return data.get("query", {}).get("search", [])


def get_article_text(title: str) -> str:
    data = _wiki_request({
        "action": "query",
        "prop": "extracts",
        "explaintext": "true",
        "titles": title,
    })
    for page in data.get("query", {}).get("pages", {}).values():
        return page.get("extract", "")
    return ""


def get_article_summary(title: str) -> str:
    data = _wiki_request({
        "action": "query",
        "prop": "extracts",
        "exintro": "true",
        "explaintext": "true",
        "titles": title,
    })
    for page in data.get("query", {}).get("pages", {}).values():
        return page.get("extract", "")
    return ""


# ── Core crawl logic ──────────────────────────────────────────────────

def crawl_topic(
    topic: str,
    keywords: list[str] | None = None,
    max_articles: int = 10,
) -> list[dict[str, str]]:
    """Crawl Wikipedia for *topic*, return Q&A entries."""
    keywords = keywords or []
    all_entries: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    visited_titles: set[str] = set()

    search_queries = [topic] + keywords

    for query in search_queries:
        results = search_wikipedia(query, limit=max_articles)
        time.sleep(CRAWL_DELAY)

        for result in results:
            title = result.get("title", "")
            if title in visited_titles:
                continue
            visited_titles.add(title)

            # Summary → direct "What is X?" entry
            summary = get_article_summary(title)
            time.sleep(CRAWL_DELAY)

            if summary and len(summary) > 50:
                domain = classify_domain(summary)
                answer = _truncate_to_sentence(summary, 500)
                q = f"What is {title}?"
                ql = q.lower()
                if ql not in seen_questions:
                    seen_questions.add(ql)
                    all_entries.append({
                        "question": q,
                        "answer": answer,
                        "domain": domain,
                        "source": f"wikipedia:{title}",
                    })

            # Full text → more Q&A via heuristics
            text = get_article_text(title)
            time.sleep(CRAWL_DELAY)

            if text and len(text) > 100:
                entries = generate_qa_from_text(text)
                for entry in entries:
                    ql = entry["question"].lower()
                    if ql not in seen_questions:
                        seen_questions.add(ql)
                        entry["source"] = f"wikipedia:{title}"
                        all_entries.append(entry)

    return all_entries


def _truncate_to_sentence(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text.strip()
    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    if last_period > max_len // 3:
        return truncated[: last_period + 1].strip()
    return truncated.strip() + "."


# ── Config management ─────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return _default_config()


def save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _default_config() -> dict:
    return {
        "topics": _default_topics(),
        "last_crawl": None,
        "crawl_interval_hours": 2,
    }


def _default_topics() -> list[dict]:
    """Crawl topics for the active trade pack, or the built-in networking set.

    A fresh crawler config is seeded from whatever trade is active, so pointing
    libaix at a new trade automatically crawls that trade's topics.
    """
    try:
        import trade_pack

        topics = trade_pack.crawl_topics_for()
        if topics:
            return topics
    except Exception:
        pass
    return [
        {
            "name": "Wi-Fi Security",
            "keywords": ["WPA3", "802.1X authentication", "wireless security"],
            "enabled": True,
            "max_articles": 8,
        },
        {
            "name": "Network Protocols",
            "keywords": ["TCP/IP protocol", "routing protocol", "network protocol"],
            "enabled": True,
            "max_articles": 8,
        },
        {
            "name": "Corporate Network Security",
            "keywords": ["enterprise security", "zero trust architecture", "NAC"],
            "enabled": True,
            "max_articles": 8,
        },
        {
            "name": "Network Troubleshooting",
            "keywords": ["Wi-Fi troubleshooting", "network diagnostics"],
            "enabled": True,
            "max_articles": 5,
        },
    ]


# ── Persistence ───────────────────────────────────────────────────────

def save_crawled_knowledge(entries: list[dict], topic_name: str) -> Path | None:
    """Persist newly-crawled entries, skipping ones already saved.

    Returns the written file, or None when a non-empty crawl produced nothing
    new — so re-crawling the same articles no longer bloats the corpus.
    """
    EXTRA_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    new_entries = dedupe_new_entries(entries, EXTRA_KNOWLEDGE_DIR)
    if entries and not new_entries:
        return None
    safe = re.sub(r"[^\w\-]", "_", topic_name.lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fp = EXTRA_KNOWLEDGE_DIR / f"crawl_{safe}_{ts}.json"
    fp.write_text(
        json.dumps(new_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return fp


# ── High-level runners ────────────────────────────────────────────────

def run_all_crawlers() -> dict:
    """Run every enabled crawler topic.  Returns results summary."""
    config = load_config()
    results: dict[str, dict] = {}
    total_new = 0

    for topic in config.get("topics", []):
        if not topic.get("enabled", True):
            results[topic["name"]] = {"status": "disabled", "entries": 0}
            continue
        try:
            entries = crawl_topic(
                topic["name"],
                topic.get("keywords", []),
                topic.get("max_articles", 8),
            )
            if entries:
                fp = save_crawled_knowledge(entries, topic["name"])
                if fp is not None:
                    results[topic["name"]] = {
                        "status": "success",
                        "entries": len(entries),
                        "file": str(fp),
                    }
                    total_new += len(entries)
                else:
                    results[topic["name"]] = {"status": "no_new", "entries": 0}
            else:
                results[topic["name"]] = {"status": "no_results", "entries": 0}
        except Exception as exc:
            results[topic["name"]] = {"status": f"error: {exc}", "entries": 0}

    config["last_crawl"] = datetime.now(timezone.utc).isoformat()
    save_config(config)
    return {"topics": results, "total_new_entries": total_new}


def crawl_single_topic(
    topic_name: str,
    keywords: list[str] | None = None,
    max_articles: int = 10,
) -> dict:
    """One-shot crawl triggered by admin prompt.  Returns result dict."""
    entries = crawl_topic(topic_name, keywords, max_articles)
    if entries:
        fp = save_crawled_knowledge(entries, topic_name)
        if fp is None:
            return {"status": "no_new", "entries": 0}
        return {
            "status": "success",
            "entries": len(entries),
            "file": str(fp),
            "samples": entries[:3],
        }
    return {"status": "no_results", "entries": 0}
