"""
site_crawler.py — Crawl any URL / website and extract topic-relevant knowledge.

Given a URL and a topic, this crawler:
  1. Fetches the page
  2. Discovers internal links (same domain)
  3. Follows links up to a configurable depth
  4. Extracts text from each page
  5. Filters for topic-relevant content only
  6. Generates Q&A entries
  7. Discards the original HTML data (no caching)
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from file_processor import classify_domain, generate_qa_from_text

USER_AGENT = (
    "libaix-crawler/1.0 (educational neural network project; "
    "github.com/lindapot-art/libaix)"
)
EXTRA_KNOWLEDGE_DIR = Path("data/extra_knowledge")
SITE_CONFIG_PATH = Path("data/site_crawl_jobs.json")
CRAWL_DELAY = 1.5  # seconds between requests
MAX_PAGE_SIZE = 2 * 1024 * 1024  # 2 MB max per page
MAX_PAGES_PER_SITE = 50  # safety limit


# ── HTML text extraction (no external deps) ──────────────────────────

class _TextExtractor(HTMLParser):
    """Simple HTML → plain text extractor."""

    _SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link", "svg", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._text.append(text)

    def get_text(self) -> str:
        return "\n".join(self._text)


class _LinkExtractor(HTMLParser):
    """Extract href links from HTML."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def _extract_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.get_text()


def _extract_links(html: str, base_url: str) -> list[str]:
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    links = []
    for href in parser.links:
        try:
            absolute = urllib.parse.urljoin(base_url, href)
            # Strip fragment
            absolute = absolute.split("#")[0]
            if absolute.startswith(("http://", "https://")):
                links.append(absolute)
        except Exception:
            continue
    return links


def _fetch_page(url: str) -> str | None:
    """Fetch a single page. Returns HTML string or None."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html",
        })
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None
            data = resp.read(MAX_PAGE_SIZE)
            return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def _is_same_domain(url: str, base_domain: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc == base_domain or parsed.netloc.endswith(f".{base_domain}")


def _is_relevant(text: str, topic: str, keywords: list[str]) -> bool:
    """Check if text contains topic-relevant content."""
    text_lower = text.lower()
    topic_words = topic.lower().split()

    # Must contain at least some topic words
    matches = sum(1 for w in topic_words if w in text_lower)
    if matches >= max(1, len(topic_words) // 2):
        return True

    # Or keywords
    for kw in keywords:
        if kw.lower() in text_lower:
            return True

    return False


# ── Core site crawler ────────────────────────────────────────────────

def crawl_site(
    start_url: str,
    topic: str,
    keywords: list[str] | None = None,
    max_pages: int = 20,
    max_depth: int = 2,
) -> dict:
    """
    Crawl a website starting from start_url, extract topic-relevant Q&A.

    Returns dict with entries, stats, and no cached data.
    """
    keywords = keywords or []
    parsed_start = urllib.parse.urlparse(start_url)
    base_domain = parsed_start.netloc

    if not base_domain:
        return {"status": "error", "error": "Invalid URL", "entries": 0}

    visited: set[str] = set()
    all_entries: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    pages_crawled = 0
    pages_relevant = 0
    total_text_processed = 0

    # BFS queue: (url, depth)
    queue: list[tuple[str, int]] = [(start_url, 0)]

    while queue and pages_crawled < min(max_pages, MAX_PAGES_PER_SITE):
        url, depth = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        # Fetch page
        html = _fetch_page(url)
        if not html:
            continue
        pages_crawled += 1

        # Extract text — discard HTML immediately
        text = _extract_text(html)
        total_text_processed += len(text)

        # Check relevance
        if not _is_relevant(text, topic, keywords):
            # Still follow links if within depth
            if depth < max_depth:
                links = _extract_links(html, url)
                for link in links:
                    if link not in visited and _is_same_domain(link, base_domain):
                        queue.append((link, depth + 1))
            # Discard original data
            del html, text
            continue

        pages_relevant += 1

        # Generate Q&A from relevant text
        entries = generate_qa_from_text(text)
        for entry in entries:
            ql = entry["question"].lower()
            if ql not in seen_questions:
                seen_questions.add(ql)
                entry["source"] = f"site:{base_domain}:{url}"
                all_entries.append(entry)

        # Also create a summary entry for the page
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
        page_title = title_match.group(1).strip() if title_match else topic
        summary_q = f"What does {page_title} describe about {topic}?"
        if summary_q.lower() not in seen_questions and len(text) > 100:
            seen_questions.add(summary_q.lower())
            domain = classify_domain(text)
            all_entries.append({
                "question": summary_q,
                "answer": _truncate(text, 500),
                "domain": domain,
                "source": f"site:{base_domain}",
            })

        # Follow internal links
        if depth < max_depth:
            links = _extract_links(html, url)
            for link in links:
                if link not in visited and _is_same_domain(link, base_domain):
                    queue.append((link, depth + 1))

        # DISCARD original data — flush
        del html, text

        time.sleep(CRAWL_DELAY)

    return {
        "status": "success" if all_entries else "no_results",
        "entries": len(all_entries),
        "entries_data": all_entries,
        "stats": {
            "pages_crawled": pages_crawled,
            "pages_relevant": pages_relevant,
            "total_text_bytes": total_text_processed,
            "unique_entries": len(all_entries),
        },
    }


# ── Job management ───────────────────────────────────────────────────

def load_site_jobs() -> list[dict]:
    if SITE_CONFIG_PATH.exists():
        return json.loads(SITE_CONFIG_PATH.read_text(encoding="utf-8"))
    return []


def save_site_jobs(jobs: list[dict]) -> None:
    SITE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITE_CONFIG_PATH.write_text(
        json.dumps(jobs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def add_site_job(url: str, topic: str, keywords: list[str] | None = None,
                 max_pages: int = 20, max_depth: int = 2) -> dict:
    """Add a new site crawl job and execute it immediately."""
    keywords = keywords or []

    # Validate URL
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {"status": "error", "error": "Invalid URL format"}

    # Crawl
    result = crawl_site(url, topic, keywords, max_pages, max_depth)

    entries = result.get("entries_data", [])
    saved_path = None
    if entries:
        saved_path = _save_site_knowledge(entries, parsed.netloc, topic)

    # Record job
    jobs = load_site_jobs()
    job = {
        "url": url,
        "domain": parsed.netloc,
        "topic": topic,
        "keywords": keywords,
        "max_pages": max_pages,
        "max_depth": max_depth,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "stats": result.get("stats", {}),
        "entries_extracted": len(entries),
        "file": str(saved_path) if saved_path else None,
    }
    jobs.append(job)
    save_site_jobs(jobs)

    # Clean up — don't keep entries_data in return (it's saved to disk)
    result.pop("entries_data", None)
    result["file"] = str(saved_path) if saved_path else None
    result["samples"] = entries[:3] if entries else []

    return result


def get_site_crawl_stats() -> dict:
    """Get aggregate stats from all site crawl jobs."""
    jobs = load_site_jobs()
    total_entries = sum(j.get("entries_extracted", 0) for j in jobs)
    total_pages = sum(j.get("stats", {}).get("pages_crawled", 0) for j in jobs)
    domains = list(set(j.get("domain", "") for j in jobs))
    return {
        "total_jobs": len(jobs),
        "total_entries": total_entries,
        "total_pages_crawled": total_pages,
        "domains_crawled": domains,
        "recent_jobs": jobs[-10:],  # last 10
    }


def clear_site_jobs() -> None:
    """Clear all job history (knowledge files remain)."""
    save_site_jobs([])


# ── Persistence ──────────────────────────────────────────────────────

def _save_site_knowledge(entries: list[dict], domain: str, topic: str) -> Path:
    EXTRA_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    safe_domain = re.sub(r"[^\w\-]", "_", domain)
    safe_topic = re.sub(r"[^\w\-]", "_", topic.lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fp = EXTRA_KNOWLEDGE_DIR / f"site_{safe_domain}_{safe_topic}_{ts}.json"
    fp.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return fp


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text.strip()
    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    if last_period > max_len // 3:
        return truncated[: last_period + 1].strip()
    return truncated.strip() + "."
