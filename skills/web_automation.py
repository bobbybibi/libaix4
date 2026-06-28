"""
web_automation.py — Web automation skill for libaix.

Provides commands to:
  • Fetch and extract readable content from a URL
  • Extract all hyperlinks from a web page
  • Search the web via DuckDuckGo HTML
  • Monitor a page for changes using content hashing

Uses only the Python standard library.  No external packages required.
HTTP requests go through ``urllib.request``; HTML parsing uses
``html.parser.HTMLParser``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

from skill_registry import Skill, SkillCommand, SkillResult

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_HTTP_TIMEOUT = 15
_MAX_CONTENT_BYTES = 100 * 1024  # 100 KB
_USER_AGENT = (
    "Mozilla/5.0 (compatible; libaix/1.0; +https://github.com/bobbybibi/libaix4)"
)
_DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/"


# ── HTML helpers ─────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, stripping scripts and styles."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "head"})

    def __init__(self) -> None:
        super().__init__()
        self.title: str = ""
        self.text_parts: list[str] = []
        self._skip_depth: int = 0
        self._in_title: bool = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in self._SKIP_TAGS:
            self._skip_depth += 1
        if lower == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if lower == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self.text_parts)


class _LinkExtractor(HTMLParser):
    """Extract all ``<a href="...">`` links and their anchor text."""

    def __init__(self, base_url: str = "") -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._base_url = base_url
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href is not None:
                self._current_href = href
                self._current_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            href = self._current_href
            if self._base_url and not href.startswith(("http://", "https://", "mailto:")):
                href = urllib.parse.urljoin(self._base_url, href)
            self.links.append({
                "url": href,
                "text": " ".join(self._current_text).strip(),
            })
            self._current_href = None
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data.strip())


class _SearchResultExtractor(HTMLParser):
    """Extract search result titles, URLs, and snippets from DuckDuckGo HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_result_link: bool = False
        self._current_title: list[str] = []
        self._current_url: str = ""
        self._in_snippet: bool = False
        self._current_snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")

        if tag.lower() == "a" and "result__a" in cls:
            self._in_result_link = True
            self._current_url = attrs_dict.get("href", "")
            self._current_title = []

        if tag.lower() == "a" and "result__snippet" in cls:
            self._in_snippet = True
            self._current_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._in_result_link:
            self._in_result_link = False
        if tag.lower() == "a" and self._in_snippet:
            self._in_snippet = False
            self.results.append({
                "title": " ".join(self._current_title).strip(),
                "url": self._current_url,
                "snippet": " ".join(self._current_snippet).strip(),
            })

    def handle_data(self, data: str) -> None:
        if self._in_result_link:
            self._current_title.append(data.strip())
        if self._in_snippet:
            self._current_snippet.append(data.strip())


# ── HTTP helper ──────────────────────────────────────────────────────

def _fetch_url(url: str, *, max_bytes: int = _MAX_CONTENT_BYTES) -> tuple[str, int]:
    """Fetch *url* and return ``(body, http_status)``.

    Raises ``urllib.error.URLError`` or ``OSError`` on failure.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        raw = resp.read(max_bytes)
        encoding = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(encoding, errors="replace"), resp.status


# ── Skill implementation ─────────────────────────────────────────────

class WebAutomationSkill(Skill):
    """Automate web tasks — fetch pages, extract data, fill forms."""

    def __init__(self) -> None:
        super().__init__(
            name="web_automation",
            description="Automate web tasks — fetch pages, extract data, fill forms",
            version="1.0.0",
            category="automation",
        )
        # page URL → SHA-256 hex digest of last-seen content
        self._page_hashes: dict[str, str] = {}

    # ── public interface ─────────────────────────────────────────────

    def get_commands(self) -> list[SkillCommand]:
        return [
            SkillCommand(
                name="fetch_page",
                description="Fetch and extract content from a URL",
                patterns=[
                    r"(fetch|get|open|read)\s+(page|url|website|site)\s+(?P<url>\S+)",
                    r"what.+(is|does)\s+(?P<url>https?://\S+)\s+(say|contain)",
                ],
                args_schema={"url": {"type": "string", "required": True}},
                category="automation",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="extract_links",
                description="Extract all links from a page",
                patterns=[
                    r"(extract|get|find)\s+links?\s+(from|on|in)\s+(?P<url>\S+)",
                ],
                args_schema={"url": {"type": "string", "required": True}},
                category="automation",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="search_web",
                description="Search for information on the web",
                patterns=[
                    r"(search|look\s+up|find)\s+(for\s+)?(?P<query>.+)\s+(on|online|on\s+the\s+web)",
                    r"(web|internet)\s+search\s+(for\s+)?(?P<query>.+)",
                ],
                args_schema={"query": {"type": "string", "required": True}},
                category="automation",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="monitor_page",
                description="Monitor a page for changes",
                patterns=[
                    r"(monitor|watch)\s+(page|url|website|site)\s+(?P<url>\S+)",
                    r"(alert|notify)\s+(me\s+)?(if|when)\s+(?P<url>\S+)\s+changes",
                ],
                args_schema={"url": {"type": "string", "required": True}},
                category="automation",
                requires_confirmation=False,
            ),
        ]

    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        """Dispatch *command* to the appropriate private handler."""
        dispatch: dict[str, Any] = {
            "fetch_page": self._fetch_page,
            "extract_links": self._extract_links,
            "search_web": self._search_web,
            "monitor_page": self._monitor_page,
        }
        handler = dispatch.get(command)
        if handler is None:
            return SkillResult(
                success=False,
                message=f"Unknown command: {command}",
            )
        try:
            return handler(args)
        except Exception as exc:
            log.exception("web_automation command '%s' failed: %s", command, exc)
            return SkillResult(
                success=False,
                message=f"Command '{command}' failed: {exc}",
            )

    # ── command implementations ──────────────────────────────────────

    def _fetch_page(self, args: dict[str, Any]) -> SkillResult:
        """Fetch a URL and return the extracted text content."""
        url = self._normalise_url(args.get("url", ""))
        if not url:
            return SkillResult(success=False, message="No URL provided.")

        body, status = self._safe_fetch(url)
        if body is None:
            return SkillResult(
                success=False,
                message=f"Failed to fetch '{url}': {status}",
            )

        extractor = _TextExtractor()
        try:
            extractor.feed(body)
        except Exception as exc:
            log.warning("HTML parse error for %s: %s", url, exc)

        text = extractor.get_text()
        title = extractor.title.strip()

        return SkillResult(
            success=True,
            message=f"Fetched '{title or url}' ({len(text)} chars extracted).",
            data={
                "url": url,
                "title": title,
                "text": text[:10000],  # cap to keep result manageable
                "text_length": len(text),
                "http_status": status,
            },
        )

    def _extract_links(self, args: dict[str, Any]) -> SkillResult:
        """Extract all hyperlinks from a web page."""
        url = self._normalise_url(args.get("url", ""))
        if not url:
            return SkillResult(success=False, message="No URL provided.")

        body, status = self._safe_fetch(url)
        if body is None:
            return SkillResult(
                success=False,
                message=f"Failed to fetch '{url}': {status}",
            )

        extractor = _LinkExtractor(base_url=url)
        try:
            extractor.feed(body)
        except Exception as exc:
            log.warning("HTML parse error for %s: %s", url, exc)

        links = extractor.links

        return SkillResult(
            success=True,
            message=f"Extracted {len(links)} link(s) from '{url}'.",
            data={
                "url": url,
                "links": links,
                "count": len(links),
                "http_status": status,
            },
        )

    def _search_web(self, args: dict[str, Any]) -> SkillResult:
        """Search the web via DuckDuckGo HTML and parse results."""
        query = args.get("query", "").strip()
        if not query:
            return SkillResult(success=False, message="No search query provided.")

        search_url = (
            f"{_DUCKDUCKGO_HTML}?q={urllib.parse.quote_plus(query)}"
        )

        body, status = self._safe_fetch(search_url)
        if body is None:
            return SkillResult(
                success=True,
                message=(
                    f"Could not reach DuckDuckGo. Try searching manually: "
                    f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"
                ),
                data={"query": query, "search_url": search_url},
            )

        extractor = _SearchResultExtractor()
        try:
            extractor.feed(body)
        except Exception as exc:
            log.warning("HTML parse error for search results: %s", exc)

        results = extractor.results

        if not results:
            return SkillResult(
                success=True,
                message=(
                    f"No results parsed for '{query}'. "
                    f"Try: https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"
                ),
                data={"query": query, "search_url": search_url, "results": []},
            )

        return SkillResult(
            success=True,
            message=f"Found {len(results)} result(s) for '{query}'.",
            data={
                "query": query,
                "results": results[:10],  # top 10
                "count": len(results),
                "search_url": search_url,
            },
        )

    def _monitor_page(self, args: dict[str, Any]) -> SkillResult:
        """Monitor a page for changes by comparing content hashes."""
        url = self._normalise_url(args.get("url", ""))
        if not url:
            return SkillResult(success=False, message="No URL provided.")

        body, status = self._safe_fetch(url)
        if body is None:
            return SkillResult(
                success=False,
                message=f"Failed to fetch '{url}': {status}",
            )

        current_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        previous_hash = self._page_hashes.get(url)

        self._page_hashes[url] = current_hash

        if previous_hash is None:
            return SkillResult(
                success=True,
                message=(
                    f"Now monitoring '{url}'. "
                    "Run this command again to check for changes."
                ),
                data={
                    "url": url,
                    "hash": current_hash,
                    "changed": False,
                    "first_check": True,
                },
            )

        changed = current_hash != previous_hash
        message = (
            f"Page '{url}' has CHANGED since last check."
            if changed
            else f"No changes detected for '{url}'."
        )

        return SkillResult(
            success=True,
            message=message,
            data={
                "url": url,
                "hash": current_hash,
                "previous_hash": previous_hash,
                "changed": changed,
                "first_check": False,
            },
        )

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise_url(url: str) -> str:
        """Ensure the URL has an ``http(s)://`` scheme."""
        url = url.strip()
        if not url:
            return ""
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "https://" + url
        return url

    @staticmethod
    def _safe_fetch(url: str) -> tuple[str | None, int | str]:
        """Fetch *url*, returning ``(body, status)`` or ``(None, error_msg)``."""
        try:
            body, status = _fetch_url(url)
            return body, status
        except urllib.error.HTTPError as exc:
            return None, f"HTTP {exc.code} {exc.reason}"
        except urllib.error.URLError as exc:
            return None, f"URL error: {exc.reason}"
        except OSError as exc:
            return None, f"Network error: {exc}"
