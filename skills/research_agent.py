"""
research_agent.py — Research agent skill for libaix.

Provides automated research capabilities:
  • Deep-research a topic using the Wikipedia REST API
  • Summarize content from any URL (HTML stripped)
  • Compare two topics side-by-side
  • Fact-check a claim with appropriate disclaimers

Results are cached in memory to avoid redundant network requests.

Uses only the Python standard library.  No external packages required.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any

from skill_registry import Skill, SkillCommand, SkillResult

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
_USER_AGENT = "libaix-research-agent/1.0 (https://github.com/bobbybibi/libaix4)"
_REQUEST_TIMEOUT = 15
_SUMMARY_MAX_CHARS = 2000


# ── HTML tag stripper ────────────────────────────────────────────────

class _HTMLTagStripper(HTMLParser):
    """Minimal HTML parser that extracts visible text."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)

    def error(self, message: str) -> None:
        log.debug("HTML parse error: %s", message)


def _strip_html(html: str) -> str:
    """Remove HTML tags and return visible text."""
    stripper = _HTMLTagStripper()
    try:
        stripper.feed(html)
    except Exception:
        # Fall back to regex if the parser chokes
        return re.sub(r"<[^>]+>", "", html)
    return stripper.get_text()


# ── Skill implementation ─────────────────────────────────────────────

class ResearchAgentSkill(Skill):
    """Research any topic using multiple sources and compile findings."""

    def __init__(self) -> None:
        super().__init__(
            name="research_agent",
            description="Research any topic using multiple sources and compile findings",
            version="1.0.0",
            category="automation",
        )
        self._research_cache: dict[str, dict[str, Any]] = {}

    # ── Skill interface ──────────────────────────────────────────────

    def get_commands(self) -> list[SkillCommand]:
        return [
            SkillCommand(
                name="research_topic",
                description="Deep research a topic using online sources",
                patterns=[
                    r"research\s+(?P<topic>.+)",
                    r"(study|learn\s+about|investigate)\s+(?P<topic>.+)",
                    r"find\s+out\s+(everything\s+)?(about\s+)?(?P<topic>.+)",
                    r"(deep\s+dive|dig\s+into)\s+(?P<topic>.+)",
                ],
                args_schema={"topic": {"type": "string", "required": True}},
                category="automation",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="summarize_url",
                description="Summarize content from a URL",
                patterns=[
                    r"summarize\s+(?P<url>https?://\S+)",
                    r"(tldr|summary)\s+(of\s+)?(?P<url>https?://\S+)",
                ],
                args_schema={"url": {"type": "string", "required": True}},
                category="automation",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="compare_topics",
                description="Compare two topics side-by-side",
                patterns=[
                    r"compare\s+(?P<topic1>.+?)\s+(and|vs\.?|versus|with)\s+(?P<topic2>.+)",
                ],
                args_schema={
                    "topic1": {"type": "string", "required": True},
                    "topic2": {"type": "string", "required": True},
                },
                category="automation",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="fact_check",
                description="Verify a claim using available sources",
                patterns=[
                    r"(fact\s+check|verify|is\s+it\s+true\s+that)\s+(?P<claim>.+)",
                ],
                args_schema={"claim": {"type": "string", "required": True}},
                category="automation",
                requires_confirmation=False,
            ),
        ]

    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        """Dispatch *command* to the appropriate private handler."""
        dispatch: dict[str, Any] = {
            "research_topic": self._research_topic,
            "summarize_url": self._summarize_url,
            "compare_topics": self._compare_topics,
            "fact_check": self._fact_check,
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
            log.exception("research_agent command '%s' failed: %s", command, exc)
            return SkillResult(
                success=False,
                message=f"Command '{command}' failed: {exc}",
            )

    # ── Command implementations ──────────────────────────────────────

    def _research_topic(self, args: dict[str, Any]) -> SkillResult:
        """Research a topic using the Wikipedia summary API."""
        topic = args.get("topic", "").strip()
        if not topic:
            return SkillResult(
                success=False,
                message="No topic provided.",
            )

        # Check cache first
        cache_key = f"topic:{topic.lower()}"
        if cache_key in self._research_cache:
            cached = self._research_cache[cache_key]
            return SkillResult(
                success=True,
                message=f"Research results for '{topic}' (cached)",
                data=cached,
            )

        result = self._fetch_wikipedia_summary(topic)
        if result is None:
            return SkillResult(
                success=False,
                message=f"Could not find information about '{topic}'.",
                data={"topic": topic},
            )

        research_data: dict[str, Any] = {
            "topic": topic,
            "title": result.get("title", topic),
            "summary": result.get("extract", ""),
            "description": result.get("description", ""),
            "source": "Wikipedia",
            "source_url": result.get("content_urls", {})
            .get("desktop", {})
            .get("page", ""),
            "thumbnail": result.get("thumbnail", {}).get("source", ""),
        }

        self._research_cache[cache_key] = research_data

        return SkillResult(
            success=True,
            message=f"Research results for '{topic}'",
            data=research_data,
        )

    def _summarize_url(self, args: dict[str, Any]) -> SkillResult:
        """Fetch a URL, strip HTML, and return the first N characters."""
        url = args.get("url", "").strip()
        if not url:
            return SkillResult(
                success=False,
                message="No URL provided.",
            )

        # Check cache
        cache_key = f"url:{url}"
        if cache_key in self._research_cache:
            cached = self._research_cache[cache_key]
            return SkillResult(
                success=True,
                message=f"Summary of {url} (cached)",
                data=cached,
            )

        try:
            raw_html = self._fetch_url(url)
        except urllib.error.HTTPError as exc:
            return SkillResult(
                success=False,
                message=f"HTTP error fetching URL: {exc.code} {exc.reason}",
                data={"url": url},
            )
        except urllib.error.URLError as exc:
            return SkillResult(
                success=False,
                message=f"Cannot reach URL: {exc.reason}",
                data={"url": url},
            )

        text = _strip_html(raw_html)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        truncated = len(text) > _SUMMARY_MAX_CHARS
        summary = text[:_SUMMARY_MAX_CHARS]

        summary_data: dict[str, Any] = {
            "url": url,
            "summary": summary,
            "char_count": len(text),
            "truncated": truncated,
        }

        self._research_cache[cache_key] = summary_data

        return SkillResult(
            success=True,
            message=f"Summary of {url} ({len(summary)} chars)",
            data=summary_data,
        )

    def _compare_topics(self, args: dict[str, Any]) -> SkillResult:
        """Research two topics and present them side-by-side."""
        topic1 = args.get("topic1", "").strip()
        topic2 = args.get("topic2", "").strip()

        if not topic1 or not topic2:
            return SkillResult(
                success=False,
                message="Two topics are required for comparison.",
            )

        result1 = self._research_topic({"topic": topic1})
        result2 = self._research_topic({"topic": topic2})

        comparison: dict[str, Any] = {
            "topic1": {
                "name": topic1,
                "found": result1.success,
                "data": result1.data if result1.success else {},
            },
            "topic2": {
                "name": topic2,
                "found": result2.success,
                "data": result2.data if result2.success else {},
            },
        }

        if result1.success and result2.success:
            message = f"Comparison of '{topic1}' and '{topic2}'"
        elif result1.success:
            message = f"Found info on '{topic1}' but not '{topic2}'"
        elif result2.success:
            message = f"Found info on '{topic2}' but not '{topic1}'"
        else:
            message = f"Could not find information on either '{topic1}' or '{topic2}'"

        return SkillResult(
            success=result1.success or result2.success,
            message=message,
            data=comparison,
        )

    def _fact_check(self, args: dict[str, Any]) -> SkillResult:
        """Attempt to verify a claim by researching its key terms."""
        claim = args.get("claim", "").strip()
        if not claim:
            return SkillResult(
                success=False,
                message="No claim provided.",
            )

        # Extract key terms — use longest words as likely topics
        words = re.findall(r"[a-zA-Z]{3,}", claim)
        # Sort by length descending, take up to 3 most significant words
        key_terms = sorted(set(words), key=len, reverse=True)[:3]

        findings: list[dict[str, Any]] = []
        for term in key_terms:
            result = self._fetch_wikipedia_summary(term)
            if result is not None:
                findings.append({
                    "term": term,
                    "title": result.get("title", term),
                    "extract": result.get("extract", ""),
                    "source": "Wikipedia",
                })

        disclaimer = (
            "DISCLAIMER: This fact-check is based on automated research "
            "from publicly available sources and should not be considered "
            "authoritative. Always verify important claims with trusted, "
            "primary sources."
        )

        return SkillResult(
            success=True,
            message=f"Fact-check results for: {claim}",
            data={
                "claim": claim,
                "key_terms": key_terms,
                "findings": findings,
                "findings_count": len(findings),
                "disclaimer": disclaimer,
            },
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def _fetch_wikipedia_summary(self, topic: str) -> dict[str, Any] | None:
        """Fetch a Wikipedia summary for *topic*, returning parsed JSON or None."""
        # URL-encode the topic (spaces → underscores for Wikipedia)
        encoded_topic = urllib.request.quote(topic.replace(" ", "_"))
        url = _WIKIPEDIA_API.format(topic=encoded_topic)

        try:
            raw = self._fetch_url(url)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            log.debug("Wikipedia lookup failed for '%s': %s", topic, exc)
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.debug("Failed to parse Wikipedia response for '%s': %s", topic, exc)
            return None

        # Wikipedia returns {"type": "disambiguation"} or similar on miss
        if data.get("type") not in ("standard", "disambiguation"):
            return None

        return data

    @staticmethod
    def _fetch_url(url: str) -> str:
        """Fetch *url* and return the response body as a string."""
        request = urllib.request.Request(
            url,
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
