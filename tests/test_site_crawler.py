"""Tests for site_crawler.py — mock all HTTP calls."""
from __future__ import annotations


import pytest

import site_crawler
from site_crawler import (
    _extract_links,
    _extract_text,
    _is_relevant,
    _is_same_domain,
    _truncate,
    add_site_job,
    clear_site_jobs,
    crawl_site,
    get_site_crawl_stats,
    load_site_jobs,
    save_site_jobs,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_fs(tmp_path, monkeypatch):
    """Redirect all file paths to tmp_path and disable network delay."""
    monkeypatch.setattr(site_crawler, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek")
    monkeypatch.setattr(site_crawler, "SITE_CONFIG_PATH", tmp_path / "sj.json")
    monkeypatch.setattr(site_crawler, "CRAWL_DELAY", 0.0)
    monkeypatch.setattr(site_crawler.time, "sleep", lambda _: None)


# ── HTML extraction ──────────────────────────────────────────────────

class TestExtractText:
    def test_basic_html(self):
        html = "<html><body><p>Hello world</p></body></html>"
        text = _extract_text(html)
        assert "Hello world" in text

    def test_skips_script_style(self):
        html = "<body><script>var x=1;</script><p>visible</p><style>.x{}</style></body>"
        text = _extract_text(html)
        assert "visible" in text
        assert "var x" not in text
        assert ".x{}" not in text

    def test_empty_html(self):
        assert _extract_text("") == ""

    def test_malformed_html_does_not_crash(self):
        _extract_text("<div><p>unclosed")


class TestExtractLinks:
    def test_extracts_absolute_links(self):
        html = '<a href="https://example.com/page1">link</a>'
        links = _extract_links(html, "https://example.com")
        assert "https://example.com/page1" in links

    def test_resolves_relative_links(self):
        html = '<a href="/about">About</a>'
        links = _extract_links(html, "https://example.com/index.html")
        assert "https://example.com/about" in links

    def test_strips_fragments(self):
        html = '<a href="https://example.com/page#section">link</a>'
        links = _extract_links(html, "https://example.com")
        assert "https://example.com/page" in links
        assert not any("#" in link for link in links)

    def test_ignores_non_http(self):
        html = '<a href="mailto:x@x.com">email</a><a href="https://ok.com">ok</a>'
        links = _extract_links(html, "https://example.com")
        assert all(link.startswith("http") for link in links)


# ── Domain/relevance checks ─────────────────────────────────────────

class TestIsSameDomain:
    def test_exact_match(self):
        assert _is_same_domain("https://example.com/p", "example.com")

    def test_subdomain(self):
        assert _is_same_domain("https://docs.example.com/p", "example.com")

    def test_different_domain(self):
        assert not _is_same_domain("https://other.com/p", "example.com")


class TestIsRelevant:
    def test_topic_words_match(self):
        assert _is_relevant("TCP is a transport protocol", "TCP protocol", [])

    def test_keyword_match(self):
        assert _is_relevant("Something about routing", "networking", ["routing"])

    def test_no_match(self):
        assert not _is_relevant("cooking recipes for dinner", "TCP networking", [])


# ── Truncate ─────────────────────────────────────────────────────────

class TestTruncate:
    def test_short_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncates(self):
        result = _truncate("a" * 200, 50)
        assert len(result) <= 51


# ── crawl_site ───────────────────────────────────────────────────────

class TestCrawlSite:
    def test_invalid_url_returns_error(self):
        result = crawl_site("not-a-url", "test")
        assert result["status"] == "error"

    def test_crawls_with_mock(self, monkeypatch):
        html = """<html><body>
        <p>TCP is a transport protocol used for reliable data delivery.</p>
        <a href="/page2">Next</a>
        </body></html>"""

        call_count = {"n": 0}

        def mock_fetch(url):
            call_count["n"] += 1
            if call_count["n"] > 3:
                return None
            return html

        monkeypatch.setattr(site_crawler, "_fetch_page", mock_fetch)
        monkeypatch.setattr("site_crawler.classify_domain", lambda t: "networking")
        monkeypatch.setattr("site_crawler.generate_qa_from_text", lambda t: [
            {"question": "What is TCP?", "answer": "TCP is a protocol.", "domain": "networking"},
        ])

        result = crawl_site("https://example.com", "TCP", max_pages=3)
        assert result["status"] in ("success", "no_results")
        assert "stats" in result

    def test_respects_max_pages(self, monkeypatch):
        monkeypatch.setattr(site_crawler, "_fetch_page", lambda url: "<p>content</p>")
        monkeypatch.setattr("site_crawler.generate_qa_from_text", lambda t: [])
        monkeypatch.setattr("site_crawler.classify_domain", lambda t: "general")

        result = crawl_site("https://example.com", "test", max_pages=1, max_depth=0)
        assert result["stats"]["pages_crawled"] <= 1


# ── Job management ───────────────────────────────────────────────────

class TestJobManagement:
    def test_save_and_load(self):
        jobs = [{"url": "https://example.com", "topic": "test"}]
        save_site_jobs(jobs)
        loaded = load_site_jobs()
        assert loaded == jobs

    def test_load_empty(self):
        assert load_site_jobs() == []

    def test_clear_jobs(self):
        save_site_jobs([{"job": 1}])
        clear_site_jobs()
        assert load_site_jobs() == []


class TestAddSiteJob:
    def test_invalid_url(self):
        result = add_site_job("not-valid", "test")
        assert result["status"] == "error"

    def test_valid_url_runs_crawl(self, monkeypatch):
        monkeypatch.setattr(site_crawler, "crawl_site", lambda *a, **kw: {
            "status": "success",
            "entries": 2,
            "entries_data": [
                {"question": "Q1?", "answer": "A1", "domain": "net"},
                {"question": "Q2?", "answer": "A2", "domain": "net"},
            ],
            "stats": {"pages_crawled": 5, "pages_relevant": 3},
        })
        result = add_site_job("https://example.com", "networking")
        assert result["status"] == "success"
        # Job should be persisted
        jobs = load_site_jobs()
        assert len(jobs) == 1


class TestGetSiteCrawlStats:
    def test_empty_stats(self):
        stats = get_site_crawl_stats()
        assert stats["total_jobs"] == 0

    def test_stats_after_jobs(self):
        save_site_jobs([
            {"entries_extracted": 10, "stats": {"pages_crawled": 5}, "domain": "example.com"},
            {"entries_extracted": 20, "stats": {"pages_crawled": 8}, "domain": "other.com"},
        ])
        stats = get_site_crawl_stats()
        assert stats["total_jobs"] == 2
        assert stats["total_entries"] == 30
        assert stats["total_pages_crawled"] == 13
