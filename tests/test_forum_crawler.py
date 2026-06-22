"""Tests for forum_crawler.py — mock all HTTP calls."""
from __future__ import annotations

import json

import pytest

import forum_crawler
from forum_crawler import (
    _strip_html,
    _truncate,
    crawl_stackexchange,
    crawl_reddit,
    crawl_hackernews,
    crawl_devto,
    crawl_forums,
    crawl_single_forum_topic,
    get_learning_stats,
    load_forum_config,
    log_learning_event,
    run_all_forum_crawlers,
    save_forum_config,
    save_forum_knowledge,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_fs(tmp_path, monkeypatch):
    """Redirect all file paths to tmp_path and kill network delay."""
    monkeypatch.setattr(forum_crawler, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek")
    monkeypatch.setattr(forum_crawler, "FORUM_CONFIG_PATH", tmp_path / "fc.json")
    monkeypatch.setattr(forum_crawler, "LEARNING_LOG_PATH", tmp_path / "ll.json")
    monkeypatch.setattr(forum_crawler, "CRAWL_DELAY", 0.0)
    monkeypatch.setattr(forum_crawler.time, "sleep", lambda _: None)


# ── Helpers ──────────────────────────────────────────────────────────

class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>hello</p>") == "hello"

    def test_handles_entities(self):
        assert _strip_html("&amp; &lt;b&gt;") == "& <b>"

    def test_empty(self):
        assert _strip_html("") == ""

    def test_nested_tags(self):
        assert _strip_html("<div><span>hi</span></div>") == "hi"


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncates_at_period(self):
        text = "Hello world. This is a test. And more stuff here."
        result = _truncate(text, 30)
        assert result.endswith(".")
        assert len(result) <= 31

    def test_truncates_with_ellipsis_when_no_period(self):
        text = "abcdefghijklmnop"
        result = _truncate(text, 10)
        assert result.endswith(".")
        assert len(result) <= 11


# ── StackExchange ────────────────────────────────────────────────────

class TestCrawlStackExchange:
    def test_returns_entries_from_mock(self, monkeypatch):
        api_response = {
            "items": [
                {
                    "title": "How does TCP work?",
                    "body": "<p>TCP uses a three-way handshake. SYN, SYN-ACK, ACK.</p>",
                    "question_id": 123,
                },
            ]
        }
        monkeypatch.setattr(forum_crawler, "_http_get_json", lambda url, **kw: api_response)
        monkeypatch.setattr("forum_crawler.classify_domain", lambda t: "networking")
        monkeypatch.setattr("forum_crawler.generate_qa_from_text", lambda t: [])

        entries = crawl_stackexchange("tcp", site="serverfault", max_questions=5)
        assert len(entries) >= 1
        assert entries[0]["question"] == "How does TCP work?"
        assert "TCP uses" in entries[0]["answer"]
        assert entries[0]["domain"] == "networking"

    def test_returns_empty_on_error(self, monkeypatch):
        monkeypatch.setattr(forum_crawler, "_http_get_json", lambda *a, **kw: (_ for _ in ()).throw(OSError("fail")))
        assert crawl_stackexchange("tcp") == []

    def test_deduplicates_by_title(self, monkeypatch):
        api_response = {
            "items": [
                {"title": "Same Question", "body": "Answer A long enough text", "question_id": 1},
                {"title": "Same Question", "body": "Answer B long enough text", "question_id": 2},
            ]
        }
        monkeypatch.setattr(forum_crawler, "_http_get_json", lambda url, **kw: api_response)
        monkeypatch.setattr("forum_crawler.classify_domain", lambda t: "general")
        monkeypatch.setattr("forum_crawler.generate_qa_from_text", lambda t: [])

        entries = crawl_stackexchange("test")
        assert len(entries) == 1


# ── Reddit ───────────────────────────────────────────────────────────

class TestCrawlReddit:
    def test_returns_entries(self, monkeypatch):
        api_response = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Understanding BGP routing",
                            "selftext": "BGP is used for routing between autonomous systems. "
                                        "It's the backbone protocol of the internet and handles path selection.",
                            "id": "abc123",
                            "over_18": False,
                            "quarantine": False,
                        }
                    }
                ]
            }
        }
        monkeypatch.setattr(forum_crawler, "_http_get_json", lambda url, **kw: api_response)
        monkeypatch.setattr("forum_crawler.classify_domain", lambda t: "networking")
        monkeypatch.setattr("forum_crawler.generate_qa_from_text", lambda t: [])

        entries = crawl_reddit("bgp", subreddit="networking")
        assert len(entries) >= 1
        assert "BGP" in entries[0]["answer"]

    def test_skips_nsfw(self, monkeypatch):
        api_response = {
            "data": {
                "children": [
                    {"data": {"title": "Bad post", "selftext": "x" * 60, "over_18": True, "id": "1"}},
                ]
            }
        }
        monkeypatch.setattr(forum_crawler, "_http_get_json", lambda url, **kw: api_response)
        assert crawl_reddit("test") == []

    def test_skips_short_selftext(self, monkeypatch):
        api_response = {
            "data": {
                "children": [
                    {"data": {"title": "Short post", "selftext": "hi", "over_18": False, "id": "1"}},
                ]
            }
        }
        monkeypatch.setattr(forum_crawler, "_http_get_json", lambda url, **kw: api_response)
        assert crawl_reddit("test") == []


# ── Hacker News ──────────────────────────────────────────────────────

class TestCrawlHackerNews:
    def test_returns_entries_with_story_text(self, monkeypatch):
        api_response = {
            "hits": [
                {
                    "title": "Why Kubernetes matters",
                    "story_text": "<p>Kubernetes provides container orchestration capabilities "
                                  "that simplify deployment and scaling of applications.</p>",
                    "objectID": "999",
                },
            ]
        }
        monkeypatch.setattr(forum_crawler, "_http_get_json", lambda url, **kw: api_response)
        monkeypatch.setattr("forum_crawler.classify_domain", lambda t: "cloud")
        monkeypatch.setattr("forum_crawler.generate_qa_from_text", lambda t: [])

        entries = crawl_hackernews("kubernetes")
        assert len(entries) >= 1

    def test_returns_empty_on_error(self, monkeypatch):
        monkeypatch.setattr(forum_crawler, "_http_get_json", lambda *a, **kw: (_ for _ in ()).throw(OSError("fail")))
        assert crawl_hackernews("kubernetes") == []


# ── DEV.to ───────────────────────────────────────────────────────────

class TestCrawlDevto:
    def test_returns_entries(self, monkeypatch):
        articles = [
            {
                "title": "Introduction to Docker",
                "description": "Docker containers allow you to package applications and their dependencies together for deployment.",
                "body_markdown": "Docker containers allow you to package applications.",
                "id": 42,
            },
        ]
        monkeypatch.setattr(forum_crawler, "_http_get", lambda url, **kw: json.dumps(articles))
        monkeypatch.setattr("forum_crawler.classify_domain", lambda t: "devops")

        entries = crawl_devto("docker")
        assert len(entries) >= 1

    def test_returns_empty_on_error(self, monkeypatch):
        monkeypatch.setattr(forum_crawler, "_http_get", lambda *a, **kw: (_ for _ in ()).throw(OSError("fail")))
        assert crawl_devto("docker") == []


# ── Config management ────────────────────────────────────────────────

class TestForumConfig:
    def test_load_default(self):
        cfg = load_forum_config()
        assert "topics" in cfg

    def test_save_and_load(self, tmp_path, monkeypatch):
        path = tmp_path / "fc2.json"
        monkeypatch.setattr(forum_crawler, "FORUM_CONFIG_PATH", path)
        save_forum_config({"topics": [], "test": True})
        loaded = load_forum_config()
        assert loaded["test"] is True


# ── Learning log ─────────────────────────────────────────────────────

class TestLearningLog:
    def test_log_and_stats(self):
        log_learning_event("test_source", "test_topic", 10)
        log_learning_event("test_source", "test_topic", 5)
        stats = get_learning_stats()
        assert stats["total_entries_learned"] == 15
        assert stats["total_events"] == 2
        assert "test_source" in stats["source_stats"]


# ── Save knowledge ───────────────────────────────────────────────────

class TestSaveForumKnowledge:
    def test_saves_json_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(forum_crawler, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek")
        entries = [{"question": "Q?", "answer": "A", "domain": "general"}]
        fp = save_forum_knowledge(entries, "test_topic")
        assert fp.exists()
        data = json.loads(fp.read_text())
        assert len(data) == 1


# ── Combined crawl ───────────────────────────────────────────────────

class TestCrawlForums:
    def test_calls_enabled_sources(self, monkeypatch):
        called = set()

        def mock_se(*a, **kw):
            called.add("stackexchange")
            return [{"question": "SE Q?", "answer": "SE A", "domain": "net", "source": "se:1"}]

        def mock_reddit(*a, **kw):
            called.add("reddit")
            return [{"question": "Reddit Q?", "answer": "Reddit A", "domain": "net", "source": "r:1"}]

        monkeypatch.setattr(forum_crawler, "crawl_stackexchange", mock_se)
        monkeypatch.setattr(forum_crawler, "crawl_reddit", mock_reddit)
        monkeypatch.setattr(forum_crawler, "crawl_hackernews", lambda *a, **kw: [])
        monkeypatch.setattr(forum_crawler, "crawl_devto", lambda *a, **kw: [])

        entries = crawl_forums("test", sources=["stackexchange", "reddit"])
        assert "stackexchange" in called
        assert "reddit" in called
        assert len(entries) >= 2


class TestRunAllForumCrawlers:
    def test_runs_enabled_topics(self, monkeypatch):
        monkeypatch.setattr(forum_crawler, "crawl_forums", lambda *a, **kw: [
            {"question": "Q?", "answer": "A long enough answer for testing.", "domain": "net", "source": "test:1"},
        ])
        result = run_all_forum_crawlers()
        assert "topics" in result
        assert "total_new_entries" in result


class TestCrawlSingleForumTopic:
    def test_returns_results(self, monkeypatch):
        monkeypatch.setattr(forum_crawler, "crawl_forums", lambda *a, **kw: [
            {"question": "Q?", "answer": "A long enough answer.", "domain": "net", "source": "test:1"},
        ])
        result = crawl_single_forum_topic("test_topic")
        assert result["status"] == "success"
        assert result["entries"] >= 1

    def test_returns_no_results(self, monkeypatch):
        monkeypatch.setattr(forum_crawler, "crawl_forums", lambda *a, **kw: [])
        result = crawl_single_forum_topic("empty_topic")
        assert result["status"] == "no_results"
