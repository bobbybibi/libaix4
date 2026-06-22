"""Tests for crawler.py — Wikipedia knowledge crawler (mocked, no network)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


from crawler import (
    _truncate_to_sentence,
    load_config,
    save_config,
    _default_config,
    save_crawled_knowledge,
    crawl_topic,
    crawl_single_topic,
)


class TestTruncateToSentence:
    def test_short_text_unchanged(self):
        assert _truncate_to_sentence("Hello world.", 100) == "Hello world."

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence that is very long."
        result = _truncate_to_sentence(text, 40)
        assert result.endswith(".")
        assert len(result) <= 42  # allow period

    def test_hard_truncate_adds_period(self):
        text = "A" * 200
        result = _truncate_to_sentence(text, 50)
        assert result.endswith(".")
        assert len(result) <= 52


class TestConfig:
    def test_default_config_has_topics(self):
        cfg = _default_config()
        assert "topics" in cfg
        assert len(cfg["topics"]) > 0

    def test_save_and_load_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("crawler.CONFIG_PATH", tmp_path / "cfg.json")
        cfg = _default_config()
        cfg["topics"][0]["name"] = "TestTopic"
        save_config(cfg)
        loaded = load_config()
        assert loaded["topics"][0]["name"] == "TestTopic"

    def test_load_config_defaults_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("crawler.CONFIG_PATH", tmp_path / "nonexistent.json")
        cfg = load_config()
        assert "topics" in cfg


class TestSaveCrawledKnowledge:
    def test_saves_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("crawler.EXTRA_KNOWLEDGE_DIR", tmp_path)
        entries = [{"question": "What is TCP?", "answer": "A protocol.", "domain": "networking"}]
        fp = save_crawled_knowledge(entries, "networking basics")
        assert fp.exists()
        data = json.loads(fp.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["question"] == "What is TCP?"

    def test_filename_contains_topic(self, tmp_path, monkeypatch):
        monkeypatch.setattr("crawler.EXTRA_KNOWLEDGE_DIR", tmp_path)
        fp = save_crawled_knowledge([], "Wi-Fi Security")
        assert "wi-fi_security" in fp.stem


class TestCrawlTopic:
    @patch("crawler.get_article_text", return_value="TCP is a transport protocol that provides reliable data delivery over networks.")
    @patch("crawler.get_article_summary", return_value="TCP (Transmission Control Protocol) is a connection-oriented transport layer protocol that provides reliable data delivery.")
    @patch("crawler.search_wikipedia", return_value=[{"title": "Transmission Control Protocol"}])
    @patch("crawler.time.sleep")
    def test_crawl_returns_entries(self, mock_sleep, mock_search, mock_summary, mock_text):
        entries = crawl_topic("TCP", max_articles=1)
        assert len(entries) >= 1
        assert entries[0]["question"].startswith("What is")
        mock_search.assert_called_once()

    @patch("crawler.search_wikipedia", return_value=[])
    @patch("crawler.time.sleep")
    def test_crawl_empty_results(self, mock_sleep, mock_search):
        entries = crawl_topic("nonexistent_topic_xyz", max_articles=1)
        assert entries == []


class TestCrawlSingleTopic:
    @patch("crawler.crawl_topic", return_value=[
        {"question": "What is X?", "answer": "X is Y.", "domain": "general"}
    ])
    @patch("crawler.save_crawled_knowledge", return_value=Path("fake.json"))
    def test_returns_success(self, mock_save, mock_crawl):
        result = crawl_single_topic("Test Topic")
        assert result["status"] == "success"
        assert result["entries"] == 1

    @patch("crawler.crawl_topic", return_value=[])
    def test_returns_no_results(self, mock_crawl):
        result = crawl_single_topic("Empty Topic")
        assert result["status"] == "no_results"
        assert result["entries"] == 0
