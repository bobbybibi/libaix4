"""Unit tests for the digest_engine module."""
from __future__ import annotations

from digest_engine import (
    score_entry_quality,
    get_digest_stats,
)


class TestScoreEntryQuality:
    def test_scores_are_between_zero_and_one(self):
        entries = [
            {"question": "What is TCP?", "answer": "TCP is a transport protocol that provides reliable data delivery.", "domain": "networking"},
            {"question": "DNS?", "answer": "DNS.", "domain": "general"},
        ]
        scored = score_entry_quality(entries)
        for e in scored:
            assert 0 <= e["quality_score"] <= 1

    def test_longer_answer_scores_higher(self):
        entries = [
            {"question": "What is TCP?", "answer": "TCP is a transport protocol that provides reliable ordered delivery of data between applications. It uses three-way handshake for connection setup and sequence numbers for ordering.", "domain": "networking"},
            {"question": "What is UDP?", "answer": "UDP sends data.", "domain": "networking"},
        ]
        scored = score_entry_quality(entries)
        assert scored[0]["quality_score"] > scored[1]["quality_score"]

    def test_question_with_question_word_scores_higher(self):
        entries = [
            {"question": "What is a firewall?", "answer": "A firewall is a network security device that monitors traffic.", "domain": "security"},
            {"question": "firewall", "answer": "A firewall is a network security device that monitors traffic.", "domain": "security"},
        ]
        scored = score_entry_quality(entries)
        assert scored[0]["quality_score"] > scored[1]["quality_score"]

    def test_empty_entries_returns_empty(self):
        assert score_entry_quality([]) == []

    def test_preserves_original_fields(self):
        entries = [{"question": "What is X?", "answer": "X is a thing that does stuff.", "domain": "general"}]
        scored = score_entry_quality(entries)
        assert scored[0]["question"] == "What is X?"
        assert scored[0]["answer"] == "X is a thing that does stuff."
        assert scored[0]["domain"] == "general"
        assert "quality_score" in scored[0]


class TestGetDigestStats:
    def test_returns_dict(self):
        stats = get_digest_stats()
        assert isinstance(stats, dict)

    def test_has_expected_keys(self):
        stats = get_digest_stats()
        assert "digest_count" in stats
        assert "quality" in stats
