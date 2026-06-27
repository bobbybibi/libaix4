"""Tests for the zero-training TF-IDF cosine retrieval engine."""

from __future__ import annotations

import pytest

from retrieval import KnowledgeRetriever, dedupe_entries

ENTRIES = [
    ("What is TCP?", "TCP is a reliable, connection-oriented transport protocol.", "networking"),
    ("What is UDP?", "UDP is a connectionless, best-effort transport protocol.", "networking"),
    ("What is DNS?", "DNS resolves domain names to IP addresses.", "internet"),
    ("What is a firewall?", "A firewall filters network traffic by policy.", "security"),
    ("What is WPA3?", "WPA3 is the latest Wi-Fi security standard with SAE.", "wifi_security"),
]


def test_dedupe_entries_collapses_duplicate_qa():
    entries = ENTRIES + [ENTRIES[0], ("What is TCP?", ENTRIES[0][1], "internet")]
    assert len(dedupe_entries(entries)) == len(ENTRIES)


def test_fit_size_reflects_unique_entries():
    r = KnowledgeRetriever.fit(ENTRIES + ENTRIES)  # duplicated input
    assert r.size == len(ENTRIES)


def test_query_returns_most_similar_answer():
    r = KnowledgeRetriever.fit(ENTRIES)
    best = r.best("tell me about tcp")
    assert best is not None
    assert "TCP" in best["answer"]
    assert best["domain"] == "networking"


def test_query_firewall_matches_security():
    r = KnowledgeRetriever.fit(ENTRIES)
    best = r.best("how does a firewall work")
    assert "firewall" in best["answer"].lower()
    assert best["domain"] == "security"


def test_exact_question_scores_near_one():
    r = KnowledgeRetriever.fit(ENTRIES)
    best = r.best("What is DNS?")
    assert best["answer"].startswith("DNS resolves")
    assert best["score"] == pytest.approx(1.0, abs=1e-4)


def test_top_k_ordering_descending():
    r = KnowledgeRetriever.fit(ENTRIES)
    results = r.query("what is a transport protocol tcp udp", top_k=3)
    assert len(results) == 3
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_empty():
    r = KnowledgeRetriever.fit(ENTRIES)
    assert r.query("") == []
    assert r.query("   ") == []
    assert r.best("") is None


def test_fit_rejects_empty_corpus():
    with pytest.raises(ValueError):
        KnowledgeRetriever.fit([])


def test_save_load_round_trip(tmp_path):
    r = KnowledgeRetriever.fit(ENTRIES)
    r.save(tmp_path / "idx")
    loaded = KnowledgeRetriever.load(tmp_path / "idx")
    assert loaded.size == r.size
    before = r.best("what is dns")
    after = loaded.best("what is dns")
    assert before["answer"] == after["answer"]
    assert after["score"] == pytest.approx(before["score"], abs=1e-5)


def test_top_k_capped_to_corpus_size():
    r = KnowledgeRetriever.fit(ENTRIES)
    results = r.query("network", top_k=100)
    assert len(results) <= r.size
