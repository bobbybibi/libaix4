"""Tests for training-data scaling safeguards in train_knowledge.

The crawlers produce a corpus that is ~99% exact duplicates; these tests cover
the deduplication that keeps training tractable.
"""

from __future__ import annotations

from train_knowledge import dedupe_entries


def test_dedupe_removes_exact_qa_duplicates():
    entries = [
        ("What is TCP?", "A transport protocol.", "networking"),
        ("What is TCP?", "A transport protocol.", "networking"),  # exact dup
        ("What is TCP?", "A transport protocol.", "internet"),    # dup (q,a) diff domain
        ("What is UDP?", "A datagram protocol.", "networking"),
    ]
    out = dedupe_entries(entries)
    assert len(out) == 2
    assert out[0] == ("What is TCP?", "A transport protocol.", "networking")
    assert out[1] == ("What is UDP?", "A datagram protocol.", "networking")


def test_dedupe_keeps_same_question_with_different_answer():
    # Same question, different answer → both kept (distinct (q, a) pairs).
    entries = [
        ("Define DNS", "Maps names to IPs.", "internet"),
        ("Define DNS", "A hierarchical naming system.", "internet"),
    ]
    out = dedupe_entries(entries)
    assert len(out) == 2


def test_dedupe_preserves_first_seen_order():
    entries = [
        ("q3", "a3", "d"),
        ("q1", "a1", "d"),
        ("q3", "a3", "d"),  # dup of first
        ("q2", "a2", "d"),
    ]
    out = [q for q, _, _ in dedupe_entries(entries)]
    assert out == ["q3", "q1", "q2"]


def test_dedupe_empty():
    assert dedupe_entries([]) == []


def test_dedupe_is_massively_effective_on_repeated_corpus():
    # Simulate the crawler re-emitting the same 3 entries many times.
    unique = [
        ("q1", "a1", "d"),
        ("q2", "a2", "d"),
        ("q3", "a3", "d"),
    ]
    bloated = unique * 1000
    assert len(bloated) == 3000
    assert len(dedupe_entries(bloated)) == 3
