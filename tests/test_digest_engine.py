"""Unit tests for the digest_engine module."""
from __future__ import annotations

import json

import numpy as np
import pytest

from digest_engine import (
    _cosine_similarity_matrix,
    _default_config,
    _extract_topic,
    _load_all_entries,
    _read_json,
    _timestamp,
    _write_json,
    cross_reference_entries,
    deduplicate_entries,
    generate_derived_knowledge,
    get_digest_stats,
    load_digest_config,
    run_digest_cycle,
    save_digest_config,
    score_entry_quality,
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


# ── Config helpers ───────────────────────────────────────────────────

class TestDigestConfig:
    def test_default_config_keys(self):
        cfg = _default_config()
        assert "dedup_threshold" in cfg
        assert "quality_min_score" in cfg
        assert cfg["digest_count"] == 0

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("digest_engine.DIGEST_CONFIG_PATH", tmp_path / "dc.json")
        cfg = _default_config()
        cfg["digest_count"] = 42
        save_digest_config(cfg)
        loaded = load_digest_config()
        assert loaded["digest_count"] == 42

    def test_load_returns_defaults_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("digest_engine.DIGEST_CONFIG_PATH", tmp_path / "nope.json")
        cfg = load_digest_config()
        assert cfg["digest_count"] == 0


# ── Cosine similarity ────────────────────────────────────────────────

class TestCosineSimilarityMatrix:
    def test_identity_when_same(self):
        v = np.array([[1.0, 0.0], [1.0, 0.0]])
        sim = _cosine_similarity_matrix(v)
        np.testing.assert_almost_equal(sim[0, 1], 1.0)

    def test_orthogonal_is_zero(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = _cosine_similarity_matrix(v)
        np.testing.assert_almost_equal(sim[0, 1], 0.0)

    def test_symmetric(self):
        v = np.array([[0.6, 0.8], [0.8, 0.6]])
        sim = _cosine_similarity_matrix(v)
        np.testing.assert_almost_equal(sim[0, 1], sim[1, 0])


# ── IO helpers ───────────────────────────────────────────────────────

class TestIOHelpers:
    def test_write_and_read_json(self, tmp_path):
        path = tmp_path / "sub" / "test.json"
        data = [{"key": "value"}]
        _write_json(path, data)
        result = _read_json(path)
        assert result == data

    def test_read_json_missing_file(self, tmp_path):
        path = tmp_path / "missing.json"
        assert _read_json(path) == []

    def test_read_json_corrupt(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all{{{", encoding="utf-8")
        assert _read_json(path) == []

    def test_timestamp_format(self):
        ts = _timestamp()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert "_" in ts


# ── Topic extraction ─────────────────────────────────────────────────

class TestExtractTopic:
    def test_strips_question_prefix(self):
        assert _extract_topic("What is TCP?") == "TCP"

    def test_strips_article(self):
        assert _extract_topic("What is a firewall?") == "firewall"

    def test_preserves_core_noun(self):
        assert _extract_topic("Explain the BGP protocol") == "BGP protocol"

    def test_bare_word(self):
        assert _extract_topic("networking") == "networking"


# ── Load all entries ─────────────────────────────────────────────────

class TestLoadAllEntries:
    def test_returns_builtin_entries(self, tmp_path, monkeypatch):
        import digest_engine
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek_empty")
        entries = _load_all_entries()
        assert len(entries) > 0
        assert all("question" in e for e in entries)
        assert all(e["_source"] == "builtin" for e in entries)

    def test_loads_extra_knowledge(self, tmp_path, monkeypatch):
        import digest_engine
        ek = tmp_path / "ek"
        ek.mkdir()
        (ek / "test.json").write_text(json.dumps([
            {"question": "Extra Q?", "answer": "Extra A", "domain": "test"},
        ]), encoding="utf-8")
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", ek)
        entries = _load_all_entries()
        extra = [e for e in entries if e["_source"] != "builtin"]
        assert len(extra) >= 1

    def test_skips_corrupt_files(self, tmp_path, monkeypatch):
        import digest_engine
        ek = tmp_path / "ek_bad"
        ek.mkdir()
        (ek / "bad.json").write_text("NOT JSON", encoding="utf-8")
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", ek)
        entries = _load_all_entries()  # should not raise
        assert isinstance(entries, list)

    def test_skips_dedup_archives(self, tmp_path, monkeypatch):
        import digest_engine
        ek = tmp_path / "ek_dedup"
        ek.mkdir()
        (ek / "digest_dedup_test.json").write_text(json.dumps([
            {"question": "Archived Q?", "answer": "Archived A"},
        ]), encoding="utf-8")
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", ek)
        entries = _load_all_entries()
        extra = [e for e in entries if e.get("_source", "").endswith("digest_dedup_test.json")]
        assert len(extra) == 0


# ── Deduplication ────────────────────────────────────────────────────

class TestDeduplicate:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        import digest_engine
        self._ek = tmp_path / "ek"
        self._ek.mkdir()
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", self._ek)
        monkeypatch.setattr(digest_engine, "DIGEST_CONFIG_PATH", tmp_path / "dc.json")

    def test_no_entries_returns_zero(self, monkeypatch):
        import digest_engine
        monkeypatch.setattr(digest_engine, "_load_all_entries", lambda: [])
        result = deduplicate_entries()
        assert result["duplicates_found"] == 0

    def test_exact_duplicates_detected(self, monkeypatch):
        import digest_engine
        entries = [
            {"question": "What is TCP?", "answer": "Short.", "domain": "net", "_source": "builtin"},
            {"question": "What is TCP?", "answer": "TCP is a transport protocol providing reliable delivery.", "domain": "net", "_source": "builtin"},
        ]
        monkeypatch.setattr(digest_engine, "_load_all_entries", lambda: entries)
        result = deduplicate_entries()
        assert result["duplicates_found"] >= 1


# ── Cross-reference ──────────────────────────────────────────────────

class TestCrossReference:
    def test_returns_structure(self, tmp_path, monkeypatch):
        import digest_engine
        monkeypatch.setattr(digest_engine, "DIGEST_CONFIG_PATH", tmp_path / "dc.json")
        monkeypatch.setattr(digest_engine, "KNOWLEDGE_GRAPH_PATH", tmp_path / "kg.json")
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek_empty")
        result = cross_reference_entries()
        assert result["status"] == "ok"
        assert "total_terms" in result
        assert "clusters" in result

    def test_writes_knowledge_graph(self, tmp_path, monkeypatch):
        import digest_engine
        kg_path = tmp_path / "kg.json"
        monkeypatch.setattr(digest_engine, "DIGEST_CONFIG_PATH", tmp_path / "dc.json")
        monkeypatch.setattr(digest_engine, "KNOWLEDGE_GRAPH_PATH", kg_path)
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek_empty")
        cross_reference_entries()
        assert kg_path.exists()


# ── Derived knowledge ────────────────────────────────────────────────

class TestGenerateDerived:
    def test_returns_structure(self, tmp_path, monkeypatch):
        import digest_engine
        monkeypatch.setattr(digest_engine, "DIGEST_CONFIG_PATH", tmp_path / "dc.json")
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek_empty")
        monkeypatch.setattr(digest_engine, "KNOWLEDGE_GRAPH_PATH", tmp_path / "kg.json")
        result = generate_derived_knowledge()
        assert result["status"] == "ok"
        assert "entries_generated" in result

    def test_with_few_entries(self, tmp_path, monkeypatch):
        import digest_engine
        monkeypatch.setattr(digest_engine, "_load_all_entries", lambda: [
            {"question": "What is TCP?", "answer": "TCP is a transport protocol.", "domain": "net", "_source": "builtin"},
        ])
        monkeypatch.setattr(digest_engine, "DIGEST_CONFIG_PATH", tmp_path / "dc.json")
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek")
        monkeypatch.setattr(digest_engine, "KNOWLEDGE_GRAPH_PATH", tmp_path / "kg.json")
        result = generate_derived_knowledge()
        # Only 1 entry — can't generate comparisons
        assert result["entries_generated"] == 0


# ── Full digest cycle ────────────────────────────────────────────────

class TestRunDigestCycle:
    def test_runs_all_steps(self, tmp_path, monkeypatch):
        import digest_engine
        monkeypatch.setattr(digest_engine, "DIGEST_CONFIG_PATH", tmp_path / "dc.json")
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek_empty")
        monkeypatch.setattr(digest_engine, "KNOWLEDGE_GRAPH_PATH", tmp_path / "kg.json")
        result = run_digest_cycle()
        assert result["status"] == "ok"
        assert len(result["steps"]) == 4
        step_names = [s["step"] for s in result["steps"]]
        assert "deduplicate" in step_names
        assert "quality_score" in step_names
        assert "cross_reference" in step_names
        assert "derive" in step_names
        assert result["elapsed_seconds"] >= 0

    def test_increments_digest_count(self, tmp_path, monkeypatch):
        import digest_engine
        monkeypatch.setattr(digest_engine, "DIGEST_CONFIG_PATH", tmp_path / "dc.json")
        monkeypatch.setattr(digest_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "ek_empty")
        monkeypatch.setattr(digest_engine, "KNOWLEDGE_GRAPH_PATH", tmp_path / "kg.json")
        run_digest_cycle()
        cfg = load_digest_config()
        assert cfg["digest_count"] >= 1
