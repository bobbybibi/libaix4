"""Tests for the reasoning_engine module."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reasoning_engine  # noqa: E402
from reasoning_engine import (
    ConceptNode,
    ReasoningEngine,
    ReasoningRule,
    reason_about,
    _cosine_similarity,
    _extract_capitalized,
    _extract_definitions,
    _token_overlap,
)


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Redirect reasoning engine paths to tmp_path."""
    monkeypatch.setattr(reasoning_engine, "REASONING_CONFIG_PATH", tmp_path / "reasoning_config.json")
    monkeypatch.setattr(reasoning_engine, "REASONING_STATE_PATH", tmp_path / "reasoning_state.json")
    # Reset singleton so each test gets a fresh engine
    monkeypatch.setattr(reasoning_engine, "_engine_instance", None)


# ── Dataclass tests ──────────────────────────────────────────────────


class TestReasoningRule:
    def test_create_rule(self):
        rule = ReasoningRule(
            conditions=["A is B"], conclusion="therefore C",
            confidence=0.8, domain="general", source="test",
        )
        assert rule.conditions == ["A is B"]
        assert rule.confidence == 0.8
        assert rule.rule_type == "identity"

    def test_to_dict(self):
        rule = ReasoningRule(
            conditions=["X"], conclusion="Y",
            confidence=0.5, domain="d", source="s",
        )
        d = rule.to_dict()
        assert isinstance(d, dict)
        assert d["conclusion"] == "Y"

    def test_from_dict(self):
        d = {"conditions": ["A"], "conclusion": "B", "confidence": 0.7,
             "domain": "d", "source": "s", "rule_type": "purpose"}
        rule = ReasoningRule.from_dict(d)
        assert rule.rule_type == "purpose"
        assert rule.confidence == 0.7

    def test_roundtrip_dict(self):
        rule = ReasoningRule(
            conditions=["p1", "p2"], conclusion="c",
            confidence=0.9, domain="net", source="q1", rule_type="function",
        )
        restored = ReasoningRule.from_dict(rule.to_dict())
        assert restored.conditions == rule.conditions
        assert restored.conclusion == rule.conclusion


class TestConceptNode:
    def test_create_node(self):
        node = ConceptNode(name="TCP", domain="networking")
        assert node.name == "TCP"
        assert node.related == {}
        assert node.properties == []

    def test_to_dict(self):
        node = ConceptNode(name="DNS", domain="networking",
                           related={"TCP": 0.5}, properties=["protocol"])
        d = node.to_dict()
        assert d["name"] == "DNS"
        assert d["related"]["TCP"] == 0.5

    def test_from_dict(self):
        d = {"name": "HTTP", "domain": "web", "related": {}, "properties": ["stateless"]}
        node = ConceptNode.from_dict(d)
        assert node.name == "HTTP"
        assert "stateless" in node.properties


# ── Utility function tests ───────────────────────────────────────────


class TestUtilities:
    def test_cosine_similarity_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_extract_capitalized(self):
        result = _extract_capitalized("TCP is a Protocol used by HTTP")
        assert "TCP" in result
        assert "HTTP" in result
        assert "Protocol" in result

    def test_extract_definitions(self):
        text = "HTTP is a stateless protocol"
        result = _extract_definitions(text)
        assert len(result) >= 1

    def test_token_overlap_identical(self):
        assert _token_overlap(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_token_overlap_empty(self):
        assert _token_overlap([], ["a"]) == 0.0

    def test_token_overlap_disjoint(self):
        assert _token_overlap(["a", "b"], ["c", "d"]) == 0.0


# ── ReasoningEngine tests ───────────────────────────────────────────


class TestReasoningEngineCreation:
    def test_engine_initialises(self, tmp_path):
        engine = ReasoningEngine()
        assert engine.rules == []
        assert engine.concepts == {}

    def test_build_from_knowledge(self, tmp_path):
        engine = ReasoningEngine()
        stats = engine.build_from_knowledge()
        assert isinstance(stats, dict)
        assert "rules_count" in stats
        assert "concepts_count" in stats
        assert "entries_count" in stats
        assert stats["entries_count"] > 0

    def test_get_stats(self, tmp_path):
        engine = ReasoningEngine()
        engine.build_from_knowledge()
        stats = engine.get_stats()
        assert "rules_count" in stats
        assert "concepts_count" in stats
        assert "cache_size" in stats
        assert "rules_by_type" in stats


class TestDeduce:
    @pytest.fixture()
    def engine(self, tmp_path):
        e = ReasoningEngine()
        e.build_from_knowledge()
        return e

    def test_deduce_returns_list(self, engine):
        results = engine.deduce(["TCP is a transport protocol"])
        assert isinstance(results, list)

    def test_deduce_empty_premises(self, engine):
        results = engine.deduce([])
        assert results == []

    def test_deduce_results_have_expected_keys(self, engine):
        results = engine.deduce(["network protocol security"])
        for r in results:
            assert "conclusion" in r
            assert "confidence" in r
            assert "rule_type" in r
            assert "chain" in r


class TestFindAnalogies:
    @pytest.fixture()
    def engine(self, tmp_path):
        e = ReasoningEngine()
        e.build_from_knowledge()
        return e

    def test_find_analogies_returns_list(self, engine):
        # Use a concept that likely exists from build
        concept = next(iter(engine.concepts), None)
        if concept:
            result = engine.find_analogies(concept)
            assert isinstance(result, list)

    def test_find_analogies_unknown_concept(self, engine):
        result = engine.find_analogies("zzzzz_nonexistent_concept_xyz")
        assert result == []


class TestDetectContradictions:
    @pytest.fixture()
    def engine(self, tmp_path):
        e = ReasoningEngine()
        e.build_from_knowledge()
        return e

    def test_detect_contradictions_returns_list(self, engine):
        result = engine.detect_contradictions()
        assert isinstance(result, list)


class TestSynthesizeAndGeneralize:
    @pytest.fixture()
    def engine(self, tmp_path):
        e = ReasoningEngine()
        e.build_from_knowledge()
        return e

    def test_synthesize_returns_dict(self, engine):
        concepts = list(engine.concepts.keys())[:3]
        if len(concepts) >= 2:
            result = engine.synthesize(concepts)
            assert isinstance(result, dict)
            assert "answer" in result
            assert "confidence" in result

    def test_generalize_with_examples(self, engine):
        examples = [
            {"input": "What is TCP?", "output": "TCP is a transport protocol", "domain": "networking"},
            {"input": "What is HTTP?", "output": "HTTP is a web protocol", "domain": "networking"},
            {"input": "What is DNS?", "output": "DNS is a name resolution protocol", "domain": "networking"},
        ]
        result = engine.generalize(examples)
        assert isinstance(result, list)

    def test_generalize_insufficient_examples(self, engine):
        result = engine.generalize([{"input": "x", "output": "y"}])
        assert result == []


# ── Save / load state ───────────────────────────────────────────────


class TestSaveLoadState:
    def test_save_state_creates_file(self, tmp_path):
        engine = ReasoningEngine()
        engine.build_from_knowledge()
        engine.save_state()
        assert reasoning_engine.REASONING_STATE_PATH.exists()

    def test_load_state_restores_rules_and_concepts(self, tmp_path):
        engine1 = ReasoningEngine()
        engine1.build_from_knowledge()
        n_rules = len(engine1.rules)
        n_concepts = len(engine1.concepts)
        engine1.save_state()

        engine2 = ReasoningEngine()
        engine2.load_state()
        assert len(engine2.rules) == n_rules
        assert len(engine2.concepts) == n_concepts


# ── reason_about convenience ─────────────────────────────────────────


class TestReasonAbout:
    def test_reason_about_returns_dict(self, tmp_path):
        result = reason_about("What is TCP?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "confidence" in result
        assert "strategy" in result
