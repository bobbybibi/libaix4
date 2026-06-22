"""Tests for the boil_engine module."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import boil_engine  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Redirect all boil engine paths to tmp_path so tests are hermetic."""
    monkeypatch.setattr(boil_engine, "BOIL_CONFIG_PATH", tmp_path / "boil_config.json")
    monkeypatch.setattr(boil_engine, "BOIL_STATE_PATH", tmp_path / "boil_state.json")
    monkeypatch.setattr(boil_engine, "BOIL_LOG_PATH", tmp_path / "boil_log.json")
    monkeypatch.setattr(boil_engine, "GLOSSARY_PATH", tmp_path / "boil_glossary.json")
    monkeypatch.setattr(boil_engine, "REASONING_PATH", tmp_path / "boil_reasoning.json")
    monkeypatch.setattr(boil_engine, "KNOWLEDGE_GRAPH_PATH", tmp_path / "knowledge_graph.json")
    monkeypatch.setattr(boil_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "extra_knowledge")
    monkeypatch.setattr(boil_engine, "MODEL_DIR", tmp_path / "models")


# ── Config tests ─────────────────────────────────────────────────────


class TestBoilConfig:
    def test_default_config_has_expected_keys(self):
        cfg = boil_engine._default_config()
        assert "enabled" in cfg
        assert "tick_interval_seconds" in cfg
        assert "max_cycle_seconds" in cfg
        assert "mechanisms_per_tick" in cfg
        assert "mechanism_weights" in cfg
        assert "cooldowns" in cfg
        assert "created_at" in cfg

    def test_default_config_enabled_true(self):
        cfg = boil_engine._default_config()
        assert cfg["enabled"] is True

    def test_load_config_creates_default_when_missing(self, tmp_path):
        cfg = boil_engine.load_boil_config()
        assert cfg["enabled"] is True
        assert boil_engine.BOIL_CONFIG_PATH.exists()

    def test_save_and_load_config_roundtrip(self, tmp_path):
        cfg = boil_engine._default_config()
        cfg["tick_interval_seconds"] = 99
        boil_engine.save_boil_config(cfg)
        loaded = boil_engine.load_boil_config()
        assert loaded["tick_interval_seconds"] == 99

    def test_load_config_handles_corrupt_json(self, tmp_path):
        boil_engine.BOIL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        boil_engine.BOIL_CONFIG_PATH.write_text("NOT-JSON", encoding="utf-8")
        cfg = boil_engine.load_boil_config()
        # Should fall back to defaults
        assert cfg["enabled"] is True


# ── State tests ──────────────────────────────────────────────────────


class TestBoilState:
    def test_default_state_has_expected_keys(self):
        state = boil_engine._default_state()
        assert state["total_ticks"] == 0
        assert state["total_improvements"] == 0
        assert state["last_tick"] is None
        assert state["started_at"] is None

    def test_get_boil_state_returns_default(self, tmp_path):
        state = boil_engine.get_boil_state()
        assert state["total_ticks"] == 0

    def test_save_and_get_state_roundtrip(self, tmp_path):
        state = boil_engine._default_state()
        state["total_ticks"] = 42
        boil_engine._save_state(state)
        loaded = boil_engine.get_boil_state()
        assert loaded["total_ticks"] == 42


# ── Logging tests ────────────────────────────────────────────────────


class TestBoilLog:
    def test_get_improvement_log_empty_when_no_file(self, tmp_path):
        assert boil_engine.get_improvement_log() == []

    def test_append_and_get_log(self, tmp_path):
        boil_engine._append_log({"mechanism": "test", "improvements": 1, "details": "hello"})
        logs = boil_engine.get_improvement_log(10)
        assert len(logs) == 1
        assert logs[0]["mechanism"] == "test"

    def test_log_caps_at_500(self, tmp_path):
        for i in range(510):
            boil_engine._append_log({"mechanism": "x", "improvements": i})
        raw = json.loads(boil_engine.BOIL_LOG_PATH.read_text(encoding="utf-8"))
        assert len(raw) == 500


# ── Mechanism registry ───────────────────────────────────────────────


class TestMechanismRegistry:
    def test_mechanism_names_has_45_entries(self):
        assert len(boil_engine.MECHANISM_NAMES) == 45

    def test_all_mechanism_names_are_strings(self):
        for name in boil_engine.MECHANISM_NAMES:
            assert isinstance(name, str) and len(name) > 0


# ── Tick / boiling state ─────────────────────────────────────────────


class TestRunBoilTick:
    def test_run_boil_tick_returns_dict(self, tmp_path):
        result = boil_engine.run_boil_tick()
        assert isinstance(result, dict)
        assert "mechanism" in result
        assert "improvements" in result
        assert "details" in result
        assert "timestamp" in result

    def test_run_boil_tick_increments_state(self, tmp_path):
        boil_engine.run_boil_tick()
        state = boil_engine.get_boil_state()
        assert state["total_ticks"] >= 1

    def test_run_boil_tick_writes_log(self, tmp_path):
        boil_engine.run_boil_tick()
        logs = boil_engine.get_improvement_log(10)
        assert len(logs) >= 1


class TestIsBoiling:
    def test_is_boiling_false_by_default(self, monkeypatch):
        monkeypatch.setattr(boil_engine, "_boil_thread", None)
        assert boil_engine.is_boiling() is False

    def test_start_and_stop_background(self, tmp_path, monkeypatch):
        # Ensure clean state
        monkeypatch.setattr(boil_engine, "_boil_thread", None)
        boil_engine._stop_event.clear()
        started = boil_engine.start_boil_background()
        assert started is True
        assert boil_engine.is_boiling() is True
        stopped = boil_engine.stop_boil_background()
        assert stopped is True
        assert boil_engine.is_boiling() is False

    def test_start_when_already_running_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(boil_engine, "_boil_thread", None)
        boil_engine._stop_event.clear()
        boil_engine.start_boil_background()
        assert boil_engine.start_boil_background() is False
        boil_engine.stop_boil_background()

    def test_stop_when_not_running_returns_false(self, monkeypatch):
        monkeypatch.setattr(boil_engine, "_boil_thread", None)
        assert boil_engine.stop_boil_background() is False


# ── Individual mechanism helpers ─────────────────────────────────────


class TestMechanismOutputs:
    def test_result_helper(self):
        r = boil_engine._result("my_mech", 5, "did things")
        assert r["mechanism"] == "my_mech"
        assert r["improvements"] == 5
        assert r["details"] == "did things"
        assert "timestamp" in r
