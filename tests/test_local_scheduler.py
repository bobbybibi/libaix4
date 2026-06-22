"""Tests for local_scheduler.py — cron-like local job scheduler (mocked, no real jobs)."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from local_scheduler import (
    _load_cron_config,
    _load_state,
    _save_state,
    _should_run,
    get_scheduler_status,
    run_once,
    start_scheduler,
    stop_scheduler,
    JOB_RUNNERS,
)


class TestCronConfig:
    def test_default_config_has_jobs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("local_scheduler.CRON_CONFIG_PATH", tmp_path / "missing.json")
        cfg = _load_cron_config()
        assert "jobs" in cfg
        assert "auto_train" in cfg["jobs"]

    def test_loads_from_file(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "cron.json"
        cfg_path.write_text(json.dumps({"jobs": {"test_job": {"enabled": True}}}), encoding="utf-8")
        monkeypatch.setattr("local_scheduler.CRON_CONFIG_PATH", cfg_path)
        cfg = _load_cron_config()
        assert "test_job" in cfg["jobs"]


class TestState:
    def test_default_state_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("local_scheduler.SCHEDULER_STATE_PATH", tmp_path / "missing.json")
        state = _load_state()
        assert state["last_runs"] == {}
        assert state["run_counts"] == {}

    def test_save_and_load(self, tmp_path, monkeypatch):
        state_path = tmp_path / "state.json"
        monkeypatch.setattr("local_scheduler.SCHEDULER_STATE_PATH", state_path)
        _save_state({"last_runs": {"x": "2026-01-01"}, "run_counts": {"x": 5}, "errors": {}, "started_at": None})
        loaded = _load_state()
        assert loaded["last_runs"]["x"] == "2026-01-01"
        assert loaded["run_counts"]["x"] == 5


class TestShouldRun:
    def test_first_run_always_true(self):
        assert _should_run("test_job", 4, {"last_runs": {}}) is True

    def test_recently_run_returns_false(self):
        now = datetime.now(timezone.utc).isoformat()
        state = {"last_runs": {"test_job": now}}
        assert _should_run("test_job", 4, state) is False

    def test_old_run_returns_true(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        state = {"last_runs": {"test_job": old}}
        assert _should_run("test_job", 4, state) is True

    def test_zero_runs_per_hour_false(self):
        assert _should_run("test_job", 0, {"last_runs": {}}) is False

    def test_negative_runs_per_hour_false(self):
        assert _should_run("test_job", -1, {"last_runs": {}}) is False


class TestRunOnce:
    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("local_scheduler.CRON_CONFIG_PATH", tmp_path / "cron.json")
        monkeypatch.setattr("local_scheduler.SCHEDULER_STATE_PATH", tmp_path / "state.json")
        # Config with one fake enabled job
        cfg = {"jobs": {name: {"enabled": False, "runs_per_hour": 0} for name in JOB_RUNNERS}}
        (tmp_path / "cron.json").write_text(json.dumps(cfg), encoding="utf-8")

    def test_all_disabled_returns_disabled(self):
        results = run_once()
        for name, result in results.items():
            assert result["status"] == "disabled"

    @patch("local_scheduler.JOB_RUNNERS", {"auto_train": lambda: {"status": "success"}})
    def test_enabled_job_runs(self, tmp_path):
        cfg = {"jobs": {"auto_train": {"enabled": True, "runs_per_hour": 1}}}
        (tmp_path / "cron.json").write_text(json.dumps(cfg), encoding="utf-8")
        results = run_once()
        assert results["auto_train"]["status"] == "success"


class TestSchedulerStatus:
    def test_returns_status_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("local_scheduler.CRON_CONFIG_PATH", tmp_path / "cron.json")
        monkeypatch.setattr("local_scheduler.SCHEDULER_STATE_PATH", tmp_path / "state.json")
        status = get_scheduler_status()
        assert "running" in status
        assert "jobs" in status
        assert isinstance(status["running"], bool)


class TestStartStopScheduler:
    def test_start_and_stop(self, tmp_path, monkeypatch):
        monkeypatch.setattr("local_scheduler.CRON_CONFIG_PATH", tmp_path / "cron.json")
        monkeypatch.setattr("local_scheduler.SCHEDULER_STATE_PATH", tmp_path / "state.json")
        # All jobs disabled so the loop exits quickly
        cfg = {"jobs": {name: {"enabled": False, "runs_per_hour": 0} for name in JOB_RUNNERS}}
        (tmp_path / "cron.json").write_text(json.dumps(cfg), encoding="utf-8")
        start_scheduler()
        status = get_scheduler_status()
        assert status["running"] is True
        stop_scheduler()

    def test_double_start_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.setattr("local_scheduler.CRON_CONFIG_PATH", tmp_path / "cron.json")
        monkeypatch.setattr("local_scheduler.SCHEDULER_STATE_PATH", tmp_path / "state.json")
        cfg = {"jobs": {name: {"enabled": False, "runs_per_hour": 0} for name in JOB_RUNNERS}}
        (tmp_path / "cron.json").write_text(json.dumps(cfg), encoding="utf-8")
        start_scheduler()
        start_scheduler()  # Should not raise
        stop_scheduler()
