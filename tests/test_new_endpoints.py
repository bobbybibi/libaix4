"""Tests for the new Flask API endpoints (boil, reasoning, anon, forms, stats)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import boil_engine   # noqa: E402
import anon_crawler   # noqa: E402
import reasoning_engine  # noqa: E402
import form_filler    # noqa: E402
from app import app   # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Redirect all file-based paths to tmp_path for test isolation."""
    # boil_engine
    monkeypatch.setattr(boil_engine, "BOIL_CONFIG_PATH", tmp_path / "boil_config.json")
    monkeypatch.setattr(boil_engine, "BOIL_STATE_PATH", tmp_path / "boil_state.json")
    monkeypatch.setattr(boil_engine, "BOIL_LOG_PATH", tmp_path / "boil_log.json")
    monkeypatch.setattr(boil_engine, "GLOSSARY_PATH", tmp_path / "boil_glossary.json")
    monkeypatch.setattr(boil_engine, "REASONING_PATH", tmp_path / "boil_reasoning.json")
    monkeypatch.setattr(boil_engine, "KNOWLEDGE_GRAPH_PATH", tmp_path / "knowledge_graph.json")
    monkeypatch.setattr(boil_engine, "EXTRA_KNOWLEDGE_DIR", tmp_path / "extra_knowledge")
    monkeypatch.setattr(boil_engine, "MODEL_DIR", tmp_path / "models")
    # anon_crawler
    monkeypatch.setattr(anon_crawler, "ANON_CONFIG_PATH", tmp_path / "anon_config.json")
    monkeypatch.setattr(anon_crawler, "ANON_STATS_PATH", tmp_path / "anon_stats.json")
    monkeypatch.setattr(anon_crawler, "_stats", {
        "total_requests": 0, "successful_requests": 0,
        "failed_requests": 0, "bytes_downloaded": 0,
        "pages_crawled": 0, "entries_extracted": 0,
        "domains_visited": [], "last_request_at": None,
    })
    # reasoning_engine
    monkeypatch.setattr(reasoning_engine, "REASONING_CONFIG_PATH", tmp_path / "reasoning_config.json")
    monkeypatch.setattr(reasoning_engine, "REASONING_STATE_PATH", tmp_path / "reasoning_state.json")
    monkeypatch.setattr(reasoning_engine, "_engine_instance", None)
    # form_filler
    monkeypatch.setattr(form_filler, "FORM_CONFIG_PATH", tmp_path / "form_config.json")
    monkeypatch.setattr(form_filler, "FORM_HISTORY_PATH", tmp_path / "form_history.json")
    monkeypatch.setattr(form_filler, "FORM_TEMPLATES_PATH", tmp_path / "form_templates.json")
    monkeypatch.setattr(form_filler, "FORM_PROFILES_PATH", tmp_path / "form_profiles.json")


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture()
def auth_client(client):
    """Login to admin and return authenticated client."""
    client.post("/admin/login", data={"username": "testadmin", "password": "testpass123"})
    return client


# ── Boil endpoints ──────────────────────────────────────────────────


class TestBoilEndpoints:
    def test_boil_status(self, client):
        resp = client.get("/boil/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "boiling" in data
        assert "state" in data
        assert "config" in data

    def test_boil_start(self, auth_client, monkeypatch):
        monkeypatch.setattr(boil_engine, "_boil_thread", None)
        boil_engine._stop_event.clear()
        resp = auth_client.post("/boil/start")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "started" in data
        # Cleanup: stop the thread
        boil_engine.stop_boil_background()

    def test_boil_stop(self, auth_client, monkeypatch):
        monkeypatch.setattr(boil_engine, "_boil_thread", None)
        boil_engine._stop_event.clear()
        boil_engine.start_boil_background()
        resp = auth_client.post("/boil/stop")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "stopped" in data

    def test_boil_tick(self, auth_client):
        resp = auth_client.post("/boil/tick")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "mechanism" in data
        assert "improvements" in data

    def test_boil_log(self, client):
        resp = client.get("/boil/log")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_boil_log_with_n_param(self, auth_client):
        # Run a tick first to ensure there's log content
        auth_client.post("/boil/tick")
        resp = auth_client.get("/boil/log?n=5")
        assert resp.status_code == 200

    def test_boil_config_get(self, client):
        resp = client.get("/boil/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "enabled" in data

    def test_boil_config_post(self, client):
        resp = client.post("/boil/config", json={"tick_interval_seconds": 60})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["saved"] is True
        assert data["config"]["tick_interval_seconds"] == 60


# ── Reason endpoints ────────────────────────────────────────────────


class TestReasonEndpoints:
    def test_reason_post(self, client):
        resp = client.post("/reason", json={"question": "What is TCP?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "answer" in data
        assert "confidence" in data
        assert "strategy" in data

    def test_reason_missing_question(self, client):
        resp = client.post("/reason", json={})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_reason_empty_question(self, client):
        resp = client.post("/reason", json={"question": ""})
        assert resp.status_code == 400

    def test_reason_stats(self, client):
        resp = client.get("/reason/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "rules_count" in data
        assert "concepts_count" in data

    def test_reason_rebuild(self, auth_client):
        resp = auth_client.post("/reason/rebuild")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "rules_count" in data
        assert "entries_count" in data


# ── Anon endpoints ──────────────────────────────────────────────────


class TestAnonEndpoints:
    def test_anon_stats(self, client):
        resp = client.get("/anon/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "total_requests" in data

    def test_anon_crawl_missing_url(self, auth_client):
        resp = auth_client.post("/anon/crawl", json={"topic": "test"})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_anon_crawl_empty_url(self, auth_client):
        resp = auth_client.post("/anon/crawl", json={"url": "", "topic": "test"})
        assert resp.status_code == 400

    def test_anon_crawl_unsafe_url(self, auth_client):
        """Crawling an internal IP should be blocked by SSRF protection."""
        resp = auth_client.post("/anon/crawl", json={"url": "http://127.0.0.1/secret", "topic": "test"})
        # SSRF protection returns 403
        assert resp.status_code == 403

    def test_anon_config_get(self, client):
        resp = client.get("/anon/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "delay_min" in data

    def test_anon_config_post(self, client):
        resp = client.post("/anon/config", json={"delay_min": 3.0})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["saved"] is True


# ── Forms endpoints ─────────────────────────────────────────────────


SAMPLE_FORM_HTML = """
<form action="/submit" method="POST">
  <input type="text" name="name" required>
  <input type="email" name="email">
  <input type="hidden" name="csrf_token" value="tok123">
</form>
"""


class TestFormsEndpoints:
    def test_forms_extract_with_html(self, client):
        resp = client.post("/forms/extract", json={"html": SAMPLE_FORM_HTML, "url": "https://example.com"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "forms" in data
        assert data["count"] >= 1
        fields = data["forms"][0]["fields"]
        names = [f["name"] for f in fields]
        assert "name" in names
        assert "email" in names

    def test_forms_extract_empty(self, client):
        resp = client.post("/forms/extract", json={"html": "", "url": ""})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_forms_fill_with_values(self, client):
        resp = client.post("/forms/fill", json={"values": {"name": "Alice", "email": "a@b.c"}})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ready"
        assert data["parsed_values"]["name"] == "Alice"

    def test_forms_fill_with_prompt(self, client):
        resp = client.post("/forms/fill", json={"prompt": 'name=Bob email=bob@x.com'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["parsed_values"]["name"] == "Bob"

    def test_forms_fill_no_values(self, client):
        resp = client.post("/forms/fill", json={})
        assert resp.status_code == 400

    def test_forms_profiles_get_empty(self, client):
        resp = client.get("/forms/profiles")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_forms_profiles_create(self, client):
        resp = client.post("/forms/profiles", json={
            "name": "work",
            "mappings": {"email": "work@co.com"},
        })
        assert resp.status_code == 200
        assert resp.get_json()["saved"] is True

    def test_forms_profiles_create_missing_name(self, client):
        resp = client.post("/forms/profiles", json={"mappings": {"a": "b"}})
        assert resp.status_code == 400

    def test_forms_profiles_list_after_create(self, client):
        client.post("/forms/profiles", json={"name": "p1", "mappings": {"email": "x@x.com"}})
        resp = client.get("/forms/profiles")
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["name"] == "p1"

    def test_forms_templates_empty(self, client):
        resp = client.get("/forms/templates")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_forms_history_empty(self, client):
        resp = client.get("/forms/history")
        assert resp.status_code == 200
        assert resp.get_json() == []


# ── Stats/all endpoint ──────────────────────────────────────────────


class TestStatsAll:
    def test_stats_all(self, client):
        resp = client.get("/stats/all")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "knowledge" in data
        assert "models" in data
        assert "boil" in data or data.get("boil") is None
        assert "reasoning" in data or data.get("reasoning") is None
        assert "anon" in data or data.get("anon") is None
