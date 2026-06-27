"""Tests for SaaS API v1 endpoints."""

from __future__ import annotations

import uuid

from app import app


def _client():
    app.config["TESTING"] = True
    return app.test_client()


def test_api_v1_health():
    client = _client()
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["version"] == "v1"
    assert "db" in data


def test_api_v1_trade_packs_and_config():
    client = _client()

    r = client.get("/api/v1/trade-packs")
    assert r.status_code == 200
    packs = r.get_json()
    assert "trades" in packs
    assert isinstance(packs["trades"], list)

    cfg = client.get("/api/v1/config")
    assert cfg.status_code == 200
    c = cfg.get_json()
    assert "active_trade" in c


def test_api_v1_register_login_me_and_billing():
    client = _client()
    email = f"saas-{uuid.uuid4().hex[:10]}@example.com"

    reg = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "strongpass123",
            "tenant_name": "Acme Trade Co",
            "trade_id": "networking",
        },
    )
    assert reg.status_code == 201

    me = client.get("/api/v1/me")
    assert me.status_code == 200
    me_data = me.get_json()
    assert me_data["tenant"]["plan"] == "free"

    plans = client.get("/api/v1/billing/plans")
    assert plans.status_code == 200
    assert "pro" in plans.get_json()["plans"]

    sub = client.post("/api/v1/billing/subscribe", json={"plan": "pro"})
    assert sub.status_code == 200
    assert sub.get_json()["plan"] == "pro"


def test_api_v1_onboarding_requires_auth():
    client = _client()
    r = client.post("/api/v1/onboarding/select-trade", json={"trade_id": "plumbing"})
    assert r.status_code == 401


def test_api_v1_analytics_feedback():
    client = _client()
    r = client.post(
        "/api/v1/analytics/query-feedback",
        json={
            "question": "How do I fix a leaking pipe?",
            "domain": "plumbing",
            "confidence": 0.05,
            "answered": False,
            "strategy": "fallback",
        },
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["outcome"] == "unanswered"
