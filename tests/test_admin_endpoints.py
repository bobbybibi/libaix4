"""Tests for admin API endpoints using Flask's test client."""
from __future__ import annotations

import os
import uuid

import pytest

# Set admin credentials before importing app (which imports admin)
os.environ.setdefault("ADMIN_USER", "testadmin")
os.environ.setdefault("ADMIN_PASS", "testpass123")

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def auth_client(client):
    """Login to admin and return authenticated client."""
    client.post("/admin/login", data={"username": "testadmin", "password": "testpass123"})
    return client


class TestPublicRoutes:
    def test_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_datasets(self, client):
        resp = client.get("/datasets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "xor" in data

    def test_knowledge_stats(self, client):
        resp = client.get("/knowledge/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "loaded" in data
        assert "domains" in data


class TestAdminAuth:
    def test_dashboard_redirects_without_login(self, client):
        resp = client.get("/admin/")
        assert resp.status_code == 302

    def test_login_success(self, client):
        resp = client.post("/admin/login", data={"username": "testadmin", "password": "testpass123"}, follow_redirects=True)
        assert resp.status_code == 200

    def test_login_failure(self, client):
        resp = client.post("/admin/login", data={"username": "wrong", "password": "wrong"})
        assert resp.status_code == 200  # stays on login page


class TestLearningTopics:
    def test_get_topics(self, auth_client):
        resp = auth_client.get("/admin/learning-topics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "topics" in data

    def test_add_topic(self, auth_client):
        unique = f"test_topic_{uuid.uuid4().hex[:8]}"
        resp = auth_client.post("/admin/learning-topics",
            json={"name": unique, "priority": "low"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_add_duplicate_topic(self, auth_client):
        unique = f"dup_test_{uuid.uuid4().hex[:8]}"
        auth_client.post("/admin/learning-topics",
            json={"name": unique, "priority": "low"})
        resp = auth_client.post("/admin/learning-topics",
            json={"name": unique, "priority": "high"})
        assert resp.status_code == 409

    def test_toggle_topic(self, auth_client):
        unique = f"toggle_test_{uuid.uuid4().hex[:8]}"
        auth_client.post("/admin/learning-topics",
            json={"name": unique, "priority": "medium"})
        resp = auth_client.post(f"/admin/learning-topics/{unique}/toggle")
        assert resp.status_code == 200


class TestDigestEndpoints:
    def test_digest_stats(self, auth_client):
        resp = auth_client.get("/admin/digest/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "digest_count" in data
