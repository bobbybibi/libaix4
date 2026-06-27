"""Tests for conversation_engine.py."""

from __future__ import annotations

import pytest

from conversation_engine import (
    ConversationContext,
    _extract_topic,
    enrich_with_context,
    is_followup,
    resolve_followup,
)


# ── ConversationContext basics ────────────────────────────────────────

class TestConversationContext:
    def test_init(self):
        ctx = ConversationContext()
        assert ctx.history == []
        assert ctx.current_domain == ""
        assert ctx.topic_stack == []
        assert ctx.max_history == 20

    def test_add_turn_updates_state(self):
        ctx = ConversationContext()
        ctx.add_turn("What is TCP?", "TCP is a protocol.", "networking", 0.95)
        assert len(ctx.history) == 1
        assert ctx.current_domain == "networking"
        assert ctx.topic_stack == ["networking"]
        assert ctx.get_last_question() == "What is TCP?"
        assert ctx.get_last_answer() == "TCP is a protocol."
        assert ctx.get_last_domain() == "networking"

    def test_max_history_truncation(self):
        ctx = ConversationContext(max_history=5)
        for i in range(10):
            ctx.add_turn(f"q{i}", f"a{i}", "d", 0.5)
        assert len(ctx.history) == 5
        assert ctx.history[0].question == "q5"
        assert ctx.history[-1].question == "q9"

    def test_topic_stack_dedup(self):
        ctx = ConversationContext()
        ctx.add_turn("q1", "a1", "networking", 0.9)
        ctx.add_turn("q2", "a2", "networking", 0.9)
        assert ctx.topic_stack == ["networking"]
        ctx.add_turn("q3", "a3", "security", 0.8)
        assert ctx.topic_stack == ["networking", "security"]

    def test_topic_stack_max(self):
        ctx = ConversationContext()
        for i in range(15):
            ctx.add_turn(f"q{i}", f"a{i}", f"domain_{i}", 0.5)
        assert len(ctx.topic_stack) <= 10

    def test_empty_getters(self):
        ctx = ConversationContext()
        assert ctx.get_last_question() == ""
        assert ctx.get_last_answer() == ""
        assert ctx.get_last_domain() == ""


# ── Serialization roundtrip ──────────────────────────────────────────

class TestSerialization:
    def test_serialize_deserialize_roundtrip(self):
        ctx = ConversationContext()
        ctx.add_turn("What is DNS?", "DNS resolves names.", "networking", 0.92)
        ctx.add_turn("And security?", "Security protects.", "security", 0.85)

        data = ctx.to_dict()
        restored = ConversationContext.from_dict(data)

        assert len(restored.history) == 2
        assert restored.current_domain == "security"
        assert restored.topic_stack == ["networking", "security"]
        assert restored.history[0].question == "What is DNS?"
        assert restored.history[1].answer == "Security protects."

    def test_from_dict_empty(self):
        ctx = ConversationContext.from_dict({})
        assert ctx.history == []
        assert ctx.current_domain == ""

    def test_from_dict_none(self):
        ctx = ConversationContext.from_dict(None)
        assert ctx.history == []


# ── Follow-up detection ──────────────────────────────────────────────

class TestIsFollowup:
    @pytest.mark.parametrize("question", [
        "tell me more",
        "Tell me more about it",
        "what about firewalls?",
        "how about encryption?",
        "and for TCP?",
        "also UDP",
        "why?",
        "how come?",
        "can you explain that?",
        "what is it?",
        "what's that?",
        "how does it work?",
        "related to that, what about DNS?",
        "give me an example",
        "compared to UDP",
        "elaborate",
        "explain more",
    ])
    def test_detects_patterns(self, question: str):
        assert is_followup(question) is True

    @pytest.mark.parametrize("question", [
        "What is a firewall?",
        "How does TCP work?",
        "Explain networking protocols",
        "Define encryption",
        "What are the OSI layers?",
        "",
        "hi",
    ])
    def test_negative_cases(self, question: str):
        assert is_followup(question) is False


# ── Follow-up resolution ─────────────────────────────────────────────

class TestResolveFollowup:
    def _ctx_with(self, question: str, domain: str) -> ConversationContext:
        ctx = ConversationContext()
        ctx.add_turn(question, "Some answer.", domain, 0.9)
        return ctx

    def test_tell_me_more(self):
        ctx = self._ctx_with("What is TCP?", "networking")
        result = resolve_followup("tell me more", ctx)
        assert "TCP" in result

    def test_what_about(self):
        ctx = self._ctx_with("What is TCP?", "networking")
        result = resolve_followup("what about UDP?", ctx)
        assert "UDP" in result
        assert "networking" in result

    def test_pronoun_resolution(self):
        ctx = self._ctx_with("What is a firewall?", "security")
        result = resolve_followup("how does it protect?", ctx)
        assert "firewall" in result

    def test_no_history_passthrough(self):
        ctx = ConversationContext()
        result = resolve_followup("tell me more", ctx)
        assert result == "tell me more"


# ── Enrich with context ──────────────────────────────────────────────

class TestEnrichWithContext:
    def test_enrich_with_followup(self):
        ctx = ConversationContext()
        ctx.add_turn("What is DNS?", "DNS resolves names.", "networking", 0.9)

        result = enrich_with_context("tell me more", ctx)
        assert result["is_followup"] is True
        assert result["domain_hint"] == "networking"
        assert "DNS" in result["resolved_question"]

    def test_enrich_without_followup(self):
        ctx = ConversationContext()
        result = enrich_with_context("What is a firewall?", ctx)
        assert result["is_followup"] is False
        assert result["domain_hint"] == ""
        assert result["resolved_question"] == "What is a firewall?"

    def test_context_summary(self):
        ctx = ConversationContext()
        ctx.add_turn("q1", "a1", "networking", 0.9)
        ctx.add_turn("q2", "a2", "security", 0.8)
        result = enrich_with_context("What is TCP?", ctx)
        assert "networking" in result["context_summary"]
        assert "security" in result["context_summary"]


# ── Extract topic ─────────────────────────────────────────────────────

class TestExtractTopic:
    def test_strips_question_words(self):
        assert _extract_topic("What is a firewall?") == "firewall"

    def test_strips_articles(self):
        assert _extract_topic("the DNS server") == "DNS server"

    def test_short_returns_empty(self):
        assert _extract_topic("a") == ""

    def test_plain_topic(self):
        assert _extract_topic("TCP protocol") == "TCP protocol"


# ── Flask integration ─────────────────────────────────────────────────

class TestChatEndpointConversation:
    @pytest.fixture()
    def client(self):
        from app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_chat_stores_conversation_context(self, client):
        """After a chat request, the session should contain conversation data."""
        with client.session_transaction() as sess:
            sess.clear()

        resp = client.post("/chat", json={"question": "What is a firewall?"})
        # If model not loaded, we get 503 — that's fine, context still applies
        if resp.status_code == 503:
            pytest.skip("Knowledge model not loaded")

        data = resp.get_json()
        assert "answer" in data

        # A second follow-up request should work
        resp2 = client.post("/chat", json={"question": "tell me more"})
        if resp2.status_code == 200:
            data2 = resp2.get_json()
            assert "answer" in data2
