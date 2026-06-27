"""Tests for the is_action_intent function and related helpers in conversation_engine.py."""
from __future__ import annotations

import pytest

from conversation_engine import (
    ConversationContext,
    enrich_with_context,
    is_action_intent,
    is_followup,
    resolve_followup,
)


# ── is_action_intent — action phrases should return True ─────────────

class TestIsActionIntentTrue:
    """Phrases that are action commands should return True."""

    @pytest.mark.parametrize("phrase", [
        "scan my network",
        "scan ports on 192.168.1.1",
        "block domain malware.com",
        "unblock site example.com",
        "monitor /var/log",
        "watch /etc/passwd",
        "quarantine suspicious.exe",
        "turn on the lights",
        "turn off TV",
        "connect to vpn",
        "disconnect vpn",
        "start vpn",
        "stop vpn",
        "vpn connect",
        "vpn disconnect",
        "vpn status",
        "vpn on",
        "vpn off",
        "vpn up",
        "vpn down",
        "fetch page https://example.com",
        "show my devices",
        "list devices",
        "check my firewall",
        "firewall block 1.2.3.4",
        "firewall status",
        "malware scan /tmp",
        "virus scan /home",
        "am i being hacked",
        "am i hacked",
        "research quantum computing",
        "study machine learning",
        "investigate security breach",
        "deep dive into rust programming",
        "summarize this article",
        "fact check the earth is flat",
        "verify this claim",
        "dns filter status",
        "domain block status",
        "add smart device lamp",
        "register device thermostat",
        "system info",
        "system status",
        "system health",
        "port scan 10.0.0.1",
        "scan ports 10.0.0.1",
        "switch on the fan",
        "power off the heater",
        "who is on my network",
        "show blocked domains",
        "list blocked sites",
        "check blocked",
        "show my processes",
        "list connections",
    ])
    def test_action_phrase(self, phrase: str):
        assert is_action_intent(phrase) is True, f"Expected True for: {phrase!r}"


# ── is_action_intent — knowledge queries should return False ─────────

class TestIsActionIntentFalse:
    """Knowledge questions should NOT be detected as action intents."""

    @pytest.mark.parametrize("phrase", [
        "what is a firewall",
        "how does a VPN work",
        "explain DNS resolution",
        "what are ports",
        "tell me about encryption",
        "describe how hashing works",
        "why do we need firewalls",
        "what is the difference between TCP and UDP",
        "how does TLS handshake work",
        "define malware",
        "what does HTTP stand for",
        "can you explain routing",
    ])
    def test_knowledge_query(self, phrase: str):
        assert is_action_intent(phrase) is False, f"Expected False for: {phrase!r}"


# ── is_action_intent — edge cases ───────────────────────────────────

class TestIsActionIntentEdgeCases:
    def test_empty_string(self):
        assert is_action_intent("") is False

    def test_short_string(self):
        assert is_action_intent("hi") is False

    def test_whitespace_only(self):
        assert is_action_intent("   ") is False

    def test_single_word_no_match(self):
        assert is_action_intent("hello") is False

    def test_mixed_case_action(self):
        assert is_action_intent("SCAN my network") is True

    def test_mixed_case_vpn(self):
        assert is_action_intent("VPN Status") is True


# ── is_followup tests ───────────────────────────────────────────────

class TestIsFollowup:
    @pytest.mark.parametrize("phrase", [
        "tell me more",
        "more about that",
        "elaborate",
        "explain more",
        "what about security",
        "how about encryption",
        "and also this",
        "why",
        "how come",
        "can you explain that",
        "what is it",
        "how does it work",
        "related to that",
        "give me an example",
        "compared to Python",
    ])
    def test_followup_detected(self, phrase: str):
        assert is_followup(phrase) is True, f"Expected True for: {phrase!r}"

    def test_non_followup(self):
        assert is_followup("what is a firewall") is False

    def test_empty(self):
        assert is_followup("") is False

    def test_short(self):
        assert is_followup("hi") is False


# ── resolve_followup tests ──────────────────────────────────────────

class TestResolveFollowup:
    def test_no_history(self):
        ctx = ConversationContext()
        result = resolve_followup("tell me more", ctx)
        assert result == "tell me more"

    def test_tell_me_more_expands(self):
        ctx = ConversationContext()
        ctx.add_turn("what is DNS", "DNS stands for...", "networking", 0.9)
        result = resolve_followup("tell me more", ctx)
        assert "DNS" in result or "what is DNS" in result

    def test_what_about_with_domain(self):
        ctx = ConversationContext()
        ctx.add_turn("what is DNS", "DNS is...", "networking", 0.9)
        result = resolve_followup("what about caching", ctx)
        assert "caching" in result

    def test_pronoun_resolution(self):
        ctx = ConversationContext()
        ctx.add_turn("what is DNS", "DNS resolves names", "networking", 0.9)
        result = resolve_followup("how does it work", ctx)
        assert "DNS" in result or result == "how does it work"


# ── enrich_with_context tests ────────────────────────────────────────

class TestEnrichWithContext:
    def test_non_followup(self):
        ctx = ConversationContext()
        result = enrich_with_context("what is a firewall", ctx)
        assert result["is_followup"] is False
        assert result["resolved_question"] == "what is a firewall"

    def test_followup_enrichment(self):
        ctx = ConversationContext()
        ctx.add_turn("explain TCP", "TCP is...", "networking", 0.9)
        result = enrich_with_context("tell me more", ctx)
        assert result["is_followup"] is True
        assert result["domain_hint"] == "networking"

    def test_context_summary(self):
        ctx = ConversationContext()
        ctx.add_turn("q1", "a1", "domain1", 0.9)
        ctx.add_turn("q2", "a2", "domain2", 0.8)
        result = enrich_with_context("tell me more", ctx)
        assert "domain1" in result["context_summary"]
        assert "domain2" in result["context_summary"]
