"""Tests for coach.py — local, no-LLM persona coaching composition."""
from __future__ import annotations

from coach import compose_coaching_answer

PERSONA = {
    "role": "master plumber mentor",
    "tone": "encouraging, safety-first",
    "greeting": "Good question — let's break it down safely.",
    "signoff": "Stay safe, and shut off the supply before you start.",
}

HITS = [
    {"answer": "Turn off the water at the shutoff valve first. Then drain the line.", "question": "q1", "domain": "plumbing", "score": 0.9},
    {"answer": "Replace the worn flapper if it does not seal. A new flapper is cheap.", "question": "q2", "domain": "fixtures", "score": 0.7},
    {"answer": "Adjust the fill valve so the water sits below the overflow tube.", "question": "q3", "domain": "fixtures", "score": 0.6},
]


class TestComposeCoachingAnswer:
    def test_includes_greeting_core_and_signoff(self):
        out = compose_coaching_answer("how do I fix it", HITS, PERSONA)
        assert PERSONA["greeting"] in out
        assert HITS[0]["answer"] in out
        assert PERSONA["signoff"] in out

    def test_base_answer_used_as_core(self):
        out = compose_coaching_answer("q", HITS, PERSONA, base_answer="Custom core answer.")
        assert "Custom core answer." in out
        # The base answer is the core, so the top hit becomes a supporting point.
        assert "Key points to remember:" in out

    def test_disclaimers_appended(self):
        out = compose_coaching_answer(
            "q", HITS, PERSONA, disclaimers=["Not professional advice.", ""]
        )
        assert "_Not professional advice._" in out

    def test_key_points_deduped_and_capped(self):
        dup_hits = [
            {"answer": "Same point here."},
            {"answer": "Same point here."},
            {"answer": "A second distinct point."},
            {"answer": "A third distinct point."},
            {"answer": "A fourth distinct point."},
        ]
        out = compose_coaching_answer("q", dup_hits, PERSONA, base_answer="Core.", max_points=2)
        # "Same point here." appears once across the bulleted points.
        assert out.count("- Same point here.") <= 1
        assert out.count("\n- ") <= 2

    def test_no_persona_still_returns_core(self):
        out = compose_coaching_answer("q", HITS, None)
        assert HITS[0]["answer"] in out

    def test_empty_inputs_return_base(self):
        assert compose_coaching_answer("q", [], PERSONA, base_answer="") == ""
        assert compose_coaching_answer("q", None, PERSONA, base_answer="Only this.").startswith(
            PERSONA["greeting"]
        )

    def test_tone_opener_when_no_greeting(self):
        persona = {"tone": "warm and motivating"}
        out = compose_coaching_answer("q", HITS, persona, base_answer="Core.")
        # An opener line is present even without an explicit greeting.
        assert out.split("\n\n")[0].strip() != "Core."


class TestCoachWrapGate:
    """The app-level gate: off by default, composes only when enabled."""

    def test_disabled_by_default_returns_raw(self, monkeypatch):
        import app

        monkeypatch.delenv("COACH_COMPOSE", raising=False)
        assert app._coach_wrap("q", "Raw answer.", HITS, 0.9) == "Raw answer."

    def test_low_confidence_not_composed(self, monkeypatch):
        import app

        monkeypatch.setenv("COACH_COMPOSE", "1")
        assert app._coach_wrap("q", "Raw answer.", HITS, 0.05) == "Raw answer."

    def test_enabled_composes_with_active_persona(self, monkeypatch):
        import app

        monkeypatch.setenv("COACH_COMPOSE", "1")
        monkeypatch.setenv("LIBAIX_ACTIVE_TRADE", "plumbing")
        import trade_pack

        trade_pack.clear_cache()
        out = app._coach_wrap("q", "Raw answer.", HITS, 0.9)
        assert "Raw answer." in out
        assert out != "Raw answer."  # persona framing applied
        # plumbing disclaimer surfaced
        assert "licensed" in out.lower()
