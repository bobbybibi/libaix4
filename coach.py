"""
coach.py — Local, no-LLM coaching composition.

Reshapes the **already-computed** retrieval hits into a persona-framed,
encouraging, step-by-step answer. This is pure string composition: no network,
no API, fully offline and zero per-query cost. All selection and scoring happen
upstream (retrieval / reasoning); this module only changes *presentation* so the
assistant reads like a supportive mentor for the active trade.

Wiring (see app.py): the response cache stores the **raw** answer; composition
is applied **on read**, behind the ``COACH_COMPOSE`` env flag, so the persona can
differ per trade/user without polluting the cache. Composition is skipped for
low-confidence/fallback answers.
"""

from __future__ import annotations

import re

# Default openers per tone, used only when a pack defines no explicit greeting.
_TONE_OPENERS = {
    "encouraging": "Great question — let's work through this together.",
    "warm": "Love that you're asking — let's break it down.",
    "motivating": "You're on the right track — here's how to think about it.",
    "direct": "Here's the answer, step by step.",
    "technical": "Here's the technical breakdown.",
    "precise": "Here's a precise breakdown.",
}

_DEFAULT_OPENER = "Let's work through this."

_MAX_POINT_LEN = 240


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text*, trimmed."""
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentence = parts[0].strip()
    if len(sentence) > _MAX_POINT_LEN:
        sentence = sentence[:_MAX_POINT_LEN].rstrip() + "…"
    return sentence


def _norm(text: str) -> str:
    """Normalise for de-duplication (lowercased, whitespace-collapsed)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _opener(persona: dict) -> str:
    greeting = (persona.get("greeting") or "").strip()
    if greeting:
        return greeting
    tone = (persona.get("tone") or "").lower()
    for key, opener in _TONE_OPENERS.items():
        if key in tone:
            return opener
    return _DEFAULT_OPENER


def _key_points(hits: list[dict], core: str, max_points: int) -> list[str]:
    """First sentences of supporting hits, excluding the core answer, deduped."""
    seen = {_norm(core)}
    points: list[str] = []
    for hit in hits:
        answer = hit.get("answer", "") if isinstance(hit, dict) else ""
        if _norm(answer) in seen:
            continue
        sentence = _first_sentence(answer)
        if not sentence or _norm(sentence) in seen:
            continue
        seen.add(_norm(answer))
        seen.add(_norm(sentence))
        points.append(sentence)
        if len(points) >= max_points:
            break
    return points


def compose_coaching_answer(
    question: str,
    retrieval_hits: list[dict] | None,
    persona: dict | None = None,
    *,
    base_answer: str | None = None,
    reasoning_chain: list | None = None,
    disclaimers: list[str] | None = None,
    max_points: int = 3,
) -> str:
    """Compose a persona-framed, encouraging answer from retrieval hits.

    *base_answer* (the upstream-selected answer) is used as the core when given;
    otherwise the top hit's answer is used. Returns the original ``base_answer``
    (or ``""``) when there is no usable content, so callers can wrap safely.
    """
    persona = persona or {}
    hits = [h for h in (retrieval_hits or []) if isinstance(h, dict)]

    core = (base_answer if base_answer is not None else (hits[0]["answer"] if hits else "")).strip()
    if not core:
        return base_answer or ""

    parts: list[str] = []

    opener = _opener(persona)
    if opener:
        parts.append(opener)

    parts.append(core)

    points = _key_points(hits, core, max_points)
    if points:
        parts.append("Key points to remember:")
        parts.append("\n".join(f"- {p}" for p in points))

    signoff = (persona.get("signoff") or "").strip()
    if signoff:
        parts.append(signoff)

    for disclaimer in disclaimers or []:
        disclaimer = (disclaimer or "").strip()
        if disclaimer:
            parts.append(f"_{disclaimer}_")

    return "\n\n".join(parts)
