"""
conversation_engine.py — Conversation context and follow-up handling for libaix.

Provides:
  • Conversation history tracking per session
  • Follow-up question detection ("what about...", "tell me more", "and for...")
  • Context-aware answer enrichment
  • Topic continuity across messages
  • Pronoun resolution ("it", "that", "this")
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ConversationTurn:
    """Single exchange in a conversation."""
    question: str
    answer: str
    domain: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationContext:
    """Tracks conversation state for a session."""
    history: list[ConversationTurn] = field(default_factory=list)
    current_domain: str = ""
    topic_stack: list[str] = field(default_factory=list)
    max_history: int = 20

    def add_turn(self, question: str, answer: str, domain: str, confidence: float) -> None:
        """Add a new turn and update context."""
        turn = ConversationTurn(question=question, answer=answer, domain=domain, confidence=confidence)
        self.history.append(turn)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self.current_domain = domain
        # Push topic if different from current
        if not self.topic_stack or self.topic_stack[-1] != domain:
            self.topic_stack.append(domain)
            if len(self.topic_stack) > 10:
                self.topic_stack = self.topic_stack[-10:]

    def get_last_domain(self) -> str:
        """Return the domain of the most recent exchange."""
        return self.current_domain

    def get_last_answer(self) -> str:
        """Return the most recent answer."""
        if self.history:
            return self.history[-1].answer
        return ""

    def get_last_question(self) -> str:
        """Return the most recent question."""
        if self.history:
            return self.history[-1].question
        return ""

    def to_dict(self) -> dict:
        """Serialize for session storage."""
        return {
            "history": [asdict(t) for t in self.history[-10:]],
            "current_domain": self.current_domain,
            "topic_stack": self.topic_stack,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """Deserialize from session storage."""
        if not data:
            return cls()
        ctx = cls()
        ctx.current_domain = data.get("current_domain", "")
        ctx.topic_stack = data.get("topic_stack", [])
        for h in data.get("history", []):
            ctx.history.append(ConversationTurn(
                question=h.get("question", ""),
                answer=h.get("answer", ""),
                domain=h.get("domain", ""),
                confidence=h.get("confidence", 0.0),
                timestamp=h.get("timestamp", 0.0),
            ))
        return ctx


# ── Follow-up detection ──────────────────────────────────────────────

FOLLOWUP_PATTERNS = [
    re.compile(r"^(tell me more|more about|elaborate|explain more|go on|continue)", re.I),
    re.compile(r"^(what about|how about|and |also |what else)", re.I),
    re.compile(r"^(why|how come|but why|and why)", re.I),
    re.compile(r"^(can you explain|could you explain|please explain)", re.I),
    re.compile(r"^(what is|what's) (it|that|this)\b", re.I),
    re.compile(r"^(how does|how do) (it|that|this|they)\b", re.I),
    re.compile(r"\b(related to that|on that note|speaking of)\b", re.I),
    re.compile(r"^(give me an example|example|for example|show me)", re.I),
    re.compile(r"^(compared to|versus|vs\.?|difference between)", re.I),
]

PRONOUN_PATTERNS = [
    re.compile(r"\b(it|this|that)\b(?!\s+is\s+\w)", re.I),
    re.compile(r"\b(its|their|these|those)\b", re.I),
]


def is_followup(question: str) -> bool:
    """Detect if a question is a follow-up to the previous exchange."""
    q = question.strip()
    if len(q) < 3:
        return False
    return any(p.search(q) for p in FOLLOWUP_PATTERNS)


def resolve_followup(question: str, context: ConversationContext) -> str:
    """Resolve a follow-up question using conversation context.

    If the question references the previous topic with pronouns or
    follow-up phrases, expand it with context from the last exchange.
    """
    if not context.history:
        return question

    last_q = context.get_last_question()
    last_domain = context.get_last_domain()

    q = question.strip()

    # "Tell me more" → "Tell me more about {last topic}"
    more_match = re.match(r"^(tell me more|more about|elaborate|explain more)", q, re.I)
    if more_match:
        return f"{q} about {last_q}" if "about" not in q.lower() else q

    # "What about X" → already specific, keep as is but inherit domain context
    what_about = re.match(r"^(what about|how about)\s+(.+)", q, re.I)
    if what_about:
        topic = what_about.group(2)
        return f"What is {topic} in {last_domain}?" if last_domain else q

    # Pronoun resolution: "How does it work?" → "How does {last_topic} work?"
    for patt in PRONOUN_PATTERNS:
        if patt.search(q) and last_q:
            # Extract key noun from last question
            key_topic = _extract_topic(last_q)
            if key_topic:
                resolved = patt.sub(lambda m: key_topic, q, count=1)
                if resolved != q:
                    return resolved

    return question


def _extract_topic(question: str) -> str:
    """Extract the main topic from a question."""
    # Remove question words and common prefixes
    q = re.sub(r"^(what is|what are|how do|how does|explain|describe|tell me about)\s+", "", question, flags=re.I)
    q = re.sub(r"\?+$", "", q).strip()
    # Remove articles
    q = re.sub(r"^(a|an|the)\s+", "", q, flags=re.I)
    return q if len(q) > 2 else ""


def enrich_with_context(question: str, context: ConversationContext) -> dict[str, Any]:
    """Analyze question in context and return enrichment data.

    Returns dict with:
      - resolved_question: The question after follow-up resolution
      - is_followup: Whether this is a follow-up question
      - domain_hint: Suggested domain from context
      - context_summary: Brief summary of recent conversation
    """
    followup = is_followup(question)
    resolved = resolve_followup(question, context) if followup else question

    domain_hint = context.get_last_domain() if followup else ""

    # Build a brief context summary
    recent = context.history[-3:] if context.history else []
    summary = " → ".join(t.domain for t in recent) if recent else ""

    return {
        "resolved_question": resolved,
        "is_followup": followup,
        "domain_hint": domain_hint,
        "context_summary": summary,
    }


# ── Action intent detection ──────────────────────────────────────────

# Patterns that indicate the user wants an *action* rather than a knowledge answer.
ACTION_PATTERNS = [
    re.compile(r"^(scan|block|unblock|monitor|watch|quarantine|isolate)\b", re.I),
    re.compile(r"^turn\s+(on|off)\b", re.I),
    re.compile(r"^(connect|disconnect|start|stop)\s+(to\s+)?vpn\b", re.I),
    re.compile(r"\bvpn\s+(connect|disconnect|start|stop|on|off|up|down|status)\b", re.I),
    re.compile(r"^(fetch|extract|search|look\s+up)\b.*\b(page|url|links?|web|online)\b", re.I),
    re.compile(r"\b(firewall|malware|virus)\s+(scan|block|status|rules)\b", re.I),
    re.compile(r"^(show|list|check)\s+(my\s+)?(devices?|processes|connections|firewall|blocked)\b", re.I),
    re.compile(r"\b(am\s+i\s+(being\s+)?hacked)\b", re.I),
    re.compile(r"^(research|study|investigate|learn\s+about|deep\s+dive)\b", re.I),
    re.compile(r"^(summarize|fact\s+check|verify)\b", re.I),
    re.compile(r"\b(dns|domain)\s+(filter|block|status)\b", re.I),
    re.compile(r"^(add|register)\s+(smart\s+)?device\b", re.I),
    re.compile(r"\bsystem\s+(info|information|status|health)\b", re.I),
    re.compile(r"\bport\s+scan\b", re.I),
    re.compile(r"\bscan\s+ports?\b", re.I),
    re.compile(r"\b(switch|power)\s+(on|off)\b", re.I),
    re.compile(r"\bwho\s+is\s+on\s+my\s+network\b", re.I),
]


def is_action_intent(question: str) -> bool:
    """Detect if a question is an action command rather than a knowledge query."""
    q = question.strip()
    if len(q) < 3:
        return False
    return any(p.search(q) for p in ACTION_PATTERNS)
