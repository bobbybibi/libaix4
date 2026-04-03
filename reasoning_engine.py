"""
reasoning_engine.py — Deductive reasoning and analytical intelligence engine for libaix.

Provides the AI with structured reasoning capabilities beyond simple Q&A classification:
  • Deductive reasoning (if A then B, A therefore B)
  • Analogical reasoning (A:B :: C:D)
  • Causal chain analysis (A causes B causes C)
  • Concept relationship mapping
  • Multi-hop question answering
  • Confidence-weighted inference
  • Contradiction detection
  • Knowledge synthesis (combine multiple facts)
  • Pattern-based generalization
  • Contextual reasoning with memory
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from knowledge_base import KNOWLEDGE
from vectorizer import tokenize, BagOfWords

# ── Paths ────────────────────────────────────────────────────────────

REASONING_CONFIG_PATH = Path("data/reasoning_config.json")
REASONING_STATE_PATH = Path("data/reasoning_state.json")

# ── Singleton ────────────────────────────────────────────────────────

_engine_instance: ReasoningEngine | None = None

# ── Rule-extraction patterns ─────────────────────────────────────────

_IDENTITY_RE = re.compile(
    r"^(.+?)\s+is\s+(?:a|an|the)\s+(.+?)(?:\.|,|;|$)", re.IGNORECASE
)
_PURPOSE_RE = re.compile(
    r"(.+?)\s+is\s+used\s+(?:for|to)\s+(.+?)(?:\.|,|;|$)", re.IGNORECASE
)
_FUNCTION_RE = re.compile(
    r"(.+?)\s+provides?\s+(.+?)(?:\.|,|;|$)", re.IGNORECASE
)
_PROTECTION_RE = re.compile(
    r"(.+?)\s+prevents?\s+(.+?)(?:\.|,|;|$)", re.IGNORECASE
)
_COMPOSITION_RE = re.compile(
    r"(.+?)\s+consists?\s+of\s+(.+?)(?:\.|,|;|$)", re.IGNORECASE
)
_ENABLES_RE = re.compile(
    r"(.+?)\s+(?:enables?|allows?)\s+(.+?)(?:\.|,|;|$)", re.IGNORECASE
)

# Pattern name → compiled regex
_RULE_PATTERNS: dict[str, re.Pattern] = {
    "identity": _IDENTITY_RE,
    "purpose": _PURPOSE_RE,
    "function": _FUNCTION_RE,
    "protection": _PROTECTION_RE,
    "composition": _COMPOSITION_RE,
    "enables": _ENABLES_RE,
}

# ── Causal keywords ─────────────────────────────────────────────────

_CAUSAL_KEYWORDS = frozenset(
    "causes leads causes enables triggers produces results creates "
    "prevents blocks stops protects reduces".split()
)

# ── Concept-extraction helpers ───────────────────────────────────────

_DEFINITION_MARKERS = re.compile(
    r"(?:is\s+a|is\s+an|is\s+the|called|known\s+as|refers?\s+to)\s+", re.IGNORECASE
)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class ReasoningRule:
    """A single reasoning rule: if conditions then conclusion."""

    conditions: list[str]
    conclusion: str
    confidence: float
    domain: str
    source: str  # which knowledge entries derived this
    rule_type: str = "identity"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ReasoningRule:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ConceptNode:
    """A node in the concept graph."""

    name: str
    domain: str
    related: dict[str, float] = field(default_factory=dict)
    properties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ConceptNode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =====================================================================
# Config helpers
# =====================================================================

def _default_config() -> dict:
    return {
        "similarity_threshold": 0.30,
        "analogy_threshold": 0.25,
        "causal_confidence_decay": 0.85,
        "max_causal_depth": 5,
        "max_multi_hop": 3,
        "contradiction_threshold": 0.80,
        "min_rule_confidence": 0.40,
        "synthesis_min_overlap": 2,
        "generalize_min_examples": 2,
        "cache_ttl_seconds": 300,
        "last_build": None,
        "build_count": 0,
        "stats": {},
    }


def load_reasoning_config() -> dict:
    """Load reasoning config from disk, creating defaults when missing."""
    if REASONING_CONFIG_PATH.exists():
        try:
            return json.loads(REASONING_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    cfg = _default_config()
    save_reasoning_config(cfg)
    return cfg


def save_reasoning_config(config: dict) -> None:
    """Persist reasoning config to disk."""
    REASONING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    REASONING_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# =====================================================================
# Utility helpers
# =====================================================================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_capitalized(text: str) -> list[str]:
    """Extract capitalized words / acronyms as candidate concepts."""
    words = re.findall(r"\b[A-Z][A-Za-z]{2,}\b", text)
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    return list(dict.fromkeys(words + acronyms))  # deduplicate, preserve order


def _extract_definitions(text: str) -> list[str]:
    """Extract words after definition markers (is a, is the, called, …)."""
    results: list[str] = []
    for match in _DEFINITION_MARKERS.finditer(text):
        rest = text[match.end():]
        # Grab the next noun phrase (up to punctuation)
        phrase = re.match(r"([A-Za-z0-9 -]+)", rest)
        if phrase:
            results.append(phrase.group(1).strip())
    return results


def _extract_related(text: str) -> list[str]:
    """Extract concepts connected by 'and' / 'or'."""
    parts: list[str] = []
    for seg in re.split(r"\band\b|\bor\b", text, flags=re.IGNORECASE):
        seg = seg.strip().rstrip(".,;:")
        tokens = seg.split()
        if 1 <= len(tokens) <= 4:
            parts.append(seg)
    return parts


def _token_overlap(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Jaccard overlap between two token lists."""
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# =====================================================================
# ReasoningEngine
# =====================================================================

class ReasoningEngine:
    """Main reasoning engine — builds a concept graph and rule base from
    the KNOWLEDGE entries and provides multi-strategy reasoning."""

    def __init__(self) -> None:
        self.rules: list[ReasoningRule] = []
        self.concepts: dict[str, ConceptNode] = {}
        self.inference_cache: dict[str, dict] = {}
        self._bow: BagOfWords | None = None
        self._entry_vectors: np.ndarray | None = None
        self._entries: list[tuple[str, str, str]] = []
        self._config = load_reasoning_config()
        self._cache_ts: dict[str, float] = {}

    # ── Core reasoning ───────────────────────────────────────────────

    def reason(self, question: str, context: list[str] | None = None) -> dict:
        """Apply multi-strategy reasoning to answer a question.

        Returns
        -------
        dict with keys: answer, confidence, reasoning_chain, strategy
        """
        # Check cache
        cache_key = question.lower().strip()
        ttl = self._config.get("cache_ttl_seconds", 300)
        if cache_key in self.inference_cache:
            cached = self.inference_cache[cache_key]
            if time.time() - self._cache_ts.get(cache_key, 0) < ttl:
                return cached

        strategies: list[tuple[str, Any]] = [
            ("direct_match", self._strategy_direct_match),
            ("deductive", self._strategy_deductive),
            ("causal", self._strategy_causal),
            ("analogical", self._strategy_analogical),
            ("multi_hop", self._strategy_multi_hop),
            ("synthesis", self._strategy_synthesis),
        ]

        best_result: dict | None = None
        best_confidence = -1.0

        for name, strategy_fn in strategies:
            try:
                result = strategy_fn(question, context)
                if result and result.get("confidence", 0) > best_confidence:
                    best_confidence = result["confidence"]
                    best_result = result
                    best_result["strategy"] = name
                # Short-circuit on high-confidence direct match
                if best_confidence >= 0.90:
                    break
            except Exception:
                continue

        if best_result is None:
            best_result = {
                "answer": "I could not reason about this question with the available knowledge.",
                "confidence": 0.0,
                "reasoning_chain": [],
                "strategy": "none",
            }

        # Cache result
        self.inference_cache[cache_key] = best_result
        self._cache_ts[cache_key] = time.time()
        return best_result

    # ── Strategy implementations ─────────────────────────────────────

    def _strategy_direct_match(
        self, question: str, context: list[str] | None
    ) -> dict | None:
        """Find the closest knowledge entry by vector similarity."""
        if self._bow is None or self._entry_vectors is None:
            return None

        q_vec = self._bow.transform([question])[0]
        sims = self._entry_vectors @ q_vec  # cosine sims (vectors are L2-normed)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self._config.get("similarity_threshold", 0.30):
            return None

        q_text, a_text, domain = self._entries[best_idx]
        return {
            "answer": a_text,
            "confidence": min(best_sim, 1.0),
            "reasoning_chain": [
                f"Direct match to: '{q_text}' (similarity={best_sim:.3f})"
            ],
            "strategy": "direct_match",
        }

    def _strategy_deductive(
        self, question: str, context: list[str] | None
    ) -> dict | None:
        """Apply if-then rules to derive an answer."""
        q_tokens = set(tokenize(question))
        if not q_tokens:
            return None

        matched_rules: list[tuple[ReasoningRule, float]] = []
        for rule in self.rules:
            cond_tokens = set()
            for cond in rule.conditions:
                cond_tokens.update(tokenize(cond))
            if not cond_tokens:
                continue
            overlap = len(q_tokens & cond_tokens) / max(len(cond_tokens), 1)
            if overlap >= 0.3:
                matched_rules.append((rule, overlap * rule.confidence))

        if not matched_rules:
            return None

        matched_rules.sort(key=lambda r: r[1], reverse=True)
        best_rule, score = matched_rules[0]

        chain = [
            f"Rule ({best_rule.rule_type}): IF {'; '.join(best_rule.conditions)} "
            f"THEN {best_rule.conclusion}",
            f"Source: {best_rule.source}",
        ]
        # Combine top rules if multiple match
        if len(matched_rules) > 1:
            supporting = matched_rules[1:4]
            for r, s in supporting:
                chain.append(
                    f"Supporting rule ({r.rule_type}, score={s:.3f}): {r.conclusion}"
                )

        return {
            "answer": best_rule.conclusion,
            "confidence": min(score, 1.0),
            "reasoning_chain": chain,
            "strategy": "deductive",
        }

    def _strategy_analogical(
        self, question: str, context: list[str] | None
    ) -> dict | None:
        """Find analogical relationships."""
        q_tokens = set(tokenize(question))
        if not q_tokens:
            return None

        # Find the concept most related to the question
        best_concept: str | None = None
        best_overlap = 0.0
        for cname, node in self.concepts.items():
            c_tokens = set(tokenize(cname))
            overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_concept = cname

        if best_concept is None or best_overlap < 0.1:
            return None

        analogies = self.find_analogies(best_concept)
        if not analogies:
            return None

        top = analogies[0]
        chain = [
            f"Concept '{best_concept}' is analogous to '{top['target']}' "
            f"(strength={top['strength']:.3f})",
        ]
        # Build answer from the related concept's knowledge
        answer_parts: list[str] = []
        for entry_q, entry_a, _ in self._entries:
            if best_concept.lower() in entry_a.lower():
                answer_parts.append(entry_a)
                break

        if not answer_parts:
            return None

        return {
            "answer": answer_parts[0],
            "confidence": min(best_overlap * top["strength"], 1.0),
            "reasoning_chain": chain,
            "strategy": "analogical",
        }

    def _strategy_causal(
        self, question: str, context: list[str] | None
    ) -> dict | None:
        """Trace causal chains to answer the question."""
        q_tokens = set(tokenize(question))
        causal_words = {"cause", "why", "because", "result", "effect", "lead",
                        "happen", "prevent", "impact", "consequence"}
        if not q_tokens & causal_words:
            return None

        # Find best starting concept
        best_concept: str | None = None
        best_overlap = 0.0
        for cname in self.concepts:
            c_tokens = set(tokenize(cname))
            overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_concept = cname

        if best_concept is None:
            return None

        chain_result = self.trace_causal_chain(best_concept)
        if not chain_result:
            return None

        steps = [f"{step['from']} → {step['to']} ({step['type']})"
                 for step in chain_result]
        answer = (
            f"Causal chain from '{best_concept}': "
            + " → ".join(step["to"] for step in chain_result)
        )

        return {
            "answer": answer,
            "confidence": min(best_overlap * 0.8, 1.0),
            "reasoning_chain": steps,
            "strategy": "causal",
        }

    def _strategy_multi_hop(
        self, question: str, context: list[str] | None
    ) -> dict | None:
        """Combine multiple knowledge entries."""
        result = self.multi_hop_answer(question)
        if result["confidence"] > 0:
            return result
        return None

    def _strategy_synthesis(
        self, question: str, context: list[str] | None
    ) -> dict | None:
        """Synthesise across multiple concepts."""
        q_tokens = tokenize(question)
        # Find concept names mentioned in the question
        matched_concepts: list[str] = []
        for cname in self.concepts:
            c_tokens = set(tokenize(cname))
            if c_tokens & set(q_tokens):
                matched_concepts.append(cname)

        if len(matched_concepts) < 2:
            return None

        result = self.synthesize(matched_concepts[:5])
        if result["confidence"] > 0:
            return result
        return None

    # ── Public reasoning methods ─────────────────────────────────────

    def deduce(self, premises: list[str]) -> list[dict]:
        """Apply deductive reasoning rules to premises.

        Parameters
        ----------
        premises : list[str]
            A list of premise statements.

        Returns
        -------
        list[dict] — each with keys: conclusion, confidence, rule_type, chain
        """
        premise_tokens = set()
        for p in premises:
            premise_tokens.update(tokenize(p))

        results: list[dict] = []
        for rule in self.rules:
            cond_tokens = set()
            for cond in rule.conditions:
                cond_tokens.update(tokenize(cond))
            if not cond_tokens:
                continue
            matched = premise_tokens & cond_tokens
            coverage = len(matched) / len(cond_tokens)
            if coverage >= 0.5:
                results.append({
                    "conclusion": rule.conclusion,
                    "confidence": round(coverage * rule.confidence, 4),
                    "rule_type": rule.rule_type,
                    "chain": [
                        f"Premises match rule conditions ({coverage:.0%} coverage)",
                        f"Rule: {'; '.join(rule.conditions)} → {rule.conclusion}",
                    ],
                })

        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    def find_analogies(
        self, concept: str, domain: str | None = None
    ) -> list[dict]:
        """Find analogical relationships for a concept.

        Parameters
        ----------
        concept : str
            The concept to find analogies for.
        domain : str | None
            Optional domain filter.

        Returns
        -------
        list[dict] — each with keys: source, target, strength, shared_properties
        """
        node = self.concepts.get(concept)
        if node is None:
            # Try case-insensitive lookup
            for k, v in self.concepts.items():
                if k.lower() == concept.lower():
                    node = v
                    break
        if node is None:
            return []

        analogies: list[dict] = []
        threshold = self._config.get("analogy_threshold", 0.25)

        for other_name, other_node in self.concepts.items():
            if other_name == node.name:
                continue
            if domain and other_node.domain != domain:
                continue

            # Strength from direct relationship
            direct_strength = node.related.get(other_name, 0.0)

            # Shared properties
            shared = set(node.properties) & set(other_node.properties)
            prop_strength = len(shared) / max(
                len(set(node.properties) | set(other_node.properties)), 1
            )

            # Shared neighbours
            shared_neighbours = set(node.related) & set(other_node.related)
            neighbour_strength = len(shared_neighbours) / max(
                len(set(node.related) | set(other_node.related)), 1
            )

            total = (direct_strength * 0.5 +
                     prop_strength * 0.3 +
                     neighbour_strength * 0.2)

            if total >= threshold:
                analogies.append({
                    "source": node.name,
                    "target": other_name,
                    "strength": round(total, 4),
                    "shared_properties": list(shared),
                })

        analogies.sort(key=lambda a: a["strength"], reverse=True)
        return analogies

    def trace_causal_chain(
        self, cause: str, max_depth: int = 5
    ) -> list[dict]:
        """Trace cause-effect chain from a starting concept.

        Parameters
        ----------
        cause : str
            Starting concept name.
        max_depth : int
            Maximum chain length.

        Returns
        -------
        list[dict] — each with keys: from, to, type, confidence
        """
        max_depth = min(max_depth, self._config.get("max_causal_depth", 5))
        decay = self._config.get("causal_confidence_decay", 0.85)

        chain: list[dict] = []
        visited: set[str] = set()
        current = cause

        for depth in range(max_depth):
            if current in visited:
                break
            visited.add(current)

            node = self.concepts.get(current)
            if node is None:
                break

            # Find rules where current is in conditions → extract effects
            best_effect: str | None = None
            best_conf = 0.0
            effect_type = "causes"

            for rule in self.rules:
                cond_tokens = set()
                for c in rule.conditions:
                    cond_tokens.update(tokenize(c))
                if set(tokenize(current)) & cond_tokens:
                    conf = rule.confidence * (decay ** depth)
                    if conf > best_conf:
                        best_conf = conf
                        best_effect = rule.conclusion
                        effect_type = rule.rule_type

            # Also check direct graph relationships
            if node.related:
                for rel_name, strength in sorted(
                    node.related.items(), key=lambda x: -x[1]
                ):
                    if rel_name not in visited:
                        rel_conf = strength * (decay ** depth)
                        if rel_conf > best_conf:
                            best_conf = rel_conf
                            best_effect = rel_name
                            effect_type = "related"
                        break

            if best_effect is None:
                break

            chain.append({
                "from": current,
                "to": best_effect,
                "type": effect_type,
                "confidence": round(best_conf, 4),
            })
            current = best_effect

        return chain

    def multi_hop_answer(
        self, question: str, max_hops: int = 3
    ) -> dict:
        """Answer questions that require combining multiple knowledge entries.

        Parameters
        ----------
        question : str
            The question to answer.
        max_hops : int
            Maximum number of knowledge entries to chain.

        Returns
        -------
        dict with keys: answer, confidence, reasoning_chain, hops
        """
        max_hops = min(max_hops, self._config.get("max_multi_hop", 3))

        if self._bow is None or self._entry_vectors is None:
            return {"answer": "", "confidence": 0.0, "reasoning_chain": [], "hops": 0}

        q_vec = self._bow.transform([question])[0]
        sims = self._entry_vectors @ q_vec
        threshold = self._config.get("similarity_threshold", 0.30)

        # Get top-k entries above threshold
        top_indices = np.argsort(sims)[::-1]
        selected: list[int] = []
        for idx in top_indices:
            if float(sims[idx]) < threshold * 0.5:
                break
            selected.append(int(idx))
            if len(selected) >= max_hops:
                break

        if not selected:
            return {"answer": "", "confidence": 0.0, "reasoning_chain": [], "hops": 0}

        # Combine answers
        chain: list[str] = []
        answer_parts: list[str] = []
        total_sim = 0.0

        for rank, idx in enumerate(selected):
            entry_q, entry_a, domain = self._entries[idx]
            sim = float(sims[idx])
            total_sim += sim
            chain.append(
                f"Hop {rank + 1}: '{entry_q}' (sim={sim:.3f}, domain={domain})"
            )
            answer_parts.append(entry_a)

        avg_confidence = total_sim / len(selected) if selected else 0.0
        # Penalise multi-hop slightly for uncertainty
        confidence = avg_confidence * (0.95 ** (len(selected) - 1))

        combined = " Additionally, ".join(answer_parts[:max_hops])

        return {
            "answer": combined,
            "confidence": round(min(confidence, 1.0), 4),
            "reasoning_chain": chain,
            "strategy": "multi_hop",
            "hops": len(selected),
        }

    def detect_contradictions(self) -> list[dict]:
        """Find potential contradictions in the knowledge base.

        Returns
        -------
        list[dict] — each with keys: entry_a, entry_b, similarity, reason
        """
        if self._bow is None or self._entry_vectors is None:
            return []

        threshold = self._config.get("contradiction_threshold", 0.80)
        contradictions: list[dict] = []

        # Negation indicators
        negation_words = {"not", "no", "never", "without", "cannot", "doesn",
                          "isn", "aren", "won", "don", "didn"}

        n = len(self._entries)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(self._entry_vectors[i] @ self._entry_vectors[j])
                if sim < threshold:
                    continue

                q_i, a_i, d_i = self._entries[i]
                q_j, a_j, d_j = self._entries[j]

                tokens_i = set(tokenize(a_i))
                tokens_j = set(tokenize(a_j))

                neg_i = bool(tokens_i & negation_words)
                neg_j = bool(tokens_j & negation_words)

                if neg_i != neg_j:
                    contradictions.append({
                        "entry_a": q_i,
                        "entry_b": q_j,
                        "answer_a": a_i,
                        "answer_b": a_j,
                        "similarity": round(sim, 4),
                        "reason": "High similarity but opposite negation polarity",
                    })

                # Check for conflicting values (e.g., "X is A" vs "X is B")
                if d_i == d_j and sim > 0.9:
                    defs_i = _extract_definitions(a_i)
                    defs_j = _extract_definitions(a_j)
                    if defs_i and defs_j and set(defs_i) != set(defs_j):
                        overlap = _token_overlap(
                            tokenize(" ".join(defs_i)),
                            tokenize(" ".join(defs_j)),
                        )
                        if overlap < 0.3:
                            contradictions.append({
                                "entry_a": q_i,
                                "entry_b": q_j,
                                "answer_a": a_i,
                                "answer_b": a_j,
                                "similarity": round(sim, 4),
                                "reason": "Same domain, similar question, different definitions",
                            })

        return contradictions

    def synthesize(self, concepts: list[str]) -> dict:
        """Synthesize new knowledge from multiple concepts.

        Parameters
        ----------
        concepts : list[str]
            Concept names to combine.

        Returns
        -------
        dict with keys: answer, confidence, reasoning_chain, concepts_used
        """
        min_overlap = self._config.get("synthesis_min_overlap", 2)

        found_entries: list[tuple[str, str, str, float]] = []
        for concept in concepts:
            c_tokens = set(tokenize(concept))
            for idx, (q, a, d) in enumerate(self._entries):
                a_tokens = set(tokenize(a))
                q_tokens = set(tokenize(q))
                overlap = len(c_tokens & (a_tokens | q_tokens))
                if overlap > 0:
                    found_entries.append((q, a, d, overlap / max(len(c_tokens), 1)))

        if len(found_entries) < min_overlap:
            return {
                "answer": "",
                "confidence": 0.0,
                "reasoning_chain": [],
                "concepts_used": concepts,
            }

        # Sort by relevance and take the top entries
        found_entries.sort(key=lambda x: x[3], reverse=True)
        top = found_entries[:5]

        # Merge the key sentences
        chain: list[str] = []
        answer_parts: list[str] = []
        total_rel = 0.0

        for q, a, d, rel in top:
            chain.append(f"From [{d}]: {q} (relevance={rel:.3f})")
            # Take the first sentence of each answer
            first_sentence = a.split(".")[0] + "."
            if first_sentence not in answer_parts:
                answer_parts.append(first_sentence)
            total_rel += rel

        avg_rel = total_rel / len(top) if top else 0.0

        return {
            "answer": " ".join(answer_parts),
            "confidence": round(min(avg_rel, 1.0), 4),
            "reasoning_chain": chain,
            "strategy": "synthesis",
            "concepts_used": concepts,
        }

    def generalize(self, examples: list[dict]) -> list[dict]:
        """Extract general patterns from specific examples.

        Parameters
        ----------
        examples : list[dict]
            Each dict should have at least 'input' and 'output' keys.

        Returns
        -------
        list[dict] — each with keys: pattern, confidence, supporting_count
        """
        min_examples = self._config.get("generalize_min_examples", 2)
        if len(examples) < min_examples:
            return []

        # Token-frequency approach: find common tokens across examples
        input_freq: dict[str, int] = defaultdict(int)
        output_freq: dict[str, int] = defaultdict(int)

        for ex in examples:
            inp = ex.get("input", "")
            out = ex.get("output", "")
            for t in set(tokenize(inp)):
                input_freq[t] += 1
            for t in set(tokenize(out)):
                output_freq[t] += 1

        n = len(examples)
        patterns: list[dict] = []

        # Find tokens appearing in the majority of examples
        common_input = {t for t, c in input_freq.items() if c >= n * 0.5}
        common_output = {t for t, c in output_freq.items() if c >= n * 0.5}

        if common_input and common_output:
            confidence = (
                sum(input_freq[t] for t in common_input)
                + sum(output_freq[t] for t in common_output)
            ) / (2 * n * max(len(common_input) + len(common_output), 1))

            patterns.append({
                "pattern": f"When input contains [{', '.join(sorted(common_input))}] "
                           f"→ output contains [{', '.join(sorted(common_output))}]",
                "confidence": round(min(confidence, 1.0), 4),
                "supporting_count": n,
                "common_input_tokens": sorted(common_input),
                "common_output_tokens": sorted(common_output),
            })

        # Per-domain patterns from knowledge entries
        domain_patterns: dict[str, list[str]] = defaultdict(list)
        for ex in examples:
            domain = ex.get("domain", "general")
            out = ex.get("output", "")
            domain_patterns[domain].append(out)

        for domain, outputs in domain_patterns.items():
            if len(outputs) < min_examples:
                continue
            all_tokens = [set(tokenize(o)) for o in outputs]
            shared = set.intersection(*all_tokens) if all_tokens else set()
            if shared:
                patterns.append({
                    "pattern": f"[{domain}] entries commonly reference: "
                               f"{', '.join(sorted(shared))}",
                    "confidence": round(len(shared) / max(
                        len(set.union(*all_tokens)), 1
                    ), 4),
                    "supporting_count": len(outputs),
                })

        patterns.sort(key=lambda p: p["confidence"], reverse=True)
        return patterns

    # ── Knowledge base integration ───────────────────────────────────

    def build_from_knowledge(self) -> dict:
        """Build reasoning rules and concept graph from KNOWLEDGE entries.

        Returns
        -------
        dict — build statistics
        """
        start = time.time()
        entries = list(KNOWLEDGE)
        self._entries = entries

        # Build vectorizer
        questions = [q for q, _, _ in entries]
        self._bow = BagOfWords()
        self._bow.fit(questions + [a for _, a, _ in entries])
        self._entry_vectors = self._bow.transform(questions)

        # Build rules
        self.rules = self.build_rules_from_entries(entries)

        # Build concept graph
        self.concepts = self.build_concept_graph(entries)

        # Clear caches
        self.inference_cache.clear()
        self._cache_ts.clear()

        elapsed = time.time() - start

        stats = {
            "rules_count": len(self.rules),
            "concepts_count": len(self.concepts),
            "entries_count": len(entries),
            "domains": list(set(d for _, _, d in entries)),
            "build_time_seconds": round(elapsed, 3),
            "built_at": datetime.now(timezone.utc).isoformat(),
        }

        # Update config
        self._config["last_build"] = stats["built_at"]
        self._config["build_count"] = self._config.get("build_count", 0) + 1
        self._config["stats"] = stats
        save_reasoning_config(self._config)

        return stats

    def build_rules_from_entries(
        self, entries: list[tuple[str, str, str]]
    ) -> list[ReasoningRule]:
        """Extract if-then rules from Q&A entries.

        Parses answers looking for patterns:
          - "X is Y" → identity rule
          - "X is used for Y" → purpose rule
          - "X provides Y" → function rule
          - "X prevents Y" → protection rule
          - "X consists of Y" → composition rule
          - "X enables Y" → enables rule

        Parameters
        ----------
        entries : list of (question, answer, domain) tuples

        Returns
        -------
        list[ReasoningRule]
        """
        min_confidence = self._config.get("min_rule_confidence", 0.40)
        rules: list[ReasoningRule] = []

        for question, answer, domain in entries:
            # Split answer into sentences for finer-grained rule extraction
            sentences = re.split(r"[.;]", answer)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue

                for rule_type, pattern in _RULE_PATTERNS.items():
                    match = pattern.search(sentence)
                    if not match:
                        continue

                    subject = match.group(1).strip()
                    obj = match.group(2).strip()

                    # Skip overly generic matches
                    if len(subject) < 2 or len(obj) < 2:
                        continue
                    if len(tokenize(subject)) == 0 or len(tokenize(obj)) == 0:
                        continue

                    # Confidence heuristic: longer, more specific → higher confidence
                    specificity = min(
                        len(tokenize(subject)) + len(tokenize(obj)), 15
                    ) / 15.0
                    base_confidence = 0.6 + 0.4 * specificity

                    if base_confidence < min_confidence:
                        continue

                    rules.append(ReasoningRule(
                        conditions=[subject],
                        conclusion=f"{subject} {rule_type} {obj}"
                            if rule_type != "identity"
                            else f"{subject} is {obj}",
                        confidence=round(base_confidence, 4),
                        domain=domain,
                        source=question,
                        rule_type=rule_type,
                    ))

        return rules

    def build_concept_graph(
        self, entries: list[tuple[str, str, str]]
    ) -> dict[str, ConceptNode]:
        """Build concept relationship graph from knowledge entries.

        Extracts entities using simple heuristics:
          - Capitalized words / acronyms as candidate concepts
          - Words after "is a", "is the", "called" as definitions
          - Words connected by "and", "or" as related concepts

        Parameters
        ----------
        entries : list of (question, answer, domain) tuples

        Returns
        -------
        dict mapping concept name → ConceptNode
        """
        concepts: dict[str, ConceptNode] = {}

        for question, answer, domain in entries:
            full_text = f"{question} {answer}"

            # Extract candidate concept names
            candidates = _extract_capitalized(full_text)
            # Also include important tokenised terms from the question
            q_tokens = tokenize(question)
            for t in q_tokens:
                if len(t) >= 3:
                    candidates.append(t)

            # Deduplicate
            candidates = list(dict.fromkeys(candidates))

            # Ensure each candidate has a node
            for cand in candidates:
                if cand not in concepts:
                    concepts[cand] = ConceptNode(
                        name=cand, domain=domain, related={}, properties=[]
                    )

            # Add definitions as properties
            definitions = _extract_definitions(answer)
            for cand in candidates[:3]:  # attach to the primary concepts
                if cand in concepts:
                    for defn in definitions:
                        if defn not in concepts[cand].properties:
                            concepts[cand].properties.append(defn)

            # Link co-occurring concepts
            for i, c1 in enumerate(candidates):
                for c2 in candidates[i + 1:]:
                    if c1 == c2:
                        continue
                    # Strengthen relationship based on co-occurrence
                    if c1 in concepts and c2 in concepts:
                        prev = concepts[c1].related.get(c2, 0.0)
                        concepts[c1].related[c2] = min(prev + 0.15, 1.0)
                        prev2 = concepts[c2].related.get(c1, 0.0)
                        concepts[c2].related[c1] = min(prev2 + 0.15, 1.0)

            # Extract "and"/"or" related terms
            related_terms = _extract_related(answer)
            for i, t1 in enumerate(related_terms):
                for t2 in related_terms[i + 1:]:
                    t1_clean = t1.strip()
                    t2_clean = t2.strip()
                    if t1_clean in concepts and t2_clean in concepts:
                        prev = concepts[t1_clean].related.get(t2_clean, 0.0)
                        concepts[t1_clean].related[t2_clean] = min(prev + 0.2, 1.0)

        return concepts

    # ── Persistence ──────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist reasoning state (rules + concept graph) to disk."""
        state = {
            "rules": [r.to_dict() for r in self.rules],
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        REASONING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REASONING_STATE_PATH.write_text(
            json.dumps(state, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def load_state(self) -> None:
        """Load reasoning state from disk."""
        if not REASONING_STATE_PATH.exists():
            return

        try:
            state = json.loads(
                REASONING_STATE_PATH.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return

        self.rules = [
            ReasoningRule.from_dict(r) for r in state.get("rules", [])
        ]
        self.concepts = {
            k: ConceptNode.from_dict(v)
            for k, v in state.get("concepts", {}).items()
        }

        # Rebuild vectorizer from current KNOWLEDGE so vectors are available
        entries = list(KNOWLEDGE)
        self._entries = entries
        questions = [q for q, _, _ in entries]
        self._bow = BagOfWords()
        self._bow.fit(questions + [a for _, a, _ in entries])
        self._entry_vectors = self._bow.transform(questions)

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return current engine statistics.

        Returns
        -------
        dict with counts, domain breakdown, and cache info
        """
        domain_rules: dict[str, int] = defaultdict(int)
        for r in self.rules:
            domain_rules[r.domain] += 1

        domain_concepts: dict[str, int] = defaultdict(int)
        for c in self.concepts.values():
            domain_concepts[c.domain] += 1

        rule_types: dict[str, int] = defaultdict(int)
        for r in self.rules:
            rule_types[r.rule_type] += 1

        avg_relations = 0.0
        if self.concepts:
            avg_relations = sum(
                len(c.related) for c in self.concepts.values()
            ) / len(self.concepts)

        return {
            "rules_count": len(self.rules),
            "concepts_count": len(self.concepts),
            "entries_count": len(self._entries),
            "cache_size": len(self.inference_cache),
            "rules_by_domain": dict(domain_rules),
            "concepts_by_domain": dict(domain_concepts),
            "rules_by_type": dict(rule_types),
            "avg_relations_per_concept": round(avg_relations, 2),
            "config": self._config,
        }


# =====================================================================
# Module-level convenience functions
# =====================================================================

def get_reasoning_engine() -> ReasoningEngine:
    """Return the singleton ReasoningEngine, creating it if needed."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ReasoningEngine()
        # Try loading persisted state first
        if REASONING_STATE_PATH.exists():
            _engine_instance.load_state()
        else:
            _engine_instance.build_from_knowledge()
    return _engine_instance


def reason_about(question: str) -> dict:
    """Quick convenience: reason about a question using the singleton engine.

    Parameters
    ----------
    question : str
        The question to reason about.

    Returns
    -------
    dict with keys: answer, confidence, reasoning_chain, strategy
    """
    engine = get_reasoning_engine()
    return engine.reason(question)


def build_reasoning_base() -> dict:
    """Rebuild the reasoning base from current KNOWLEDGE entries.

    Returns
    -------
    dict — build statistics
    """
    engine = get_reasoning_engine()
    stats = engine.build_from_knowledge()
    engine.save_state()
    return stats
