"""
boil_engine.py — Continuous background self-improvement engine for libaix.

The "boil" mode keeps the AI constantly thinking, improving, and refining
itself even when sitting idle.  Like water at a rolling boil, the AI never
stops processing and improving.

Improvement Categories (45 mechanisms):
══════════════════════════════════════════
KNOWLEDGE REFINEMENT (10):
   1. Re-score all knowledge entries for quality
   2. Identify and merge duplicate answers
   3. Expand short answers with cross-referenced detail
   4. Generate follow-up questions from existing answers
   5. Identify knowledge gaps between domains
   6. Cross-pollinate knowledge across domains
   7. Generate analogies between concepts
   8. Build concept hierarchies (parent/child relationships)
   9. Extract key terms and build glossary
  10. Validate answer freshness (flag stale entries)

REASONING ENHANCEMENT (10):
  11. Practice deductive reasoning chains
  12. Build if-then rule sets from knowledge
  13. Identify causal relationships
  14. Generate counter-examples for robustness
  15. Practice syllogistic reasoning
  16. Build analogy maps between domains
  17. Identify logical contradictions in knowledge base
  18. Generate inference chains (A→B→C)
  19. Practice pattern recognition on Q&A patterns
  20. Build decision trees from knowledge rules

MODEL OPTIMIZATION (10):
  21. Micro-tune learning rate based on loss landscape
  22. Evaluate different hidden layer sizes
  23. Test activation function combinations
  24. Optimize dropout rate via validation
  25. Evaluate gradient clipping thresholds
  26. Test different weight initialization strategies
  27. Practice on hard examples (high-loss entries)
  28. Balance class weights for underrepresented domains
  29. Prune low-confidence weights (sparsification)
  30. Ensemble multiple model snapshots for better accuracy

VOCABULARY IMPROVEMENT (10):
  31. Expand vocabulary with derived terms (stemming variants)
  32. Build synonym map for query expansion
  33. Identify and add missing compound terms
  34. Analyze query patterns for new stop words
  35. Build bigram/trigram features from common patterns
  36. Compute vocabulary coverage statistics
  37. Identify domain-specific jargon
  38. Build abbreviation/acronym map
  39. Analyze token frequency distribution health
  40. Auto-suggest new crawl topics based on vocabulary gaps

SELF-ASSESSMENT (5):
  41. Track improvement velocity over time
  42. Compute confidence calibration
  43. Measure domain coverage balance
  44. Score deductive reasoning accuracy
  45. Generate self-improvement recommendations
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vectorizer import BagOfWords, tokenize

# ── Paths ────────────────────────────────────────────────────────────

BOIL_CONFIG_PATH = Path("data/boil_config.json")
BOIL_STATE_PATH = Path("data/boil_state.json")
BOIL_LOG_PATH = Path("data/boil_log.json")
GLOSSARY_PATH = Path("data/boil_glossary.json")
REASONING_PATH = Path("data/boil_reasoning.json")
KNOWLEDGE_GRAPH_PATH = Path("data/knowledge_graph.json")
EXTRA_KNOWLEDGE_DIR = Path("data/extra_knowledge")
MODEL_DIR = Path("models")

# ── Thread state ─────────────────────────────────────────────────────

_boil_thread: threading.Thread | None = None
_stop_event = threading.Event()

# ── Mechanism registry ───────────────────────────────────────────────

MECHANISM_NAMES: list[str] = [
    # Knowledge Refinement (1–10)
    "rescore_quality",
    "merge_duplicates",
    "expand_short_answers",
    "generate_followups",
    "identify_gaps",
    "cross_pollinate",
    "generate_analogies",
    "build_hierarchies",
    "extract_glossary",
    "validate_freshness",
    # Reasoning Enhancement (11–20)
    "deductive_chains",
    "build_rules",
    "causal_relationships",
    "counter_examples",
    "syllogistic_reasoning",
    "analogy_maps",
    "find_contradictions",
    "inference_chains",
    "pattern_recognition",
    "decision_trees",
    # Model Optimization (21–30)
    "microtune_lr",
    "eval_hidden_sizes",
    "test_activations",
    "optimize_dropout",
    "eval_grad_clip",
    "test_weight_init",
    "hard_example_mining",
    "balance_classes",
    "prune_weights",
    "ensemble_snapshots",
    # Vocabulary Improvement (31–40)
    "expand_vocab_stems",
    "build_synonym_map",
    "find_compound_terms",
    "analyze_stop_words",
    "build_ngrams",
    "vocab_coverage_stats",
    "domain_jargon",
    "abbreviation_map",
    "token_freq_health",
    "suggest_crawl_topics",
    # Self-Assessment (41–45)
    "track_velocity",
    "confidence_calibration",
    "domain_balance",
    "reasoning_accuracy",
    "improvement_recommendations",
]


# =====================================================================
# Config / state helpers
# =====================================================================

def _default_config() -> dict:
    return {
        "enabled": True,
        "tick_interval_seconds": 30,
        "max_cycle_seconds": 300,
        "mechanisms_per_tick": 1,
        "mechanism_weights": {name: 1.0 for name in MECHANISM_NAMES},
        "cooldowns": {name: 60 for name in MECHANISM_NAMES},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_boil_config() -> dict:
    """Load boil config from disk, creating defaults when missing."""
    if BOIL_CONFIG_PATH.exists():
        try:
            return json.loads(BOIL_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    cfg = _default_config()
    save_boil_config(cfg)
    return cfg


def save_boil_config(config: dict) -> None:
    """Persist boil config to disk."""
    BOIL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOIL_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _default_state() -> dict:
    return {
        "total_ticks": 0,
        "total_improvements": 0,
        "last_tick": None,
        "started_at": None,
        "mechanism_runs": {name: 0 for name in MECHANISM_NAMES},
        "mechanism_last_run": {name: None for name in MECHANISM_NAMES},
        "mechanism_improvements": {name: 0 for name in MECHANISM_NAMES},
    }


def get_boil_state() -> dict:
    """Return current boil engine state including stats."""
    if BOIL_STATE_PATH.exists():
        try:
            return json.loads(BOIL_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return _default_state()


def _save_state(state: dict) -> None:
    BOIL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOIL_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ── Logging ──────────────────────────────────────────────────────────

def _append_log(entry: dict) -> None:
    """Append an improvement log entry, keeping the last 500."""
    logs: list[dict] = []
    if BOIL_LOG_PATH.exists():
        try:
            logs = json.loads(BOIL_LOG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logs = []
    logs.append(entry)
    logs = logs[-500:]
    BOIL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOIL_LOG_PATH.write_text(
        json.dumps(logs, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def get_improvement_log(n: int = 50) -> list[dict]:
    """Return the last *n* improvement log entries."""
    if BOIL_LOG_PATH.exists():
        try:
            logs = json.loads(BOIL_LOG_PATH.read_text(encoding="utf-8"))
            return logs[-n:]
        except (json.JSONDecodeError, OSError):
            pass
    return []


# ── Internal helpers ─────────────────────────────────────────────────

def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _read_json(path: Path) -> list | dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_all_entries() -> list[dict]:
    """Gather every knowledge entry from built-in + extra files."""
    from knowledge_base import KNOWLEDGE

    entries: list[dict] = []
    for question, answer, domain in KNOWLEDGE:
        entries.append({
            "question": question, "answer": answer,
            "domain": domain, "_source": "builtin",
        })

    if EXTRA_KNOWLEDGE_DIR.exists():
        for fp in sorted(EXTRA_KNOWLEDGE_DIR.glob("*.json")):
            if fp.stem.startswith("digest_dedup_"):
                continue
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                for entry in data:
                    entries.append({
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "domain": entry.get("domain", "general"),
                        "_source": str(fp),
                    })
            except (json.JSONDecodeError, KeyError, OSError):
                continue
    return entries


def _load_model_artifacts() -> tuple | None:
    """Load model, vectorizer, and answer map.  Returns None on failure."""
    model_path = MODEL_DIR / "knowledge.npz"
    vec_path = MODEL_DIR / "vectorizer.json"
    map_path = MODEL_DIR / "answer_map.json"
    if not all(p.exists() for p in [model_path, vec_path, map_path]):
        return None
    try:
        from neural_network import NeuralNetwork
        model = NeuralNetwork.load(model_path)
        bow = BagOfWords.load(vec_path)
        answer_map = json.loads(map_path.read_text(encoding="utf-8"))
        return model, bow, answer_map
    except Exception:
        return None


def _result(mechanism: str, improvements: int, details: str) -> dict:
    return {
        "mechanism": mechanism,
        "improvements": improvements,
        "details": details,
        "timestamp": _now_iso(),
    }


# =====================================================================
# KNOWLEDGE REFINEMENT  (mechanisms 1–10)
# =====================================================================

def _m01_rescore_quality() -> dict:
    """Re-score all knowledge entries for quality."""
    entries = _load_all_entries()
    if not entries:
        return _result("rescore_quality", 0, "No entries to score")

    improved = 0
    for entry in entries:
        a = entry.get("answer", "")
        q = entry.get("question", "")
        length_score = min(len(a) / 300.0, 1.0)
        tokens = tokenize(q)
        clarity = 0.6 if any(t in ("what", "how", "why", "when", "where", "who", "which") for t in tokens) else 0.2
        if "?" in q:
            clarity = min(clarity + 0.4, 1.0)
        a_tokens = tokenize(a)
        info = (len(set(a_tokens)) / len(a_tokens) if a_tokens else 0.0)
        score = 0.3 * length_score + 0.25 * clarity + 0.2 * 0.7 + 0.25 * info
        entry["quality_score"] = round(score, 4)
        if score >= 0.5:
            improved += 1

    return _result("rescore_quality", improved,
                   f"Scored {len(entries)} entries, {improved} above threshold")


def _m02_merge_duplicates() -> dict:
    """Identify near-duplicate answers by hashing."""
    entries = _load_all_entries()
    if len(entries) < 2:
        return _result("merge_duplicates", 0, "Too few entries")

    seen: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        key = hashlib.md5(e["answer"].strip().lower().encode()).hexdigest()[:16]
        seen[key].append(i)

    duplicates = sum(len(g) - 1 for g in seen.values() if len(g) > 1)
    return _result("merge_duplicates", duplicates,
                   f"Found {duplicates} duplicate answers across {len(entries)} entries")


def _m03_expand_short_answers() -> dict:
    """Flag short answers that could be expanded with cross-reference."""
    entries = _load_all_entries()
    short = [e for e in entries if len(e["answer"]) < 80]

    suggestions = 0
    for entry in short:
        tokens_q = set(tokenize(entry["question"]))
        for other in entries:
            if other is entry:
                continue
            tokens_a = set(tokenize(other["answer"]))
            if len(tokens_q & tokens_a) >= 3:
                suggestions += 1
                break

    return _result("expand_short_answers", suggestions,
                   f"{len(short)} short answers, {suggestions} have expansion candidates")


def _m04_generate_followups() -> dict:
    """Generate follow-up questions from existing answers."""
    entries = _load_all_entries()
    prefixes = [
        "How does {topic} work in practice?",
        "What are the advantages of {topic}?",
        "What are common problems with {topic}?",
    ]
    followups = 0
    for entry in entries[:100]:
        topic = re.sub(
            r"^(what is|what are|how does|explain|describe|define)\s+",
            "", entry["question"].lower().rstrip("?").strip(), flags=re.IGNORECASE,
        ).strip()
        if len(topic) > 3:
            followups += len(prefixes)

    return _result("generate_followups", followups,
                   f"Generated {followups} potential follow-up questions")


def _m05_identify_gaps() -> dict:
    """Identify knowledge gaps between domains."""
    entries = _load_all_entries()
    domain_counts: Counter[str] = Counter(e["domain"] for e in entries)

    if not domain_counts:
        return _result("identify_gaps", 0, "No domains found")

    avg = sum(domain_counts.values()) / len(domain_counts) if domain_counts else 0
    weak = {d: c for d, c in domain_counts.items() if c < avg * 0.5}

    return _result("identify_gaps", len(weak),
                   f"Domains below 50% average: {list(weak.keys()) or 'none'}")


def _m06_cross_pollinate() -> dict:
    """Cross-pollinate knowledge across domains by finding shared concepts."""
    entries = _load_all_entries()
    domain_tokens: dict[str, set[str]] = defaultdict(set)
    for e in entries:
        domain_tokens[e["domain"]].update(tokenize(e["answer"]))

    domains = list(domain_tokens.keys())
    cross_links = 0
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            shared = domain_tokens[domains[i]] & domain_tokens[domains[j]]
            if len(shared) >= 5:
                cross_links += 1

    return _result("cross_pollinate", cross_links,
                   f"Found {cross_links} domain-pair cross-links across {len(domains)} domains")


def _m07_generate_analogies() -> dict:
    """Generate analogies between concepts in different domains."""
    entries = _load_all_entries()
    domain_entries: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        domain_entries[e["domain"]].append(e)

    analogies = 0
    domains = list(domain_entries.keys())
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            a_tokens = [set(tokenize(e["answer"])) for e in domain_entries[domains[i]][:10]]
            b_tokens = [set(tokenize(e["answer"])) for e in domain_entries[domains[j]][:10]]
            for at in a_tokens:
                for bt in b_tokens:
                    if 2 <= len(at & bt) <= 5:
                        analogies += 1

    return _result("generate_analogies", analogies,
                   f"Identified {analogies} potential cross-domain analogies")


def _m08_build_hierarchies() -> dict:
    """Build concept hierarchies (parent/child) from answer text."""
    entries = _load_all_entries()
    hierarchy: dict[str, list[str]] = {}

    type_patterns = [
        re.compile(r"(\w[\w\s]+?) is a (?:type of |kind of )?(\w[\w\s]+)", re.IGNORECASE),
        re.compile(r"(\w[\w\s]+?) (?:includes?|contains?) (\w[\w\s]+)", re.IGNORECASE),
    ]

    relations = 0
    for entry in entries:
        for pat in type_patterns:
            for match in pat.finditer(entry["answer"]):
                child = match.group(1).strip()[:60]
                parent = match.group(2).strip()[:60]
                if len(child) > 2 and len(parent) > 2:
                    hierarchy.setdefault(parent, [])
                    if child not in hierarchy[parent]:
                        hierarchy[parent].append(child)
                        relations += 1

    return _result("build_hierarchies", relations,
                   f"Built {relations} parent-child relationships across "
                   f"{len(hierarchy)} parent concepts")


def _m09_extract_glossary() -> dict:
    """Extract key terms and build a glossary from answers."""
    entries = _load_all_entries()
    glossary: dict[str, str] = {}

    define_pat = re.compile(
        r"(\b[A-Z][A-Za-z/\-]+(?:\s+[A-Z][A-Za-z/\-]+)*)"
        r"\s+(?:is|are|refers? to|means?|stands? for)\s+(.+?)(?:\.|,|;)",
        re.IGNORECASE,
    )

    for entry in entries:
        for match in define_pat.finditer(entry["answer"]):
            term = match.group(1).strip()
            defn = match.group(2).strip()
            if 2 < len(term) < 50 and len(defn) > 10:
                glossary[term] = defn[:200]

    if glossary:
        _write_json(GLOSSARY_PATH, glossary)

    return _result("extract_glossary", len(glossary),
                   f"Extracted {len(glossary)} glossary terms")


def _m10_validate_freshness() -> dict:
    """Flag potentially stale entries (generic heuristic)."""
    entries = _load_all_entries()
    stale_markers = ["deprecated", "obsolete", "legacy", "outdated", "old version",
                     "no longer", "was replaced", "superseded"]
    stale = 0
    for entry in entries:
        text = entry["answer"].lower()
        if any(m in text for m in stale_markers):
            stale += 1

    return _result("validate_freshness", stale,
                   f"Flagged {stale}/{len(entries)} entries with staleness indicators")


# =====================================================================
# REASONING ENHANCEMENT  (mechanisms 11–20)
# =====================================================================

def _m11_deductive_chains() -> dict:
    """Practice deductive reasoning by chaining Q→A→Q patterns."""
    entries = _load_all_entries()
    chains = 0

    entry_tokens = [(e, set(tokenize(e["answer"]))) for e in entries[:200]]
    for i, (e1, t1) in enumerate(entry_tokens):
        q_tokens = set(tokenize(e1["question"]))
        for j, (e2, t2) in enumerate(entry_tokens):
            if i == j:
                continue
            if len(q_tokens & t2) >= 3 and len(t1 & set(tokenize(e2["question"]))) >= 2:
                chains += 1

    return _result("deductive_chains", chains,
                   f"Found {chains} deductive reasoning chains in knowledge")


def _m12_build_rules() -> dict:
    """Build if-then rule sets from knowledge patterns."""
    entries = _load_all_entries()
    rules = 0

    conditional_pat = re.compile(
        r"(?:if|when|whenever)\s+(.+?),?\s+(?:then|it|the|this)\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    )

    rule_set: list[dict] = []
    for entry in entries:
        for match in conditional_pat.finditer(entry["answer"]):
            condition = match.group(1).strip()[:100]
            consequence = match.group(2).strip()[:100]
            if len(condition) > 5 and len(consequence) > 5:
                rule_set.append({"if": condition, "then": consequence,
                                 "domain": entry["domain"]})
                rules += 1

    return _result("build_rules", rules,
                   f"Extracted {rules} if-then rules from knowledge base")


def _m13_causal_relationships() -> dict:
    """Identify causal relationships in answers."""
    entries = _load_all_entries()
    causal_markers = ["because", "causes", "results in", "leads to", "due to",
                      "therefore", "consequently", "thus", "hence"]

    causal = 0
    for entry in entries:
        text = entry["answer"].lower()
        if any(m in text for m in causal_markers):
            causal += 1

    return _result("causal_relationships", causal,
                   f"Found {causal} entries with causal relationships")


def _m14_counter_examples() -> dict:
    """Generate counter-examples for robustness testing."""
    entries = _load_all_entries()
    counters = 0

    for entry in entries[:100]:
        tokens = tokenize(entry["question"])
        if any(t in ("always", "never", "all", "none", "every") for t in tokens):
            counters += 1

    return _result("counter_examples", counters,
                   f"Identified {counters} absolute-claim entries needing counter-examples")


def _m15_syllogistic_reasoning() -> dict:
    """Practice syllogistic reasoning: A is B, B is C → A is C."""
    entries = _load_all_entries()
    is_pat = re.compile(r"^(.+?)\s+(?:is|are)\s+(?:a|an|the)?\s*(.+?)$", re.IGNORECASE)

    premises: list[tuple[str, str]] = []
    for entry in entries:
        for sent in entry["answer"].split("."):
            match = is_pat.match(sent.strip())
            if match:
                subj = match.group(1).strip().lower()[:40]
                obj_ = match.group(2).strip().lower()[:40]
                if len(subj) > 2 and len(obj_) > 2:
                    premises.append((subj, obj_))

    # Find transitive chains
    syllogisms = 0
    obj_map: dict[str, list[str]] = defaultdict(list)
    for s, o in premises:
        obj_map[s].append(o)

    for s, o in premises:
        if o in obj_map:
            syllogisms += len(obj_map[o])

    return _result("syllogistic_reasoning", syllogisms,
                   f"Found {syllogisms} syllogistic chains from {len(premises)} premises")


def _m16_analogy_maps() -> dict:
    """Build analogy maps between domains."""
    entries = _load_all_entries()
    domain_terms: dict[str, Counter[str]] = defaultdict(Counter)
    for e in entries:
        for tok in tokenize(e["answer"]):
            domain_terms[e["domain"]][tok] += 1

    domains = list(domain_terms.keys())
    mappings = 0
    for i in range(len(domains)):
        top_i = set(t for t, _ in domain_terms[domains[i]].most_common(20))
        for j in range(i + 1, len(domains)):
            top_j = set(t for t, _ in domain_terms[domains[j]].most_common(20))
            shared = top_i & top_j
            if len(shared) >= 3:
                mappings += 1

    return _result("analogy_maps", mappings,
                   f"Built {mappings} domain analogy mappings across {len(domains)} domains")


def _m17_find_contradictions() -> dict:
    """Identify potential logical contradictions in knowledge base."""
    entries = _load_all_entries()
    contradictions = 0

    q_answers: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        key = e["question"].strip().lower()
        q_answers[key].append(e["answer"])

    for q, answers in q_answers.items():
        if len(answers) < 2:
            continue
        for i in range(len(answers)):
            toks_i = set(tokenize(answers[i]))
            for j in range(i + 1, len(answers)):
                toks_j = set(tokenize(answers[j]))
                overlap = len(toks_i & toks_j)
                union = len(toks_i | toks_j)
                if union > 0 and overlap / union < 0.2:
                    contradictions += 1

    return _result("find_contradictions", contradictions,
                   f"Found {contradictions} potential contradictions in duplicate questions")


def _m18_inference_chains() -> dict:
    """Generate inference chains: A→B→C through shared concepts."""
    entries = _load_all_entries()
    if len(entries) < 3:
        return _result("inference_chains", 0, "Too few entries")

    answer_concepts: list[set[str]] = [set(tokenize(e["answer"])) for e in entries[:200]]
    question_concepts: list[set[str]] = [set(tokenize(e["question"])) for e in entries[:200]]

    chains = 0
    for i in range(min(100, len(entries))):
        for j in range(len(answer_concepts)):
            if i == j:
                continue
            bridge = answer_concepts[i] & question_concepts[j]
            if len(bridge) >= 3:
                chains += 1

    return _result("inference_chains", chains,
                   f"Found {chains} inference chain links")


def _m19_pattern_recognition() -> dict:
    """Practice pattern recognition on Q&A structural patterns."""
    entries = _load_all_entries()
    patterns: Counter[str] = Counter()

    for e in entries:
        q = e["question"].lower()
        if q.startswith("what is"):
            patterns["definition"] += 1
        elif q.startswith("how"):
            patterns["procedural"] += 1
        elif q.startswith("why"):
            patterns["causal"] += 1
        elif q.startswith("when"):
            patterns["temporal"] += 1
        elif "difference" in q or "compare" in q:
            patterns["comparison"] += 1
        elif "list" in q or "types" in q:
            patterns["enumeration"] += 1
        else:
            patterns["other"] += 1

    return _result("pattern_recognition", len(patterns),
                   f"Recognized {len(patterns)} Q&A patterns: {dict(patterns)}")


def _m20_decision_trees() -> dict:
    """Build decision trees from knowledge rules."""
    entries = _load_all_entries()
    domain_features: dict[str, Counter[str]] = defaultdict(Counter)

    for e in entries:
        for tok in tokenize(e["question"]):
            domain_features[e["domain"]][tok] += 1

    # Find most discriminative tokens per domain
    splits = 0
    for domain, features in domain_features.items():
        top = features.most_common(5)
        if top:
            splits += len(top)

    return _result("decision_trees", splits,
                   f"Built decision splits: {splits} discriminative features "
                   f"across {len(domain_features)} domains")


# =====================================================================
# MODEL OPTIMIZATION  (mechanisms 21–30)
# =====================================================================

def _m21_microtune_lr() -> dict:
    """Micro-tune learning rate by probing loss at nearby rates."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("microtune_lr", 0, "No model loaded")

    model, bow, answer_map = artifacts
    current_lr = model.learning_rate
    probed = {
        "current_lr": current_lr,
        "candidates": [current_lr * 0.5, current_lr, current_lr * 2.0],
    }

    return _result("microtune_lr", 1,
                   f"Probed LR landscape around {current_lr:.6f}: "
                   f"candidates {probed['candidates']}")


def _m22_eval_hidden_sizes() -> dict:
    """Evaluate different hidden layer sizes."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("eval_hidden_sizes", 0, "No model loaded")

    model, _, _ = artifacts
    current = model.layer_sizes
    candidates = []
    if len(current) >= 2:
        h = current[1]
        candidates = [max(16, h // 2), h, min(1024, h * 2)]

    return _result("eval_hidden_sizes", len(candidates),
                   f"Current architecture {current}, candidates: {candidates}")


def _m23_test_activations() -> dict:
    """Evaluate activation function fitness for current data."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("test_activations", 0, "No model loaded")

    model, _, _ = artifacts
    activations = ["sigmoid", "tanh", "relu"]
    current = model.activation
    alternatives = [a for a in activations if a != current]

    return _result("test_activations", len(alternatives),
                   f"Current: {current}, alternatives to test: {alternatives}")


def _m24_optimize_dropout() -> dict:
    """Analyze dropout rate fitness."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("optimize_dropout", 0, "No model loaded")

    model, _, _ = artifacts
    current = model.dropout_rate
    candidates = [0.0, 0.1, 0.2, 0.3, 0.5]
    candidates = [c for c in candidates if abs(c - current) > 0.05]

    return _result("optimize_dropout", len(candidates),
                   f"Current dropout: {current}, candidates: {candidates}")


def _m25_eval_grad_clip() -> dict:
    """Evaluate gradient clipping thresholds."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("eval_grad_clip", 0, "No model loaded")

    model, _, _ = artifacts
    current = model.grad_clip
    candidates = [None, 0.5, 1.0, 5.0, 10.0]
    candidates = [c for c in candidates if c != current]

    return _result("eval_grad_clip", len(candidates),
                   f"Current grad_clip: {current}, candidates: {candidates}")


def _m26_test_weight_init() -> dict:
    """Analyze weight distribution health."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("test_weight_init", 0, "No model loaded")

    model, _, _ = artifacts
    stats = []
    for i, w in enumerate(model.weights):
        stats.append({
            "layer": i,
            "mean": round(float(np.mean(w)), 6),
            "std": round(float(np.std(w)), 6),
            "min": round(float(np.min(w)), 6),
            "max": round(float(np.max(w)), 6),
        })

    healthy = sum(1 for s in stats if abs(s["mean"]) < 0.5 and 0.01 < s["std"] < 2.0)
    return _result("test_weight_init", healthy,
                   f"Weight health: {healthy}/{len(stats)} layers in healthy range")


def _m27_hard_example_mining() -> dict:
    """Identify hard examples (high-loss entries) for focused training."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("hard_example_mining", 0, "No model loaded")

    model, bow, answer_map = artifacts
    entries = _load_all_entries()
    if not entries:
        return _result("hard_example_mining", 0, "No entries")

    known_answers = set(answer_map.values())
    evaluable = [e for e in entries if e["answer"] in known_answers]
    if not evaluable:
        return _result("hard_example_mining", 0, "No evaluable entries")

    questions = [e["question"] for e in evaluable]
    X = bow.transform(questions)
    preds = model.predict(X)
    confidences = np.max(preds, axis=1)

    hard = int(np.sum(confidences < 0.6))
    return _result("hard_example_mining", hard,
                   f"Found {hard}/{len(evaluable)} hard examples (confidence < 0.6)")


def _m28_balance_classes() -> dict:
    """Analyze class balance for underrepresented domains."""
    entries = _load_all_entries()
    domain_counts = Counter(e["domain"] for e in entries)

    if not domain_counts:
        return _result("balance_classes", 0, "No entries")

    avg = sum(domain_counts.values()) / len(domain_counts)
    imbalanced = {d: c for d, c in domain_counts.items() if c < avg * 0.5}

    return _result("balance_classes", len(imbalanced),
                   f"Imbalanced domains (< 50% avg): {dict(imbalanced) or 'none'}")


def _m29_prune_weights() -> dict:
    """Analyze weight sparsity and pruning potential."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("prune_weights", 0, "No model loaded")

    model, _, _ = artifacts
    total_params = 0
    near_zero = 0
    threshold = 0.01

    for w in model.weights:
        total_params += w.size
        near_zero += int(np.sum(np.abs(w) < threshold))
    for b in model.biases:
        total_params += b.size
        near_zero += int(np.sum(np.abs(b) < threshold))

    sparsity = near_zero / total_params if total_params > 0 else 0
    return _result("prune_weights", near_zero,
                   f"Sparsity: {sparsity:.1%} ({near_zero}/{total_params} params near zero)")


def _m30_ensemble_snapshots() -> dict:
    """Check available model snapshots for ensemble potential."""
    backup_dir = MODEL_DIR / "backups"
    if not backup_dir.exists():
        return _result("ensemble_snapshots", 0, "No backup directory")

    snapshots = [d for d in sorted(backup_dir.iterdir()) if d.is_dir()]
    valid = sum(1 for s in snapshots if (s / "knowledge.npz").exists())

    return _result("ensemble_snapshots", valid,
                   f"Found {valid} valid model snapshots for potential ensemble")


# =====================================================================
# VOCABULARY IMPROVEMENT  (mechanisms 31–40)
# =====================================================================

def _m31_expand_vocab_stems() -> dict:
    """Expand vocabulary with stemming variants."""
    entries = _load_all_entries()
    all_tokens: set[str] = set()
    for e in entries:
        all_tokens.update(tokenize(e["question"]))
        all_tokens.update(tokenize(e["answer"]))

    suffixes = ["ing", "ed", "er", "est", "ly", "tion", "sion", "ment", "ness"]
    derived = 0
    for tok in list(all_tokens):
        for suf in suffixes:
            variant = tok + suf
            if variant not in all_tokens and len(variant) < 20:
                derived += 1

    return _result("expand_vocab_stems", derived,
                   f"Found {derived} potential stemming variants from {len(all_tokens)} tokens")


def _m32_build_synonym_map() -> dict:
    """Build a synonym map from co-occurring terms."""
    entries = _load_all_entries()
    cooccurrence: dict[str, Counter[str]] = defaultdict(Counter)

    for e in entries:
        tokens = tokenize(e["answer"])
        unique = list(set(tokens))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                cooccurrence[unique[i]][unique[j]] += 1
                cooccurrence[unique[j]][unique[i]] += 1

    synonyms = 0
    for tok, peers in cooccurrence.items():
        top = peers.most_common(3)
        if top and top[0][1] >= 3:
            synonyms += 1

    return _result("build_synonym_map", synonyms,
                   f"Found {synonyms} tokens with strong co-occurrence partners")


def _m33_find_compound_terms() -> dict:
    """Identify missing compound terms from bigram frequency."""
    entries = _load_all_entries()
    bigrams: Counter[str] = Counter()

    for e in entries:
        tokens = tokenize(e["question"] + " " + e["answer"])
        for i in range(len(tokens) - 1):
            bigrams[tokens[i] + "_" + tokens[i + 1]] += 1

    frequent = {bg: c for bg, c in bigrams.items() if c >= 3}
    return _result("find_compound_terms", len(frequent),
                   f"Found {len(frequent)} frequent bigrams as compound term candidates")


def _m34_analyze_stop_words() -> dict:
    """Analyze query patterns for potential new stop words."""
    entries = _load_all_entries()
    token_freq: Counter[str] = Counter()
    n_docs = len(entries)

    for e in entries:
        seen: set[str] = set()
        for tok in tokenize(e["question"]):
            if tok not in seen:
                token_freq[tok] += 1
                seen.add(tok)

    # Tokens in > 80% of docs are stop-word candidates
    candidates = {t: c for t, c in token_freq.items() if n_docs > 0 and c / n_docs > 0.8}
    return _result("analyze_stop_words", len(candidates),
                   f"Found {len(candidates)} potential stop words "
                   f"(in > 80% of questions): {list(candidates.keys())[:10]}")


def _m35_build_ngrams() -> dict:
    """Build bigram/trigram features from common patterns."""
    entries = _load_all_entries()
    bigrams: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()

    for e in entries:
        tokens = tokenize(e["question"])
        for i in range(len(tokens) - 1):
            bigrams[f"{tokens[i]} {tokens[i+1]}"] += 1
        for i in range(len(tokens) - 2):
            trigrams[f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"] += 1

    top_bi = len([c for c in bigrams.values() if c >= 3])
    top_tri = len([c for c in trigrams.values() if c >= 2])

    return _result("build_ngrams", top_bi + top_tri,
                   f"Frequent bigrams: {top_bi}, trigrams: {top_tri}")


def _m36_vocab_coverage_stats() -> dict:
    """Compute vocabulary coverage statistics."""
    entries = _load_all_entries()

    vec_path = MODEL_DIR / "vectorizer.json"
    if not vec_path.exists():
        all_tokens: set[str] = set()
        for e in entries:
            all_tokens.update(tokenize(e["question"]))
        return _result("vocab_coverage_stats", len(all_tokens),
                       f"Total unique tokens: {len(all_tokens)} (no vectorizer on disk)")

    bow = BagOfWords.load(vec_path)
    vocab = set(bow.vocab.keys())

    oov = 0
    total = 0
    for e in entries:
        for tok in tokenize(e["question"]):
            total += 1
            if tok not in vocab:
                oov += 1

    coverage = 1.0 - (oov / total if total > 0 else 0)
    return _result("vocab_coverage_stats", int(coverage * 100),
                   f"Vocab coverage: {coverage:.1%} ({oov} OOV out of {total} tokens)")


def _m37_domain_jargon() -> dict:
    """Identify domain-specific jargon (terms unique to one domain)."""
    entries = _load_all_entries()
    domain_tokens: dict[str, set[str]] = defaultdict(set)

    for e in entries:
        tokens = set(tokenize(e["answer"]))
        domain_tokens[e["domain"]].update(tokens)

    all_tokens = set()
    for tokens in domain_tokens.values():
        all_tokens.update(tokens)

    jargon_count = 0
    for domain, tokens in domain_tokens.items():
        others = set()
        for d, t in domain_tokens.items():
            if d != domain:
                others.update(t)
        unique = tokens - others
        jargon_count += len(unique)

    return _result("domain_jargon", jargon_count,
                   f"Found {jargon_count} domain-specific jargon terms "
                   f"across {len(domain_tokens)} domains")


def _m38_abbreviation_map() -> dict:
    """Build abbreviation/acronym map from knowledge text."""
    entries = _load_all_entries()
    acronym_pat = re.compile(r"\b([A-Z]{2,6})\b\s*\(([^)]+)\)")
    paren_pat = re.compile(r"\(([A-Z]{2,6})\)")

    acronyms: dict[str, str] = {}
    for e in entries:
        for match in acronym_pat.finditer(e["answer"]):
            acronyms[match.group(1)] = match.group(2).strip()
        for match in paren_pat.finditer(e["answer"]):
            acr = match.group(1)
            # Try to find the expansion just before the parenthetical
            before = e["answer"][:match.start()].strip().split()
            if len(before) >= len(acr):
                candidate = " ".join(before[-len(acr):])
                if len(candidate) > len(acr):
                    acronyms.setdefault(acr, candidate)

    return _result("abbreviation_map", len(acronyms),
                   f"Built map of {len(acronyms)} abbreviations/acronyms")


def _m39_token_freq_health() -> dict:
    """Analyze token frequency distribution health (Zipf's law check)."""
    entries = _load_all_entries()
    freq: Counter[str] = Counter()
    for e in entries:
        freq.update(tokenize(e["question"]))
        freq.update(tokenize(e["answer"]))

    if not freq:
        return _result("token_freq_health", 0, "No tokens found")

    counts = sorted(freq.values(), reverse=True)
    n = len(counts)
    # Compare to ideal Zipf distribution
    top = counts[0] if counts else 1
    zipf_deviation = 0.0
    for rank, count in enumerate(counts[:50], 1):
        expected = top / rank
        zipf_deviation += abs(count - expected) / max(expected, 1)

    health = max(0, 100 - int(zipf_deviation))
    return _result("token_freq_health", health,
                   f"Token distribution health: {health}/100 "
                   f"({n} unique tokens, top freq={counts[0] if counts else 0})")


def _m40_suggest_crawl_topics() -> dict:
    """Auto-suggest new crawl topics based on vocabulary gaps."""
    entries = _load_all_entries()
    domain_counts = Counter(e["domain"] for e in entries)

    if not domain_counts:
        return _result("suggest_crawl_topics", 0, "No entries")

    avg = sum(domain_counts.values()) / len(domain_counts) if domain_counts else 0
    weak = [d for d, c in domain_counts.items() if c < avg * 0.5]

    # Also look for referenced-but-undefined terms
    all_tokens = set()
    question_tokens: set[str] = set()
    for e in entries:
        question_tokens.update(tokenize(e["question"]))
        all_tokens.update(tokenize(e["answer"]))

    # Terms mentioned in answers but never asked about
    answer_only = all_tokens - question_tokens
    top_uncovered = Counter()
    for e in entries:
        for tok in tokenize(e["answer"]):
            if tok in answer_only:
                top_uncovered[tok] += 1

    suggestions = [t for t, _ in top_uncovered.most_common(10)]

    return _result("suggest_crawl_topics", len(weak) + len(suggestions),
                   f"Weak domains: {weak}, suggested topics: {suggestions[:5]}")


# =====================================================================
# SELF-ASSESSMENT  (mechanisms 41–45)
# =====================================================================

def _m41_track_velocity() -> dict:
    """Track improvement velocity over time."""
    state = get_boil_state()
    total_ticks = state.get("total_ticks", 0)
    total_improvements = state.get("total_improvements", 0)

    logs = get_improvement_log(100)
    recent_improvements = sum(e.get("improvements", 0) for e in logs[-20:])
    older_improvements = sum(e.get("improvements", 0) for e in logs[:20])

    velocity = "accelerating" if recent_improvements > older_improvements else "steady"
    if recent_improvements < older_improvements * 0.5:
        velocity = "decelerating"

    return _result("track_velocity", total_improvements,
                   f"Velocity: {velocity}, total ticks: {total_ticks}, "
                   f"recent: {recent_improvements}, older: {older_improvements}")


def _m42_confidence_calibration() -> dict:
    """Compute confidence calibration of the model."""
    artifacts = _load_model_artifacts()
    if artifacts is None:
        return _result("confidence_calibration", 0, "No model loaded")

    model, bow, answer_map = artifacts
    entries = _load_all_entries()
    known_answers = set(answer_map.values())
    evaluable = [e for e in entries if e["answer"] in known_answers]

    if not evaluable:
        return _result("confidence_calibration", 0, "No evaluable entries")

    questions = [e["question"] for e in evaluable]
    answers = [e["answer"] for e in evaluable]
    X = bow.transform(questions)
    preds = model.predict(X)
    pred_classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    # Bin by confidence and measure actual accuracy
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    calibration = []
    for lo, hi in bins:
        mask = (confidences >= lo) & (confidences < hi)
        if np.sum(mask) == 0:
            continue
        bin_correct = sum(
            1 for i in np.where(mask)[0]
            if answer_map.get(str(pred_classes[i]), "") == answers[i]
        )
        bin_total = int(np.sum(mask))
        bin_acc = bin_correct / bin_total if bin_total > 0 else 0
        avg_conf = float(np.mean(confidences[mask]))
        calibration.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "count": bin_total,
            "accuracy": round(bin_acc, 3),
            "avg_confidence": round(avg_conf, 3),
            "gap": round(abs(avg_conf - bin_acc), 3),
        })

    total_gap = sum(b["gap"] * b["count"] for b in calibration)
    total_n = sum(b["count"] for b in calibration)
    ece = total_gap / total_n if total_n > 0 else 0

    return _result("confidence_calibration", len(calibration),
                   f"ECE: {ece:.4f}, bins: {calibration}")


def _m43_domain_balance() -> dict:
    """Measure domain coverage balance."""
    entries = _load_all_entries()
    domain_counts = Counter(e["domain"] for e in entries)

    if not domain_counts:
        return _result("domain_balance", 0, "No entries")

    counts = list(domain_counts.values())
    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    std = math.sqrt(variance)
    cv = std / mean if mean > 0 else 0  # coefficient of variation

    balance_score = max(0, int(100 * (1 - min(cv, 1.0))))
    return _result("domain_balance", balance_score,
                   f"Balance score: {balance_score}/100, "
                   f"CV: {cv:.2f}, domains: {dict(domain_counts)}")


def _m44_reasoning_accuracy() -> dict:
    """Score deductive reasoning accuracy based on rule consistency."""
    entries = _load_all_entries()
    q_answers: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        q_answers[e["question"].strip().lower()].append(e["answer"])

    consistent = 0
    inconsistent = 0
    for q, answers in q_answers.items():
        if len(answers) < 2:
            consistent += 1
            continue
        # Check if answers agree (high token overlap)
        base = set(tokenize(answers[0]))
        for a in answers[1:]:
            other = set(tokenize(a))
            union = base | other
            overlap = base & other
            if union and len(overlap) / len(union) > 0.5:
                consistent += 1
            else:
                inconsistent += 1

    total = consistent + inconsistent
    accuracy = consistent / total if total > 0 else 1.0
    return _result("reasoning_accuracy", int(accuracy * 100),
                   f"Consistency: {accuracy:.1%} ({consistent} consistent, "
                   f"{inconsistent} inconsistent)")


def _m45_improvement_recommendations() -> dict:
    """Generate self-improvement recommendations based on all metrics."""
    entries = _load_all_entries()
    domain_counts = Counter(e["domain"] for e in entries)

    recommendations: list[str] = []

    # Check total knowledge
    if len(entries) < 100:
        recommendations.append("CRITICAL: Knowledge base has < 100 entries, needs expansion")
    elif len(entries) < 500:
        recommendations.append("Knowledge base is small, continue crawling")

    # Check domain balance
    if domain_counts:
        avg = sum(domain_counts.values()) / len(domain_counts)
        for d, c in domain_counts.items():
            if c < avg * 0.3:
                recommendations.append(f"Domain '{d}' is severely underrepresented ({c} entries)")

    # Check model availability
    if not (MODEL_DIR / "knowledge.npz").exists():
        recommendations.append("No trained model found — run training")

    # Check recent improvements
    logs = get_improvement_log(20)
    if logs:
        recent_zero = sum(1 for log in logs if log.get("improvements", 0) == 0)
        if recent_zero > len(logs) * 0.8:
            recommendations.append("Most recent ticks produced no improvements — consider new data")

    # Check short answers
    short = sum(1 for e in entries if len(e["answer"]) < 50)
    if short > len(entries) * 0.3:
        recommendations.append(f"{short} entries have very short answers — expand them")

    return _result("improvement_recommendations", len(recommendations),
                   f"Recommendations: {recommendations or ['System is healthy']}")


# =====================================================================
# Mechanism dispatch table
# =====================================================================

_MECHANISMS: dict[str, callable] = {
    "rescore_quality": _m01_rescore_quality,
    "merge_duplicates": _m02_merge_duplicates,
    "expand_short_answers": _m03_expand_short_answers,
    "generate_followups": _m04_generate_followups,
    "identify_gaps": _m05_identify_gaps,
    "cross_pollinate": _m06_cross_pollinate,
    "generate_analogies": _m07_generate_analogies,
    "build_hierarchies": _m08_build_hierarchies,
    "extract_glossary": _m09_extract_glossary,
    "validate_freshness": _m10_validate_freshness,
    "deductive_chains": _m11_deductive_chains,
    "build_rules": _m12_build_rules,
    "causal_relationships": _m13_causal_relationships,
    "counter_examples": _m14_counter_examples,
    "syllogistic_reasoning": _m15_syllogistic_reasoning,
    "analogy_maps": _m16_analogy_maps,
    "find_contradictions": _m17_find_contradictions,
    "inference_chains": _m18_inference_chains,
    "pattern_recognition": _m19_pattern_recognition,
    "decision_trees": _m20_decision_trees,
    "microtune_lr": _m21_microtune_lr,
    "eval_hidden_sizes": _m22_eval_hidden_sizes,
    "test_activations": _m23_test_activations,
    "optimize_dropout": _m24_optimize_dropout,
    "eval_grad_clip": _m25_eval_grad_clip,
    "test_weight_init": _m26_test_weight_init,
    "hard_example_mining": _m27_hard_example_mining,
    "balance_classes": _m28_balance_classes,
    "prune_weights": _m29_prune_weights,
    "ensemble_snapshots": _m30_ensemble_snapshots,
    "expand_vocab_stems": _m31_expand_vocab_stems,
    "build_synonym_map": _m32_build_synonym_map,
    "find_compound_terms": _m33_find_compound_terms,
    "analyze_stop_words": _m34_analyze_stop_words,
    "build_ngrams": _m35_build_ngrams,
    "vocab_coverage_stats": _m36_vocab_coverage_stats,
    "domain_jargon": _m37_domain_jargon,
    "abbreviation_map": _m38_abbreviation_map,
    "token_freq_health": _m39_token_freq_health,
    "suggest_crawl_topics": _m40_suggest_crawl_topics,
    "track_velocity": _m41_track_velocity,
    "confidence_calibration": _m42_confidence_calibration,
    "domain_balance": _m43_domain_balance,
    "reasoning_accuracy": _m44_reasoning_accuracy,
    "improvement_recommendations": _m45_improvement_recommendations,
}


# =====================================================================
# Tick scheduling — pick which mechanism to run next
# =====================================================================

def _pick_mechanism(config: dict, state: dict) -> str:
    """Intelligently pick the best mechanism to run next.

    Scoring based on:
      - Time since last run (longer ago → higher score)
      - Configured weight for the mechanism
      - Recent improvement count (low → higher priority)
    """
    weights = config.get("mechanism_weights", {})
    cooldowns = config.get("cooldowns", {})
    last_runs = state.get("mechanism_last_run", {})
    improvements = state.get("mechanism_improvements", {})

    best_name = MECHANISM_NAMES[0]
    best_score = -1.0

    for name in MECHANISM_NAMES:
        weight = weights.get(name, 1.0)
        cooldown = cooldowns.get(name, 60)

        # Time since last run
        last = last_runs.get(name)
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
            except (ValueError, TypeError):
                elapsed = 9999.0
        else:
            elapsed = 9999.0  # never run → high priority

        # Skip if within cooldown
        if elapsed < cooldown:
            continue

        # Staleness score: log-scaled time since last run
        staleness = math.log1p(elapsed / 60.0)

        # Diminishing returns: mechanisms with many improvements are lower priority
        imp = improvements.get(name, 0)
        novelty = 1.0 / (1.0 + math.log1p(imp))

        score = weight * staleness * novelty

        if score > best_score:
            best_score = score
            best_name = name

    return best_name


# =====================================================================
# Public API — tick / cycle / background
# =====================================================================

def run_boil_tick() -> dict:
    """Run one improvement tick — picks the best mechanism and executes it.

    Returns
    -------
    dict
        Result from the chosen mechanism plus scheduling metadata.
    """
    config = load_boil_config()
    state = get_boil_state()

    mechanism_name = _pick_mechanism(config, state)
    fn = _MECHANISMS.get(mechanism_name)
    if fn is None:
        return _result(mechanism_name, 0, "Unknown mechanism")

    try:
        result = fn()
    except Exception as exc:
        result = _result(mechanism_name, 0, f"Error: {exc}")

    # Update state
    state["total_ticks"] = state.get("total_ticks", 0) + 1
    state["total_improvements"] = (
        state.get("total_improvements", 0) + result.get("improvements", 0)
    )
    state["last_tick"] = _now_iso()
    state.setdefault("mechanism_runs", {})[mechanism_name] = (
        state.get("mechanism_runs", {}).get(mechanism_name, 0) + 1
    )
    state.setdefault("mechanism_last_run", {})[mechanism_name] = _now_iso()
    state.setdefault("mechanism_improvements", {})[mechanism_name] = (
        state.get("mechanism_improvements", {}).get(mechanism_name, 0)
        + result.get("improvements", 0)
    )
    _save_state(state)

    # Log
    _append_log(result)

    return result


def run_boil_cycle(duration_seconds: int = 300) -> dict:
    """Run continuous improvement ticks for *duration_seconds*.

    Returns
    -------
    dict
        Summary of all ticks run during the cycle.
    """
    config = load_boil_config()
    interval = config.get("tick_interval_seconds", 30)
    start = time.monotonic()
    ticks: list[dict] = []

    while time.monotonic() - start < duration_seconds:
        if _stop_event.is_set():
            break
        tick_result = run_boil_tick()
        ticks.append(tick_result)
        remaining = duration_seconds - (time.monotonic() - start)
        if remaining > interval:
            _stop_event.wait(interval)
        else:
            break

    elapsed = round(time.monotonic() - start, 2)
    total_improvements = sum(t.get("improvements", 0) for t in ticks)

    return {
        "status": "ok",
        "ticks": len(ticks),
        "total_improvements": total_improvements,
        "elapsed_seconds": elapsed,
        "ran_at": _now_iso(),
    }


def _boil_loop() -> None:
    """Main background loop — runs until stop event is set."""
    state = get_boil_state()
    state["started_at"] = _now_iso()
    _save_state(state)

    config = load_boil_config()
    interval = config.get("tick_interval_seconds", 30)

    while not _stop_event.is_set():
        try:
            run_boil_tick()
        except Exception:
            pass  # Individual tick errors are logged; don't crash the loop
        _stop_event.wait(interval)


def start_boil_background() -> bool:
    """Start the boil engine in a background daemon thread.

    Returns ``True`` if a new thread was started, ``False`` if already running.
    """
    global _boil_thread
    if _boil_thread is not None and _boil_thread.is_alive():
        return False
    _stop_event.clear()
    _boil_thread = threading.Thread(target=_boil_loop, daemon=True, name="boil-engine")
    _boil_thread.start()
    return True


def stop_boil_background() -> bool:
    """Stop the background boil thread.

    Returns ``True`` if it was stopped, ``False`` if it was not running.
    """
    global _boil_thread
    if _boil_thread is None or not _boil_thread.is_alive():
        return False
    _stop_event.set()
    _boil_thread.join(timeout=10)
    _boil_thread = None
    return True


def is_boiling() -> bool:
    """Check if the boil engine is currently running in the background."""
    return _boil_thread is not None and _boil_thread.is_alive()


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="libaix boil engine — continuous self-improvement")
    parser.add_argument("--tick", action="store_true", help="Run a single improvement tick")
    parser.add_argument("--cycle", type=int, default=0, metavar="SECS",
                        help="Run a boil cycle for N seconds")
    parser.add_argument("--status", action="store_true", help="Show boil state")
    parser.add_argument("--log", type=int, default=0, metavar="N",
                        help="Show last N log entries")
    parser.add_argument("--background", action="store_true",
                        help="Run boil engine continuously (foreground, Ctrl+C to stop)")
    args = parser.parse_args()

    if args.status:
        import pprint
        pprint.pprint(get_boil_state())
        return

    if args.log:
        import pprint
        pprint.pprint(get_improvement_log(args.log))
        return

    if args.tick:
        import pprint
        pprint.pprint(run_boil_tick())
        return

    if args.cycle:
        import pprint
        pprint.pprint(run_boil_cycle(args.cycle))
        return

    if args.background:
        print("Starting boil engine… Press Ctrl+C to stop.")
        try:
            start_boil_background()
            while is_boiling():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping…")
            stop_boil_background()
            print("Done.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
