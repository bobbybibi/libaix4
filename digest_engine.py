"""
digest_engine.py — Digestive-mode data-processing engine for libaix.

Processes existing knowledge data to build better patterns for deductive
reasoning.  Think of it as the AI's "crunching through data" mode that can
run continuously in the background.

Capabilities
────────────
  1. Deduplicate near-identical knowledge entries (cosine similarity on TF-IDF)
  2. Score entry quality (answer length, question clarity, domain specificity)
  3. Cross-reference entries across domains (simple knowledge graph)
  4. Generate derived / synthesized Q&A entries from existing knowledge
  5. Run a full digest cycle orchestrating all of the above

All state is persisted to JSON files inside ``data/``.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vectorizer import BagOfWords, tokenize

# ── Paths ────────────────────────────────────────────────────────────

DIGEST_CONFIG_PATH = Path("data/digest_config.json")
KNOWLEDGE_GRAPH_PATH = Path("data/knowledge_graph.json")
EXTRA_KNOWLEDGE_DIR = Path("data/extra_knowledge")

# ── Question-word set for clarity scoring ────────────────────────────

_QUESTION_WORDS = frozenset(
    "what which who whom whose when where why how "
    "explain describe define compare contrast list name".split()
)


# =====================================================================
# Config helpers
# =====================================================================

def _default_config() -> dict:
    return {
        "dedup_threshold": 0.85,
        "quality_min_score": 0.3,
        "cross_ref_min_shared_terms": 3,
        "derive_similarity_range": [0.3, 0.7],
        "max_derived_per_cycle": 50,
        "last_digest": None,
        "digest_count": 0,
        "stats": {},
    }


def load_digest_config() -> dict:
    """Load digest config from disk, creating defaults when missing."""
    if DIGEST_CONFIG_PATH.exists():
        try:
            return json.loads(DIGEST_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    cfg = _default_config()
    save_digest_config(cfg)
    return cfg


def save_digest_config(config: dict) -> None:
    """Persist digest config to disk."""
    DIGEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIGEST_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# =====================================================================
# Internal helpers
# =====================================================================

def _load_all_entries() -> list[dict]:
    """Gather every knowledge entry from built-in + extra files.

    Returns a list of dicts with keys ``question``, ``answer``, ``domain``,
    and ``_source`` (file path or ``"builtin"``).
    """
    from knowledge_base import KNOWLEDGE

    entries: list[dict] = []

    # Built-in knowledge
    for question, answer, domain in KNOWLEDGE:
        entries.append({
            "question": question,
            "answer": answer,
            "domain": domain,
            "_source": "builtin",
        })

    # Extra-knowledge JSON files
    if EXTRA_KNOWLEDGE_DIR.exists():
        for fp in sorted(EXTRA_KNOWLEDGE_DIR.glob("*.json")):
            # Skip digest-generated dedup archives to avoid re-processing
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


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine-similarity matrix for row vectors.

    The input is assumed to be L2-normalised (as ``BagOfWords.transform``
    already does), so the similarity is simply ``vectors @ vectors.T``.
    """
    # Clamp to [-1, 1] to guard against floating-point drift.
    sim = vectors @ vectors.T
    return np.clip(sim, -1.0, 1.0)


def _timestamp() -> str:
    """UTC timestamp suitable for file names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, data: object) -> None:
    """Safely write JSON to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _read_json(path: Path) -> list | dict:
    """Read a JSON file, returning an empty structure on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


# =====================================================================
# 1. Deduplication
# =====================================================================

def deduplicate_entries() -> dict:
    """Find and remove near-duplicate knowledge entries across all sources.

    Uses a two-pass strategy for scalability with large datasets:

    1. **Exact-match pass** — entries with identical lowercased questions
       are grouped; only the longest answer is kept.
    2. **Near-duplicate pass** — token-overlap blocking narrows candidate
       pairs, then cosine similarity on TF-IDF vectors confirms duplicates
       above the configured threshold.

    The entry with the longer, more detailed answer is always kept.
    Duplicates are archived to
    ``data/extra_knowledge/digest_dedup_<timestamp>.json`` and removed
    from their original source files.

    Returns
    -------
    dict
        ``{"status", "duplicates_found", "duplicates_removed",
        "archive_file"}``
    """
    config = load_digest_config()
    threshold: float = config.get("dedup_threshold", 0.85)

    entries = _load_all_entries()
    if len(entries) < 2:
        return {"status": "ok", "duplicates_found": 0, "duplicates_removed": 0,
                "archive_file": None}

    n = len(entries)
    duplicate_indices: set[int] = set()
    duplicate_pairs: list[dict] = []

    # ── Pass 1: exact question-text duplicates ──────────────────────
    q_lower_map: dict[str, list[int]] = defaultdict(list)
    for idx, e in enumerate(entries):
        q_lower_map[e["question"].strip().lower()].append(idx)

    for _key, group in q_lower_map.items():
        if len(group) < 2:
            continue
        # Keep the entry with the longest answer
        best = max(group, key=lambda i: len(entries[i]["answer"]))
        for idx in group:
            if idx != best and idx not in duplicate_indices:
                duplicate_indices.add(idx)
                duplicate_pairs.append({
                    "kept": _entry_summary(entries[best]),
                    "removed": _entry_detail(entries[idx]),
                    "similarity": 1.0,
                })

    # ── Pass 2: near-duplicate detection via blocking ───────────────
    # Build an inverted index of tokens → entry indices for blocking.
    # Only compare pairs sharing at least one token (avoids O(n²)).
    entry_tokens: list[set[str]] = [
        set(tokenize(e["question"])) for e in entries
    ]

    # Build inverted index; skip tokens appearing in > 500 entries to
    # keep the candidate set manageable.
    inv_index: dict[str, list[int]] = defaultdict(list)
    for idx, tokens in enumerate(entry_tokens):
        if idx in duplicate_indices:
            continue
        for tok in tokens:
            inv_index[tok].append(idx)

    inv_index = {
        tok: ids for tok, ids in inv_index.items()
        if len(ids) <= 500
    }

    # Collect candidate pairs
    candidate_pairs: set[tuple[int, int]] = set()
    for _tok, ids in inv_index.items():
        for a_pos in range(len(ids)):
            if ids[a_pos] in duplicate_indices:
                continue
            for b_pos in range(a_pos + 1, min(a_pos + 50, len(ids))):
                if ids[b_pos] in duplicate_indices:
                    continue
                candidate_pairs.add((ids[a_pos], ids[b_pos]))
        # Cap total candidates per cycle
        if len(candidate_pairs) > 200_000:
            break

    # Vectorise only the entries that appear in candidate pairs
    involved = sorted({i for pair in candidate_pairs for i in pair} - duplicate_indices)
    if involved:
        idx_map = {orig: pos for pos, orig in enumerate(involved)}
        involved_questions = [entries[i]["question"] for i in involved]
        bow = BagOfWords()
        vectors = bow.fit_transform(involved_questions)

        for i, j in candidate_pairs:
            if i in duplicate_indices or j in duplicate_indices:
                continue
            if i not in idx_map or j not in idx_map:
                continue
            vi = vectors[idx_map[i]]
            vj = vectors[idx_map[j]]
            sim_val = float(np.dot(vi, vj))  # already L2-normalised
            if sim_val >= threshold:
                keep, drop = (i, j) if len(entries[i]["answer"]) >= len(entries[j]["answer"]) else (j, i)
                if drop not in duplicate_indices:
                    duplicate_indices.add(drop)
                    duplicate_pairs.append({
                        "kept": _entry_summary(entries[keep]),
                        "removed": _entry_detail(entries[drop]),
                        "similarity": round(sim_val, 4),
                    })

    if not duplicate_pairs:
        return {"status": "ok", "duplicates_found": 0, "duplicates_removed": 0,
                "archive_file": None}

    # Archive removed duplicates
    archive_entries = [p["removed"] for p in duplicate_pairs]
    archive_name = f"digest_dedup_{_timestamp()}.json"
    archive_path = EXTRA_KNOWLEDGE_DIR / archive_name
    _write_json(archive_path, archive_entries)

    # ── Actually remove duplicates from their source files ───────────
    drops_by_source: dict[str, set[str]] = defaultdict(set)
    for idx in duplicate_indices:
        src = entries[idx]["_source"]
        if src == "builtin":
            continue  # Cannot modify the in-code KNOWLEDGE list
        drops_by_source[src].add(entries[idx]["question"])

    files_modified = 0
    for src_path_str, questions_to_remove in drops_by_source.items():
        src_path = Path(src_path_str)
        if not src_path.exists():
            continue
        try:
            file_data: list[dict] = json.loads(src_path.read_text(encoding="utf-8"))
            original_len = len(file_data)
            file_data = [
                e for e in file_data
                if e.get("question") not in questions_to_remove
            ]
            if len(file_data) < original_len:
                if file_data:
                    _write_json(src_path, file_data)
                else:
                    src_path.unlink(missing_ok=True)
                files_modified += 1
        except (json.JSONDecodeError, OSError):
            continue

    result = {
        "status": "ok",
        "duplicates_found": len(duplicate_pairs),
        "duplicates_removed": len(duplicate_indices),
        "files_modified": files_modified,
        "archive_file": str(archive_path),
    }

    config["stats"]["last_dedup"] = _timestamp()
    config["stats"]["last_dedup_removed"] = len(duplicate_indices)
    save_digest_config(config)

    return result


def _entry_summary(entry: dict) -> dict:
    """Short summary of an entry for archive metadata."""
    answer = entry["answer"]
    return {
        "question": entry["question"],
        "answer": answer[:120] + "…" if len(answer) > 120 else answer,
        "domain": entry["domain"],
    }


def _entry_detail(entry: dict) -> dict:
    """Full entry detail for the duplicate archive."""
    return {
        "question": entry["question"],
        "answer": entry["answer"],
        "domain": entry["domain"],
        "_source": entry.get("_source", "unknown"),
    }


# =====================================================================
# 2. Quality scoring
# =====================================================================

def score_entry_quality(entries: list[dict]) -> list[dict]:
    """Score each entry on a 0–1 scale and attach a ``quality_score`` field.

    Scoring criteria (each 0–1, combined as weighted average):
      • **Answer length** — longer answers score higher, plateauing at
        ~300 characters.
      • **Question clarity** — presence of recognised question / prompt
        words.
      • **Domain specificity** — the entry's ``domain`` is not the
        generic ``"general"`` bucket and contains domain-typical terms.
      • **Answer informativeness** — ratio of unique tokens to total
        tokens; penalises repetitive or very short answers.

    Parameters
    ----------
    entries : list[dict]
        Each dict must contain ``question``, ``answer``, ``domain``.

    Returns
    -------
    list[dict]
        The same dicts with ``quality_score`` (float 0–1) added.
    """
    for entry in entries:
        q = entry.get("question", "")
        a = entry.get("answer", "")
        domain = entry.get("domain", "general")

        # --- answer length (0–1, sigmoid-like ramp, plateau ~300 chars) ---
        alen = len(a)
        length_score = min(alen / 300.0, 1.0)

        # --- question clarity ---
        q_lower = q.lower()
        q_tokens = set(re.findall(r"[a-z]+", q_lower))
        has_question_word = bool(q_tokens & _QUESTION_WORDS)
        has_question_mark = "?" in q
        clarity_score = 0.0
        if has_question_word:
            clarity_score += 0.6
        if has_question_mark:
            clarity_score += 0.4
        clarity_score = min(clarity_score, 1.0)

        # --- domain specificity ---
        domain_score = 0.2 if domain.lower() == "general" else 0.7
        # Bonus: domain name appears in question or answer
        if domain.lower() in q_lower or domain.lower() in a.lower():
            domain_score = min(domain_score + 0.3, 1.0)

        # --- answer informativeness ---
        a_tokens = tokenize(a)
        if len(a_tokens) > 0:
            unique_ratio = len(set(a_tokens)) / len(a_tokens)
            info_score = min(unique_ratio * 1.2, 1.0)  # slight boost
        else:
            info_score = 0.0

        # Weighted combination
        quality = (
            0.30 * length_score
            + 0.25 * clarity_score
            + 0.20 * domain_score
            + 0.25 * info_score
        )
        entry["quality_score"] = round(float(quality), 4)

    return entries


# =====================================================================
# 3. Cross-referencing
# =====================================================================

def cross_reference_entries() -> dict:
    """Build a simple knowledge graph by cross-referencing entries.

    Key terms are tokens that appear in 2+ entries but fewer than 50 % of
    all entries.  Entries sharing ≥ ``cross_ref_min_shared_terms`` key
    terms are considered *related* and grouped into clusters.

    The graph is saved to ``data/knowledge_graph.json``.

    Returns
    -------
    dict
        ``{"status", "total_terms", "total_connections", "clusters"}``
    """
    config = load_digest_config()
    min_shared: int = config.get("cross_ref_min_shared_terms", 3)

    entries = _load_all_entries()
    if not entries:
        return {"status": "ok", "total_terms": 0, "total_connections": 0,
                "clusters": 0}

    n = len(entries)
    half = n * 0.5

    # Tokenise all entries (question + answer) and build doc-frequency
    entry_tokens: list[set[str]] = []
    doc_freq: dict[str, int] = defaultdict(int)

    for e in entries:
        tokens = set(tokenize(e["question"] + " " + e["answer"]))
        entry_tokens.append(tokens)
        for tok in tokens:
            doc_freq[tok] += 1

    # Keep key terms: appear in 2+ entries but < 50 % of entries
    key_terms: set[str] = {
        tok for tok, freq in doc_freq.items()
        if 2 <= freq < half
    }

    # Build term → [entry indices]
    # Cap posting-list length to 200 to keep pair enumeration tractable.
    _POSTING_CAP = 200
    term_index: dict[str, list[int]] = defaultdict(list)
    for idx, tokens in enumerate(entry_tokens):
        for tok in tokens & key_terms:
            if len(term_index[tok]) < _POSTING_CAP:
                term_index[tok].append(idx)

    # Sort the lists for determinism
    term_index = {k: sorted(v) for k, v in sorted(term_index.items())}

    # Find related pairs (share ≥ min_shared key terms)
    # Build overlap counts efficiently via the inverted index.
    # We cap pair accumulation to avoid runaway memory usage.
    _MAX_PAIRS = 500_000
    pair_shared: dict[tuple[int, int], int] = defaultdict(int)
    _pair_overflow = False
    for _term, indices in term_index.items():
        for a_pos in range(len(indices)):
            for b_pos in range(a_pos + 1, len(indices)):
                pair = (indices[a_pos], indices[b_pos])
                pair_shared[pair] += 1
                if len(pair_shared) > _MAX_PAIRS:
                    _pair_overflow = True
                    break
            if _pair_overflow:
                break
        if _pair_overflow:
            break

    connections: list[dict] = []
    adj: dict[int, set[int]] = defaultdict(set)
    for (i, j), count in pair_shared.items():
        if count >= min_shared:
            connections.append({
                "entry_a": i,
                "entry_b": j,
                "shared_terms": count,
            })
            adj[i].add(j)
            adj[j].add(i)

    # Simple connected-component clustering (BFS)
    visited: set[int] = set()
    clusters: list[list[int]] = []
    for node in sorted(adj.keys()):
        if node in visited:
            continue
        cluster: list[int] = []
        queue = [node]
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            cluster.append(curr)
            for neighbour in sorted(adj[curr]):
                if neighbour not in visited:
                    queue.append(neighbour)
        clusters.append(sorted(cluster))

    # Persist the knowledge graph
    graph_data = {
        "terms": term_index,
        "clusters": clusters,
        "connections": connections[:500],  # cap for file-size sanity
        "entry_count": n,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(KNOWLEDGE_GRAPH_PATH, graph_data)

    result = {
        "status": "ok",
        "total_terms": len(term_index),
        "total_connections": len(connections),
        "clusters": len(clusters),
        "largest_cluster": max((len(c) for c in clusters), default=0),
        "graph_file": str(KNOWLEDGE_GRAPH_PATH),
    }

    config["stats"]["last_cross_ref"] = _timestamp()
    config["stats"]["cross_ref_terms"] = len(term_index)
    config["stats"]["cross_ref_clusters"] = len(clusters)
    save_digest_config(config)

    return result


# =====================================================================
# 4. Derived knowledge generation
# =====================================================================

def generate_derived_knowledge() -> dict:
    """Create new Q&A entries by combining / synthesizing existing knowledge.

    Strategies
    ──────────
    • **Comparison pairs** — for two entries in the same domain whose
      cosine similarity is between 0.3 and 0.7, generate a "How does X
      relate to Y?" question with a merged answer.
    • **Linking entries** — if entry A mentions a term that entry B
      defines (i.e. appears in B's question), create a bridging entry.
    • **Cluster summaries** — for each cluster from the knowledge graph,
      produce a summary entry.

    Generated entries are saved to
    ``data/extra_knowledge/digest_derived_<timestamp>.json``.
    A maximum of ``max_derived_per_cycle`` entries is produced per call.

    Returns
    -------
    dict
        ``{"status", "entries_generated", "output_file"}``
    """
    config = load_digest_config()
    sim_lo, sim_hi = config.get("derive_similarity_range", [0.3, 0.7])
    max_derived: int = config.get("max_derived_per_cycle", 50)

    entries = _load_all_entries()
    if len(entries) < 2:
        return {"status": "ok", "entries_generated": 0, "output_file": None}

    # For very large datasets, sample a manageable subset for vectorisation.
    _MAX_ENTRIES = 5000
    if len(entries) > _MAX_ENTRIES:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(entries), size=_MAX_ENTRIES, replace=False)
        sample_idx.sort()
        working = [entries[i] for i in sample_idx]
    else:
        working = entries

    # Vectorise
    questions = [e["question"] for e in working]
    bow = BagOfWords()
    vectors = bow.fit_transform(questions)

    derived: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()

    # ── Strategy 1: comparison entries (same domain, moderate similarity) ──
    # Use blocking (shared-token pairs) to avoid full O(n²) scan.
    working_tokens: list[set[str]] = [set(tokenize(e["question"])) for e in working]
    inv: dict[str, list[int]] = defaultdict(list)
    for idx, tokens in enumerate(working_tokens):
        for tok in tokens:
            if len(inv[tok]) < 300:
                inv[tok].append(idx)

    candidate_pairs: set[tuple[int, int]] = set()
    for _tok, ids in inv.items():
        for a_pos in range(len(ids)):
            for b_pos in range(a_pos + 1, min(a_pos + 30, len(ids))):
                pair = (ids[a_pos], ids[b_pos]) if ids[a_pos] < ids[b_pos] else (ids[b_pos], ids[a_pos])
                candidate_pairs.add(pair)
        if len(candidate_pairs) > 100_000:
            break

    for i, j in candidate_pairs:
        if len(derived) >= max_derived:
            break
        if (i, j) in seen_pairs:
            continue
        if working[i]["domain"] != working[j]["domain"]:
            continue
        sim_val = float(np.dot(vectors[i], vectors[j]))  # L2-normalised
        if not (sim_lo <= sim_val <= sim_hi):
            continue

        topic_i = _extract_topic(working[i]["question"])
        topic_j = _extract_topic(working[j]["question"])
        if topic_i == topic_j:
            continue

        question = f"How does {topic_i} relate to {topic_j}?"
        answer = (
            f"{topic_i}: {working[i]['answer']}  |  "
            f"{topic_j}: {working[j]['answer']}"
        )
        derived.append({
            "question": question,
            "answer": answer,
            "domain": working[i]["domain"],
            "source": "digest_derived:comparison",
        })
        seen_pairs.add((i, j))

    # ── Strategy 2: linking entries (term mentioned ↔ term defined) ─────
    # Build a quick lookup: first notable noun in question → entry index
    topic_lookup: dict[str, int] = {}
    for idx, e in enumerate(working):
        topic = _extract_topic(e["question"]).lower()
        if topic and topic not in topic_lookup:
            topic_lookup[topic] = idx

    for idx, e in enumerate(working):
        if len(derived) >= max_derived:
            break
        a_tokens = set(tokenize(e["answer"]))
        for term, def_idx in topic_lookup.items():
            if len(derived) >= max_derived:
                break
            if def_idx == idx:
                continue
            pair = tuple(sorted((idx, def_idx)))
            if pair in seen_pairs:
                continue
            term_tokens = set(tokenize(term))
            if term_tokens and term_tokens.issubset(a_tokens):
                topic_a = _extract_topic(working[idx]["question"])
                topic_b = _extract_topic(working[def_idx]["question"])
                question = f"What is the connection between {topic_a} and {topic_b}?"
                answer = (
                    f"{topic_a} references {topic_b}. "
                    f"{working[def_idx]['answer']}"
                )
                derived.append({
                    "question": question,
                    "answer": answer,
                    "domain": e["domain"],
                    "source": "digest_derived:link",
                })
                seen_pairs.add(pair)

    # ── Strategy 3: cluster summaries ───────────────────────────────────
    if KNOWLEDGE_GRAPH_PATH.exists() and len(derived) < max_derived:
        try:
            graph = json.loads(KNOWLEDGE_GRAPH_PATH.read_text(encoding="utf-8"))
            for cluster in graph.get("clusters", []):
                if len(derived) >= max_derived:
                    break
                if len(cluster) < 3:
                    continue
                # Pick up to 5 representative entries from the cluster
                sample_indices = cluster[:5]
                sample_entries = [entries[i] for i in sample_indices if i < len(entries)]
                if not sample_entries:
                    continue
                domain = sample_entries[0]["domain"]
                topics = [_extract_topic(e["question"]) for e in sample_entries]
                topics = [t for t in topics if t]
                if len(topics) < 2:
                    continue
                question = f"Summarize: {', '.join(topics[:4])}"
                answer = " | ".join(
                    f"{_extract_topic(e['question'])}: "
                    f"{e['answer'][:150].rstrip()}"
                    for e in sample_entries
                )
                derived.append({
                    "question": question,
                    "answer": answer,
                    "domain": domain,
                    "source": "digest_derived:cluster_summary",
                })
        except (json.JSONDecodeError, OSError):
            pass

    # ── Persist ─────────────────────────────────────────────────────────
    output_file = None
    if derived:
        fname = f"digest_derived_{_timestamp()}.json"
        output_path = EXTRA_KNOWLEDGE_DIR / fname
        _write_json(output_path, derived)
        output_file = str(output_path)

    result = {
        "status": "ok",
        "entries_generated": len(derived),
        "output_file": output_file,
    }

    config["stats"]["last_derive"] = _timestamp()
    config["stats"]["last_derived_count"] = len(derived)
    save_digest_config(config)

    return result


def _extract_topic(question: str) -> str:
    """Best-effort extraction of the main topic from a question string.

    Strips common question prefixes and returns the remaining noun phrase.
    """
    q = question.strip().rstrip("?").strip()
    # Remove leading question patterns
    q = re.sub(
        r"^(what is|what are|who is|tell me about|explain|describe|define|how does|how do)\s+",
        "",
        q,
        flags=re.IGNORECASE,
    )
    # Remove leading articles
    q = re.sub(r"^(a|an|the)\s+", "", q, flags=re.IGNORECASE)
    return q.strip()


# =====================================================================
# 5. Full digest cycle
# =====================================================================

def run_digest_cycle() -> dict:
    """Execute a full digest cycle: deduplicate → score → cross-ref → derive.

    Returns
    -------
    dict
        Aggregated results from every sub-step plus overall timing.
    """
    config = load_digest_config()
    start = time.monotonic()
    results: dict = {"steps": []}

    # Step 1 — Deduplication
    try:
        dedup = deduplicate_entries()
        results["steps"].append({"step": "deduplicate", **dedup})
    except Exception as exc:
        results["steps"].append({"step": "deduplicate", "status": "error",
                                 "message": str(exc)})

    # Step 2 — Quality scoring (informational, not destructive)
    try:
        entries = _load_all_entries()
        scored = score_entry_quality(entries)
        avg_quality = (
            float(np.mean([e["quality_score"] for e in scored]))
            if scored else 0.0
        )
        low_quality = sum(
            1 for e in scored
            if e["quality_score"] < config.get("quality_min_score", 0.3)
        )
        results["steps"].append({
            "step": "quality_score",
            "status": "ok",
            "entries_scored": len(scored),
            "avg_quality": round(avg_quality, 4),
            "low_quality_count": low_quality,
        })
    except Exception as exc:
        results["steps"].append({"step": "quality_score", "status": "error",
                                 "message": str(exc)})

    # Step 3 — Cross-referencing
    try:
        xref = cross_reference_entries()
        results["steps"].append({"step": "cross_reference", **xref})
    except Exception as exc:
        results["steps"].append({"step": "cross_reference", "status": "error",
                                 "message": str(exc)})

    # Step 4 — Derived knowledge
    try:
        derived = generate_derived_knowledge()
        results["steps"].append({"step": "derive", **derived})
    except Exception as exc:
        results["steps"].append({"step": "derive", "status": "error",
                                 "message": str(exc)})

    elapsed = round(time.monotonic() - start, 2)
    results["status"] = "ok"
    results["elapsed_seconds"] = elapsed
    results["ran_at"] = datetime.now(timezone.utc).isoformat()

    # Persist cycle metadata
    config["last_digest"] = results["ran_at"]
    config["digest_count"] = config.get("digest_count", 0) + 1
    config["stats"]["last_elapsed"] = elapsed
    save_digest_config(config)

    return results


# =====================================================================
# 6. Stats
# =====================================================================

def get_digest_stats() -> dict:
    """Return digest-engine statistics for the admin dashboard.

    Returns
    -------
    dict
        Last run info, dedup counts, quality scores, knowledge-graph stats.
    """
    config = load_digest_config()
    stats: dict = {
        "digest_count": config.get("digest_count", 0),
        "last_digest": config.get("last_digest"),
        "config": {
            k: v for k, v in config.items()
            if k not in ("stats",)
        },
    }

    # Dedup stats
    stats["dedup"] = {
        "last_run": config.get("stats", {}).get("last_dedup"),
        "last_removed": config.get("stats", {}).get("last_dedup_removed", 0),
    }

    # Quality stats — compute live if entries are available
    try:
        entries = _load_all_entries()
        scored = score_entry_quality(entries)
        scores = [e["quality_score"] for e in scored]
        stats["quality"] = {
            "total_entries": len(scored),
            "avg_score": round(float(np.mean(scores)), 4) if scores else 0.0,
            "min_score": round(float(np.min(scores)), 4) if scores else 0.0,
            "max_score": round(float(np.max(scores)), 4) if scores else 0.0,
            "low_quality_count": sum(
                1 for s in scores
                if s < config.get("quality_min_score", 0.3)
            ),
        }
    except Exception:
        stats["quality"] = {"error": "unable to compute"}

    # Knowledge-graph stats
    if KNOWLEDGE_GRAPH_PATH.exists():
        try:
            graph = json.loads(KNOWLEDGE_GRAPH_PATH.read_text(encoding="utf-8"))
            stats["knowledge_graph"] = {
                "total_terms": len(graph.get("terms", {})),
                "clusters": len(graph.get("clusters", [])),
                "connections": len(graph.get("connections", [])),
                "generated_at": graph.get("generated_at"),
            }
        except (json.JSONDecodeError, OSError):
            stats["knowledge_graph"] = {"error": "unable to read"}
    else:
        stats["knowledge_graph"] = {"status": "not_yet_generated"}

    # Derived-knowledge stats
    stats["derived"] = {
        "last_run": config.get("stats", {}).get("last_derive"),
        "last_count": config.get("stats", {}).get("last_derived_count", 0),
    }

    return stats
