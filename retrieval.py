"""retrieval.py — Zero-training knowledge retrieval via TF-IDF cosine similarity.

A 15k-way softmax classifier with roughly one example per class is really a
*retrieval* problem wearing a classifier costume. This module answers a question
by finding the most similar known question — bag-of-words TF-IDF vectors are
L2-normalized, so cosine similarity is just a dot product — and returning its
answer.

Why this is the right tool here:
  • No neural-network training: building the index is just vectorizing the known
    questions, so it is effectively instant and cannot time out or OOM.
  • Adding knowledge is incremental — no retraining, no catastrophic forgetting.
  • Quality scales with the corpus instead of degrading as classes multiply.

NumPy only — no external ML frameworks, consistent with the rest of libaix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vectorizer import BagOfWords

EXTRA_KNOWLEDGE_DIR = Path("data/extra_knowledge")


def dedupe_entries(
    entries: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Drop exact duplicate (question, answer) pairs, preserving first-seen order."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for q, a, d in entries:
        key = (q, a)
        if key not in seen:
            seen.add(key)
            out.append((q, a, d))
    return out


class KnowledgeRetriever:
    """Cosine-similarity nearest-question retriever over a knowledge corpus."""

    def __init__(
        self,
        bow: BagOfWords,
        questions: list[str],
        answers: list[str],
        domains: list[str],
        matrix: np.ndarray,
    ) -> None:
        self.bow = bow
        self.questions = questions
        self.answers = answers
        self.domains = domains
        # (n_entries, vocab) L2-normalized TF-IDF, float32 to halve memory.
        self._matrix = matrix

    @property
    def size(self) -> int:
        return len(self.questions)

    # ── Construction ────────────────────────────────────────────────────
    @classmethod
    def fit(
        cls,
        entries: list[tuple[str, str, str]],
        *,
        max_n: int = 1,
        min_df: int = 1,
    ) -> "KnowledgeRetriever":
        """Build a retriever from (question, answer, domain) entries."""
        entries = dedupe_entries(entries)
        if not entries:
            raise ValueError("Cannot build a retriever from zero entries.")
        questions = [q for q, _, _ in entries]
        answers = [a for _, a, _ in entries]
        domains = [d for _, _, d in entries]
        bow = BagOfWords(max_n=max_n, min_df=min_df)
        matrix = bow.fit_transform(questions).astype(np.float32)
        return cls(bow, questions, answers, domains, matrix)

    @classmethod
    def build_from_knowledge(
        cls,
        extra_dir: str | Path = EXTRA_KNOWLEDGE_DIR,
        *,
        max_n: int = 1,
    ) -> "KnowledgeRetriever":
        """Build from the built-in KNOWLEDGE plus all crawled/uploaded extras."""
        from knowledge_base import KNOWLEDGE, load_extra_knowledge

        entries: list[tuple[str, str, str]] = list(KNOWLEDGE)
        extra_dir = Path(extra_dir)
        if extra_dir.exists():
            for fp in sorted(extra_dir.glob("*.json")):
                try:
                    entries.extend(load_extra_knowledge(fp))
                except Exception:
                    pass
        return cls.fit(entries, max_n=max_n)

    # ── Query ───────────────────────────────────────────────────────────
    def _scores(self, text: str) -> np.ndarray:
        vec = self.bow.transform([text]).astype(np.float32)  # (1, vocab), L2-norm
        return (vec @ self._matrix.T)[0]  # cosine similarity per entry

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """Return the *top_k* most similar entries, highest score first."""
        if not text or not text.strip() or self.size == 0:
            return []
        sims = self._scores(text)
        k = min(top_k, sims.size)
        # Cheap top-k: partition then sort just the k candidates.
        cand = np.argpartition(sims, -k)[-k:]
        cand = cand[np.argsort(sims[cand])[::-1]]
        return [
            {
                "answer": self.answers[i],
                "question": self.questions[i],
                "domain": self.domains[i],
                "score": float(sims[i]),
            }
            for i in cand
        ]

    def best(self, text: str) -> dict | None:
        """Return the single best match, or None."""
        results = self.query(text, top_k=1)
        return results[0] if results else None

    # ── Persistence ─────────────────────────────────────────────────────
    def save(self, directory: str | Path) -> None:
        """Persist the vectorizer + entries (the matrix is rebuilt on load)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bow.save(directory / "vectorizer.json")
        (directory / "entries.json").write_text(
            json.dumps(
                [
                    {"q": q, "a": a, "d": d}
                    for q, a, d in zip(self.questions, self.answers, self.domains)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeRetriever":
        directory = Path(directory)
        bow = BagOfWords.load(directory / "vectorizer.json")
        data = json.loads((directory / "entries.json").read_text(encoding="utf-8"))
        questions = [e["q"] for e in data]
        answers = [e["a"] for e in data]
        domains = [e["d"] for e in data]
        matrix = bow.transform(questions).astype(np.float32)
        return cls(bow, questions, answers, domains, matrix)
