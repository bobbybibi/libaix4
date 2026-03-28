"""
vectorizer.py — Bag-of-words text vectorizer built with only Python builtins + NumPy.

Converts text into numerical feature vectors for the neural network.
No external NLP libraries — just tokenization, stopword removal, and BoW encoding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# Common English stop words (minimal set to keep things lightweight)
STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could am it its i me my we our you your he him "
    "his she her they them their this that these those of in to for on with at "
    "by from as into through during before after above below between out off "
    "over under again further then once here there when where why how all each "
    "every both few more most other some such no nor not only own same so than "
    "too very just don t isn aren wasn weren doesn didn won wouldn shan shouldn "
    "haven hasn hadn".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens, remove stop words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class BagOfWords:
    """A simple bag-of-words vectorizer.

    Builds a vocabulary from training texts and converts new texts into
    fixed-length numerical vectors.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def fit(self, texts: list[str]) -> "BagOfWords":
        """Build vocabulary from a list of texts."""
        word_freq: dict[str, int] = {}
        doc_freq: dict[str, int] = {}
        for text in texts:
            tokens = tokenize(text)
            seen = set()
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
                if token not in seen:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
                    seen.add(token)

        # Keep words that appear in at least 1 doc, sort by frequency for stability
        sorted_words = sorted(word_freq.keys(), key=lambda w: (-word_freq[w], w))
        self.vocab = {w: i for i, w in enumerate(sorted_words)}

        # Compute IDF: log(N / df) + 1 (smoothed)
        n = len(texts)
        self.idf = np.ones(len(self.vocab), dtype=np.float64)
        for word, idx in self.vocab.items():
            df = doc_freq.get(word, 1)
            self.idf[idx] = np.log(n / df) + 1.0

        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Convert texts to TF-IDF weighted bag-of-words vectors."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            for token in tokens:
                if token in self.vocab:
                    matrix[i, self.vocab[token]] += 1.0

        # Apply TF-IDF weighting
        if self.idf is not None:
            matrix *= self.idf

        # L2 normalize each row
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix /= norms

        return matrix

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)

    def save(self, path: str | Path) -> None:
        """Save vectorizer state to JSON."""
        data = {
            "vocab": self.vocab,
            "idf": self.idf.tolist() if self.idf is not None else None,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BagOfWords":
        """Load vectorizer from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        bow = cls()
        bow.vocab = data["vocab"]
        bow.idf = np.array(data["idf"], dtype=np.float64) if data["idf"] else None
        bow._fitted = True
        return bow
