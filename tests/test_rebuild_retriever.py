"""Test live retrieval-index rebuild wiring in app.rebuild_retriever."""

from __future__ import annotations

import app
from retrieval import KnowledgeRetriever


def test_rebuild_retriever_swaps_in_new_index(monkeypatch, tmp_path):
    tiny = KnowledgeRetriever.fit(
        [("What is TCP?", "TCP is a reliable transport protocol.", "networking")]
    )
    # Avoid the multi-second full-corpus build and keep saves out of the repo.
    monkeypatch.setattr(
        KnowledgeRetriever,
        "build_from_knowledge",
        classmethod(lambda cls, *a, **k: tiny),
    )
    monkeypatch.setattr(app, "MODEL_DIR", tmp_path)

    assert app.rebuild_retriever() is True
    assert app.knowledge_retriever is tiny
    # New knowledge is immediately answerable.
    assert app.knowledge_retriever.best("tcp")["answer"].startswith("TCP is")
    # The index was persisted under the (patched) model dir.
    assert (tmp_path / "retrieval" / "entries.json").exists()


def test_rebuild_retriever_survives_build_failure(monkeypatch):
    def boom(cls, *a, **k):
        raise RuntimeError("build failed")

    monkeypatch.setattr(
        KnowledgeRetriever, "build_from_knowledge", classmethod(boom)
    )
    # Failure is swallowed and reported as False (existing retriever untouched).
    assert app.rebuild_retriever() is False
