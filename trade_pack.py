"""
trade_pack.py — Trade Pack registry: parameterize the pipeline for any trade.

A **Trade Pack** is a JSON file under ``data/trade_packs/<slug>.json`` that
re-skins the existing retrieval/learning pipeline for a specific trade
(networking, plumbing, fitness coaching, auto mechanic, …) without any new ML.
It describes the trade's persona, domains, domain keywords, crawl topics, seed
knowledge and disclaimers.

This module deliberately imports **no Flask** so crawlers, scripts and GitHub
Actions can use it directly. It also avoids importing ``file_processor`` /
``knowledge_base`` to stay free of circular imports — callers that want a hard
fallback (e.g. the original ``DOMAIN_KEYWORDS``) own that fallback themselves.

Layout
------
    data/trade_packs/<slug>.json     # pack definitions (tracked in git)
    data/active_trade.json           # {"trade_id": "<slug>"} (runtime state)
    data/extra_knowledge/<slug>/     # per-trade crawled/uploaded knowledge
    models/retrieval/<slug>/         # per-trade retrieval index

Environment overrides (handy for tests / multi-tenant hosting):
    LIBAIX_TRADE_PACKS_DIR   — directory holding pack JSON files
    LIBAIX_ACTIVE_TRADE_PATH — path to the active-trade pointer file
    LIBAIX_ACTIVE_TRADE      — force the active trade id (wins over the file)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_TRADE_ID = "networking"

EXTRA_KNOWLEDGE_ROOT = Path("data/extra_knowledge")
RETRIEVAL_ROOT = Path("models/retrieval")


def _trade_packs_dir() -> Path:
    return Path(os.environ.get("LIBAIX_TRADE_PACKS_DIR", "data/trade_packs"))


def _active_trade_path() -> Path:
    return Path(os.environ.get("LIBAIX_ACTIVE_TRADE_PATH", "data/active_trade.json"))


# ── Pack loading (mtime-cached) ───────────────────────────────────────────

# Cache successful loads keyed by id → (mtime, pack). The mtime check means a
# rewritten pack file (common in tests) is transparently reloaded.
_cache: dict[str, tuple[float, dict]] = {}


def clear_cache() -> None:
    """Drop the in-memory pack cache (used by tests)."""
    _cache.clear()


def _pack_path(trade_id: str) -> Path:
    return _trade_packs_dir() / f"{trade_id}.json"


def list_trades() -> list[str]:
    """Return the sorted slugs of every available trade pack."""
    directory = _trade_packs_dir()
    if not directory.exists():
        return []
    return sorted(fp.stem for fp in directory.glob("*.json"))


def load_trade(trade_id: str) -> dict | None:
    """Load and return the pack dict for *trade_id*, or ``None`` if missing."""
    if not trade_id:
        return None
    path = _pack_path(trade_id)
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _cache.get(trade_id)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("id", trade_id)
    _cache[trade_id] = (mtime, data)
    return data


# ── Active trade ──────────────────────────────────────────────────────────

def get_active_trade_id() -> str:
    """Return the currently active trade id.

    Resolution order: ``LIBAIX_ACTIVE_TRADE`` env var → ``active_trade.json`` →
    :data:`DEFAULT_TRADE_ID`.
    """
    env = os.environ.get("LIBAIX_ACTIVE_TRADE")
    if env:
        return env
    path = _active_trade_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tid = data.get("trade_id")
            if isinstance(tid, str) and tid:
                return tid
        except (json.JSONDecodeError, OSError, AttributeError):
            pass
    return DEFAULT_TRADE_ID


def set_active_trade_id(trade_id: str) -> None:
    """Persist *trade_id* as the active trade (writes ``active_trade.json``)."""
    path = _active_trade_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"trade_id": trade_id}, indent=2),
        encoding="utf-8",
    )


def _minimal_default() -> dict:
    """A safe, content-free pack used when no pack file can be found."""
    return {
        "id": "general",
        "name": "General Assistant",
        "persona": {
            "role": "helpful expert",
            "tone": "encouraging",
            "greeting": "",
            "signoff": "",
        },
        "domains": [],
        "domain_keywords": {},
        "crawl_topics": [],
        "forum_topics": [],
        "seed_knowledge": [],
        "fallback": (
            "I'm not sure about that yet. Try rephrasing, or say "
            "'research <topic>' to teach me something new!"
        ),
        "disclaimers": [],
        "eval_set": [],
    }


def active_pack() -> dict:
    """Return the active pack dict, falling back to networking then minimal."""
    return resolve_pack(get_active_trade_id())


def resolve_pack(trade_id: str | None) -> dict:
    """Return the pack for *trade_id* with sane fallbacks.

    Falls back to the default trade, then to a minimal content-free pack, so
    callers always receive a usable dict.
    """
    tid = trade_id or get_active_trade_id()
    pack = load_trade(tid)
    if pack is None and tid != DEFAULT_TRADE_ID:
        pack = load_trade(DEFAULT_TRADE_ID)
    return pack if pack is not None else _minimal_default()


# ── Field accessors (trade_id=None → active trade) ────────────────────────

def domain_keywords_for(trade_id: str | None = None) -> dict[str, list[str]]:
    """Return the ``{domain: [keywords]}`` map for the given/active trade."""
    return dict(resolve_pack(trade_id).get("domain_keywords") or {})


def persona_for(trade_id: str | None = None) -> dict:
    return dict(resolve_pack(trade_id).get("persona") or {})


def domains_for(trade_id: str | None = None) -> list[str]:
    return list(resolve_pack(trade_id).get("domains") or [])


def crawl_topics_for(trade_id: str | None = None) -> list[dict]:
    return list(resolve_pack(trade_id).get("crawl_topics") or [])


def forum_topics_for(trade_id: str | None = None) -> list[dict]:
    return list(resolve_pack(trade_id).get("forum_topics") or [])


def fallback_for(trade_id: str | None = None) -> str:
    return resolve_pack(trade_id).get("fallback") or _minimal_default()["fallback"]


def disclaimers_for(trade_id: str | None = None) -> list[str]:
    return list(resolve_pack(trade_id).get("disclaimers") or [])


def eval_set_for(trade_id: str | None = None) -> list:
    return list(resolve_pack(trade_id).get("eval_set") or [])


def seed_knowledge_for(trade_id: str | None = None) -> list:
    """Return inline seed entries, or load them from a referenced JSON file.

    ``seed_knowledge`` may be either an inline list of
    ``{"question","answer","domain"}`` dicts or a path string pointing to such
    a JSON file.
    """
    seed = resolve_pack(trade_id).get("seed_knowledge")
    if isinstance(seed, str):
        path = Path(seed)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []
    return list(seed or [])


# ── Per-trade storage locations ───────────────────────────────────────────

def extra_dir_for(trade_id: str | None = None) -> Path:
    """Per-trade crawled/uploaded knowledge directory."""
    tid = trade_id or get_active_trade_id()
    return EXTRA_KNOWLEDGE_ROOT / tid


def retrieval_dir_for(trade_id: str | None = None) -> Path:
    """Per-trade retrieval index directory."""
    tid = trade_id or get_active_trade_id()
    return RETRIEVAL_ROOT / tid
