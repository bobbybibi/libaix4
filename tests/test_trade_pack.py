"""Tests for trade_pack.py — the Trade Pack registry that parameterizes the
pipeline for any trade — plus the pack-aware behaviour of classify_domain."""
from __future__ import annotations

import json

import pytest

import trade_pack
from file_processor import DOMAIN_KEYWORDS, classify_domain

FLAGSHIP = {"networking", "plumbing", "fitness_coaching", "auto_mechanic"}


@pytest.fixture(autouse=True)
def _clean_pack_state(monkeypatch, tmp_path):
    """Deterministic active-trade resolution: no env override, no pointer file."""
    monkeypatch.delenv("LIBAIX_ACTIVE_TRADE", raising=False)
    monkeypatch.delenv("LIBAIX_TRADE_PACKS_DIR", raising=False)
    # Point the active-trade pointer at a path that does not exist so resolution
    # falls back to the DEFAULT_TRADE_ID regardless of the dev's working tree.
    monkeypatch.setenv("LIBAIX_ACTIVE_TRADE_PATH", str(tmp_path / "nope.json"))
    trade_pack.clear_cache()
    yield
    trade_pack.clear_cache()


# ── Pack discovery / loading ──────────────────────────────────────────────

class TestPackLoading:
    def test_flagship_packs_are_present(self):
        assert FLAGSHIP.issubset(set(trade_pack.list_trades()))

    def test_every_pack_is_valid_json_with_required_fields(self):
        for slug in trade_pack.list_trades():
            pack = trade_pack.load_trade(slug)
            assert pack is not None, slug
            assert pack["id"] == slug
            assert isinstance(pack.get("domain_keywords"), dict)
            assert isinstance(pack.get("persona"), dict)

    def test_load_missing_trade_returns_none(self):
        assert trade_pack.load_trade("does_not_exist") is None
        assert trade_pack.load_trade("") is None

    def test_networking_keywords_mirror_domain_keywords(self):
        # The networking pack must reproduce the built-in constants exactly so
        # existing networking behaviour and tests stay unchanged.
        assert trade_pack.domain_keywords_for("networking") == DOMAIN_KEYWORDS


# ── Active trade resolution ───────────────────────────────────────────────

class TestActiveTrade:
    def test_default_is_networking(self):
        assert trade_pack.get_active_trade_id() == "networking"
        assert trade_pack.active_pack()["id"] == "networking"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("LIBAIX_ACTIVE_TRADE", "plumbing")
        assert trade_pack.get_active_trade_id() == "plumbing"
        assert trade_pack.active_pack()["id"] == "plumbing"

    def test_pointer_file_selects_trade(self, monkeypatch, tmp_path):
        pointer = tmp_path / "active.json"
        pointer.write_text(json.dumps({"trade_id": "auto_mechanic"}))
        monkeypatch.setenv("LIBAIX_ACTIVE_TRADE_PATH", str(pointer))
        assert trade_pack.get_active_trade_id() == "auto_mechanic"

    def test_set_active_trade_id_roundtrip(self, monkeypatch, tmp_path):
        pointer = tmp_path / "active.json"
        monkeypatch.setenv("LIBAIX_ACTIVE_TRADE_PATH", str(pointer))
        trade_pack.set_active_trade_id("fitness_coaching")
        assert json.loads(pointer.read_text())["trade_id"] == "fitness_coaching"
        assert trade_pack.get_active_trade_id() == "fitness_coaching"


# ── Fallback behaviour ────────────────────────────────────────────────────

class TestResolveFallbacks:
    def test_unknown_trade_falls_back_to_networking(self):
        assert trade_pack.resolve_pack("totally_unknown")["id"] == "networking"

    def test_empty_packs_dir_yields_minimal_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LIBAIX_TRADE_PACKS_DIR", str(tmp_path))
        trade_pack.clear_cache()
        pack = trade_pack.resolve_pack("networking")
        assert pack["id"] == "general"
        assert pack["domain_keywords"] == {}
        assert pack["fallback"]  # always a usable string


# ── Per-trade storage locations ───────────────────────────────────────────

class TestStorageLocations:
    def test_extra_dir_is_per_trade(self):
        assert trade_pack.extra_dir_for("plumbing").as_posix().endswith(
            "data/extra_knowledge/plumbing"
        )

    def test_retrieval_dir_is_per_trade(self):
        assert trade_pack.retrieval_dir_for("plumbing").as_posix().endswith(
            "models/retrieval/plumbing"
        )

    def test_dirs_default_to_active_trade(self):
        assert trade_pack.extra_dir_for().as_posix().endswith(
            "data/extra_knowledge/networking"
        )


# ── seed_knowledge accessor ───────────────────────────────────────────────

class TestSeedKnowledge:
    def test_inline_seed_entries_returned(self):
        seed = trade_pack.seed_knowledge_for("plumbing")
        assert seed and all({"question", "answer", "domain"} <= set(e) for e in seed)

    def test_seed_from_path(self, monkeypatch, tmp_path):
        seed_file = tmp_path / "seed.json"
        seed_file.write_text(json.dumps([{"question": "q", "answer": "a", "domain": "d"}]))
        pack_dir = tmp_path / "packs"
        pack_dir.mkdir()
        (pack_dir / "demo.json").write_text(
            json.dumps({"id": "demo", "seed_knowledge": str(seed_file)})
        )
        monkeypatch.setenv("LIBAIX_TRADE_PACKS_DIR", str(pack_dir))
        trade_pack.clear_cache()
        assert trade_pack.seed_knowledge_for("demo") == [
            {"question": "q", "answer": "a", "domain": "d"}
        ]


# ── Pack-aware classify_domain ────────────────────────────────────────────

class TestPackAwareClassify:
    def test_default_classify_still_networking(self):
        # Active trade defaults to networking → identical to legacy behaviour.
        assert classify_domain("The TCP protocol uses IP addresses for routing") == "networking"

    def test_explicit_pack_keywords_classify_other_trades(self):
        plumbing_kw = trade_pack.domain_keywords_for("plumbing")
        assert classify_domain("Replace the worn flapper in the toilet tank", plumbing_kw) == "fixtures"
        assert classify_domain("Install a GFCI outlet on a 20 amp circuit", plumbing_kw) == "electrical"

    def test_active_trade_switches_classification(self, monkeypatch):
        monkeypatch.setenv("LIBAIX_ACTIVE_TRADE", "auto_mechanic")
        trade_pack.clear_cache()
        assert classify_domain("The check engine light shows a P0420 trouble code") == "diagnostics"

    def test_unknown_text_returns_general(self):
        assert classify_domain("The quick brown fox jumps over the lazy dog") == "general"


# ── Trade-aware crawl / forum defaults ────────────────────────────────────

class TestTradeAwareCrawlDefaults:
    def test_crawler_default_topics_are_networking_by_default(self):
        import crawler

        cfg = crawler._default_config()
        names = {t["name"] for t in cfg["topics"]}
        assert "Network Protocols" in names
        # Must equal the networking pack's crawl topics.
        assert cfg["topics"] == trade_pack.crawl_topics_for("networking")

    def test_crawler_default_topics_follow_active_trade(self, monkeypatch):
        import crawler

        monkeypatch.setenv("LIBAIX_ACTIVE_TRADE", "plumbing")
        trade_pack.clear_cache()
        cfg = crawler._default_config()
        assert cfg["topics"] == trade_pack.crawl_topics_for("plumbing")
        assert any("Plumbing" in t["name"] for t in cfg["topics"])

    def test_forum_default_topics_networking_unchanged(self):
        import forum_crawler

        cfg = forum_crawler._default_forum_config()
        names = [t["name"] for t in cfg["topics"]]
        assert names == ["Wi-Fi Security", "Network Troubleshooting"]
        for t in cfg["topics"]:
            assert t["sources"] == ["stackexchange", "reddit", "hackernews", "devto"]

    def test_forum_topics_derived_for_trade_without_explicit_forum_topics(self, monkeypatch):
        import forum_crawler

        monkeypatch.setenv("LIBAIX_ACTIVE_TRADE", "auto_mechanic")
        trade_pack.clear_cache()
        cfg = forum_crawler._default_forum_config()
        # auto_mechanic declares no forum_topics → derived from its crawl topics.
        crawl_names = [t["name"] for t in trade_pack.crawl_topics_for("auto_mechanic")]
        assert [t["name"] for t in cfg["topics"]] == crawl_names
        assert all(set(t) >= {"name", "keywords", "sources", "max_per_source"} for t in cfg["topics"])


# ── Trade-aware fallback message ──────────────────────────────────────────

class TestTradeFallback:
    def test_fallback_is_trade_specific(self):
        assert "networking" in trade_pack.fallback_for("networking").lower()
        assert "plumbing" in trade_pack.fallback_for("plumbing").lower()
        assert "licensed professional" in trade_pack.fallback_for("plumbing").lower()
