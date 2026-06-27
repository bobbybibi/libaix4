#!/usr/bin/env python3
"""consolidate_knowledge.py — Collapse duplicate crawler knowledge files.

Crawlers historically appended a new file every cycle, so data/extra_knowledge/
accumulated hundreds of ~99%-duplicate ``crawl_*`` / ``forum_*`` files. This
merges them into a single de-duplicated file and removes the originals, which
dramatically speeds up index building and slims the working tree. Hand-curated
``curated_*.json`` packs are left untouched, and content is fully preserved
(verified before anything is deleted).

    python scripts/consolidate_knowledge.py            # consolidate + delete dups
    python scripts/consolidate_knowledge.py --dry-run  # report only, change nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXTRA = Path("data/extra_knowledge")
OUT = EXTRA / "consolidated_crawled.json"


def _load(fp: Path) -> list:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _keys(entries: list) -> set[tuple[str, str]]:
    return {
        (e.get("question", ""), e.get("answer", ""))
        for e in entries
        if isinstance(e, dict)
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate duplicate crawler knowledge files")
    ap.add_argument("--dry-run", action="store_true", help="Report only; change nothing")
    args = ap.parse_args()

    all_files = sorted(EXTRA.glob("*.json"))
    curated = [f for f in all_files if f.name.startswith("curated_")]
    crawler = [
        f for f in all_files
        if f.name.startswith(("crawl_", "forum_")) and f != OUT
    ]

    curated_keys: set[tuple[str, str]] = set()
    for f in curated:
        curated_keys |= _keys(_load(f))

    seen: set[tuple[str, str]] = set(curated_keys)
    consolidated: list = []
    raw = 0
    for f in crawler:
        for e in _load(f):
            if not isinstance(e, dict):
                continue
            raw += 1
            key = (e.get("question", ""), e.get("answer", ""))
            if key in seen:
                continue
            seen.add(key)
            consolidated.append(e)

    print(f"crawler files:      {len(crawler)}")
    print(f"raw crawler entries: {raw:,}")
    print(f"unique new entries:  {len(consolidated):,}")
    print(f"curated files kept:  {len(curated)} ({len(curated_keys):,} entries)")

    if args.dry_run:
        print("\n(dry run — nothing written or deleted)")
        return

    # Compute the full union from the originals up front, for verification.
    full = set(curated_keys)
    for f in crawler:
        full |= _keys(_load(f))

    OUT.write_text(json.dumps(consolidated, ensure_ascii=False), encoding="utf-8")

    after = set(curated_keys) | _keys(consolidated)
    if after != full:
        print("VERIFY FAILED — consolidated set differs from original; keeping originals.")
        sys.exit(1)

    for f in crawler:
        f.unlink()
    print(f"\nConsolidated {len(crawler)} files → {OUT.name} ({len(consolidated):,} entries). "
          f"Content verified identical; originals removed.")


if __name__ == "__main__":
    main()
