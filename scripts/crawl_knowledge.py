#!/usr/bin/env python3
"""
crawl_knowledge.py — CLI wrapper for the libaix knowledge crawler.

Usage:
    python scripts/crawl_knowledge.py              # Run all enabled topics
    python scripts/crawl_knowledge.py --topic "OSPF routing"
    python scripts/crawl_knowledge.py --topic "VPN" --keywords "IPsec,WireGuard"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawler import crawl_single_topic, run_all_crawlers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="libaix knowledge crawler")
    parser.add_argument("--topic", type=str, help="Single topic to crawl")
    parser.add_argument("--keywords", type=str, default="",
                        help="Comma-separated keywords for the topic")
    parser.add_argument("--max-articles", type=int, default=10,
                        help="Max Wikipedia articles per search query")
    args = parser.parse_args()

    if args.topic:
        kw = [k.strip() for k in args.keywords.split(",") if k.strip()]
        print(f"Crawling topic: {args.topic}")
        result = crawl_single_topic(args.topic, kw, args.max_articles)
        print(f"Status: {result['status']}")
        print(f"Entries: {result.get('entries', 0)}")
        if result.get("file"):
            print(f"Saved to: {result['file']}")
    else:
        print("Running all enabled crawlers…")
        results = run_all_crawlers()
        for name, info in results.get("topics", {}).items():
            print(f"  {name}: {info['status']} ({info.get('entries', 0)} entries)")
        print(f"\nTotal new entries: {results.get('total_new_entries', 0)}")


if __name__ == "__main__":
    main()
