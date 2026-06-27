#!/usr/bin/env python3
"""build_retrieval_index.py — Build the zero-training retrieval index.

Rebuilds ``models/retrieval/`` from the built-in KNOWLEDGE plus all crawled and
uploaded entries. This is cheap (seconds) — run it after crawls or on a schedule
instead of expensive neural retraining.

    python scripts/build_retrieval_index.py
    python scripts/build_retrieval_index.py --out models/retrieval --max-n 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/build_retrieval_index.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import KnowledgeRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the libaix retrieval index")
    parser.add_argument("--out", default="models/retrieval", help="Output directory")
    parser.add_argument(
        "--max-n", type=int, default=1, help="Max n-gram size for the vectorizer"
    )
    args = parser.parse_args()

    t0 = time.time()
    retriever = KnowledgeRetriever.build_from_knowledge(max_n=args.max_n)
    retriever.save(args.out)
    print(
        f"Built retrieval index: {retriever.size:,} entries → {args.out} "
        f"in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
