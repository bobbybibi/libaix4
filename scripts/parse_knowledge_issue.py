#!/usr/bin/env python3
"""
parse_knowledge_issue.py — Parse a GitHub Issue body into knowledge entries.

Called by the learn-from-issues GitHub Actions workflow.
Reads the issue body from an environment variable and appends valid
(question, answer, domain) entries to knowledge_base.py.

Expected issue body format (one or more blocks):
    ## Question
    What is WiFi?

    ## Answer
    WiFi is a wireless networking technology using radio waves.

    ## Domain
    networking
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

KNOWLEDGE_FILE = Path("knowledge_base.py")
VALID_DOMAINS = {"networking", "internet", "intranet", "security", "general"}


def parse_issue_body(body: str) -> list[tuple[str, str, str]]:
    """Extract Q&A entries from a structured issue body."""
    entries: list[tuple[str, str, str]] = []

    # Split into blocks by "## Question" headers
    blocks = re.split(r"(?=##\s*[Qq]uestion)", body)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract question
        q_match = re.search(
            r"##\s*[Qq]uestion\s*\n(.+?)(?=##\s*[Aa]nswer|$)", block, re.DOTALL
        )
        if not q_match:
            continue
        question = q_match.group(1).strip()

        # Extract answer
        a_match = re.search(
            r"##\s*[Aa]nswer\s*\n(.+?)(?=##\s*[Dd]omain|$)", block, re.DOTALL
        )
        if not a_match:
            continue
        answer = a_match.group(1).strip()

        # Extract domain
        d_match = re.search(r"##\s*[Dd]omain\s*\n(.+?)(?=##|$)", block, re.DOTALL)
        domain = d_match.group(1).strip().lower() if d_match else "general"

        # Validate
        if not question or not answer:
            continue
        if len(question) < 5 or len(answer) < 10:
            print(f"  Skipping (too short): {question!r}")
            continue
        if domain not in VALID_DOMAINS:
            print(f"  Unknown domain {domain!r}, defaulting to 'general'")
            domain = "general"

        # Sanitize — remove potential code injection
        for field in (question, answer, domain):
            if any(c in field for c in ("\\", "\x00", "exec(", "import ", "__")):
                print(f"  Skipping suspicious entry: {question!r}")
                break
        else:
            entries.append((question, answer, domain))

    return entries


def append_to_knowledge_base(entries: list[tuple[str, str, str]]) -> int:
    """Append entries to knowledge_base.py before the closing bracket."""
    content = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    # Find the last entry before the closing ]
    # We insert new entries before the final `]` of the KNOWLEDGE list
    closing_idx = content.rfind("\n]")
    if closing_idx == -1:
        print("ERROR: Could not find KNOWLEDGE list closing bracket")
        return 0

    new_lines = "\n"
    for question, answer, domain in entries:
        # Escape quotes in strings
        q = question.replace('"', '\\"')
        a = answer.replace('"', '\\"')
        d = domain.replace('"', '\\"')
        new_lines += f'\n    ("{q}",\n     "{a}",\n     "{d}"),\n'

    updated = content[:closing_idx] + new_lines + content[closing_idx:]
    KNOWLEDGE_FILE.write_text(updated, encoding="utf-8")
    return len(entries)


def main() -> None:
    body = os.environ.get("ISSUE_BODY", "")
    if not body:
        print("ERROR: ISSUE_BODY environment variable is empty")
        sys.exit(1)

    print(f"Parsing issue body ({len(body)} chars)...")
    entries = parse_issue_body(body)

    if not entries:
        print("No valid knowledge entries found in the issue.")
        sys.exit(1)

    print(f"Found {len(entries)} valid entries:")
    for q, a, d in entries:
        print(f"  [{d}] {q}")

    added = append_to_knowledge_base(entries)
    print(f"\nAppended {added} entries to {KNOWLEDGE_FILE}")


if __name__ == "__main__":
    main()
