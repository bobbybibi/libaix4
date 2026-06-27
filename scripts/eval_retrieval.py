#!/usr/bin/env python3
"""eval_retrieval.py — Evaluate libaix retrieval engine answer quality.

Runs a built-in set of ~30 defensive networking/security questions against the
KnowledgeRetriever, checks domain match and score, and reports pass/fail stats.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --threshold 0.25 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/eval_retrieval.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import KnowledgeRetriever  # noqa: E402

# ---------------------------------------------------------------------------
# Built-in evaluation set: (question, expected_domain)
# Questions are phrased like a real user — not copied verbatim from the KB.
# Domains must be one of the allowed set below.
# ---------------------------------------------------------------------------
ALLOWED_DOMAINS = {
    "networking",
    "security",
    "wifi",
    "wifi_security",
    "wifi_policy",
    "internet",
    "intranet",
    "troubleshooting",
}

EVAL_SET: list[tuple[str, str]] = [
    # networking
    ("how do I stop BGP route hijacking", "networking"),
    ("what protocol prevents Layer 2 loops on switches", "networking"),
    ("explain how OSPF finds the shortest path", "networking"),
    ("what is the purpose of a subnet mask", "networking"),
    ("how does NAT let multiple devices share one public IP", "networking"),
    ("what does a network switch do at Layer 2", "networking"),
    ("how does VLAN segmentation improve security", "networking"),
    ("what is the difference between TCP and UDP transport protocols", "networking"),
    # security
    ("what detects lateral movement inside my network", "security"),
    ("how does a man-in-the-middle attack work", "security"),
    ("how can I protect against SQL injection", "security"),
    ("what is zero trust and why should I care", "security"),
    ("what does an IDS do versus an IPS", "security"),
    ("how does ransomware spread across a corporate network", "security"),
    ("what is microsegmentation and how does it limit breach damage", "security"),
    # wifi
    ("why does my wireless connection keep disconnecting on 5 GHz", "wifi"),
    ("how does beamforming improve wireless signal quality", "wifi"),
    ("what is the difference between 2.4 GHz and 5 GHz Wi-Fi bands", "wifi"),
    ("how many clients can a single enterprise access point handle", "wifi"),
    ("what does OFDMA do in Wi-Fi 6", "wifi"),
    ("what is band steering and why use it", "wifi"),
    # wifi_security
    ("how does WPA3 improve on WPA2", "wifi_security"),
    ("why should companies use WPA2-Enterprise instead of a pre-shared key", "wifi_security"),
    ("what is an evil twin attack on wireless networks", "wifi_security"),
    ("how do certificates make EAP-TLS more secure than PEAP", "wifi_security"),
    ("what is a rogue access point and how do I detect one", "wifi_security"),
    # wifi_policy
    ("how does bandwidth throttling work for guest Wi-Fi users", "wifi_policy"),
    ("what does AP isolation do on a guest SSID", "wifi_policy"),
    ("how does dynamic VLAN assignment work after 802.1X login", "wifi_policy"),
    ("what is a Wi-Fi acceptable use policy", "wifi_policy"),
    # internet
    ("what makes HTTPS more secure than HTTP", "internet"),
    ("how does TLS protect data in transit", "internet"),
    ("what is the purpose of an API on the web", "internet"),
    # intranet
    ("what is Active Directory used for in a corporate network", "intranet"),
    ("how does a VPN allow remote workers to reach intranet resources", "intranet"),
    # troubleshooting
    ("how do I troubleshoot slow Wi-Fi speeds", "troubleshooting"),
    ("how do I use traceroute to diagnose network packet loss", "troubleshooting"),
    ("how do I troubleshoot 802.1X authentication failures", "troubleshooting"),
]


def _validate_eval_set() -> None:
    """Assert every expected domain is in ALLOWED_DOMAINS."""
    for question, domain in EVAL_SET:
        if domain not in ALLOWED_DOMAINS:
            raise ValueError(
                f"Domain {domain!r} is not in ALLOWED_DOMAINS for question: {question!r}"
            )


def _truncate(text: str, max_len: int = 120) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the libaix retrieval engine."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Minimum cosine-similarity score to count as a PASS (default: 0.30)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print the top-1 answer text (truncated)",
    )
    args = parser.parse_args()

    _validate_eval_set()

    print("Building retrieval index …")
    retriever = KnowledgeRetriever.build_from_knowledge()
    print(f"Index ready: {retriever.size:,} entries\n")

    passed = 0
    total = len(EVAL_SET)
    score_sum = 0.0

    for question, expected_domain in EVAL_SET:
        result = retriever.best(question)

        if result is None:
            got_domain = ""
            score = 0.0
            answer = ""
        else:
            got_domain = result["domain"]
            score = result["score"]
            answer = result["answer"]

        ok = (
            got_domain == expected_domain
            and score >= args.threshold
            and bool(answer)
        )
        verdict = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        score_sum += score

        print(
            f"{verdict}  score={score:.2f}  got={got_domain or '(none)':15s}"
            f"  exp={expected_domain:15s}  {question}"
        )
        if args.verbose and answer:
            print(f"       answer: {_truncate(answer)}")

    mean_score = score_sum / total if total > 0 else 0.0
    pass_rate = passed / total if total > 0 else 0.0
    fail_count = total - passed

    print()
    print(
        f"{passed}/{total} passed ({pass_rate:.0%}),"
        f"  mean top-1 score={mean_score:.2f}"
        f"  ({fail_count} FAIL{'s' if fail_count != 1 else ''})"
    )

    sys.exit(0 if pass_rate >= 0.80 else 1)


if __name__ == "__main__":
    main()
