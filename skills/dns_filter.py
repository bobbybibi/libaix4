"""
dns_filter.py — DNS-level domain filtering skill for libaix.

Provides commands to manage a local DNS blocklist:
  • Block a domain (adds to hosts-style blocklist)
  • Unblock a domain (removes from blocklist)
  • List all currently blocked domains
  • Show DNS filter status and statistics

The blocklist is persisted in ``data/dns_blocklist.json`` and ships
with a default set of ~20 well-known ad/tracking/malware domains.
Domains are organised by category (ads, trackers, malware).

Uses only the Python standard library.  No external packages required.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from skill_registry import Skill, SkillCommand, SkillResult

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_BLOCKLIST_PATH = Path("data/dns_blocklist.json")

# Valid domain pattern (basic sanity check)
_DOMAIN_RE = re.compile(r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$")

# Default blocked domains, grouped by category
_DEFAULT_BLOCKED: dict[str, list[str]] = {
    "ads": [
        "doubleclick.net",
        "googlesyndication.com",
        "googleadservices.com",
        "adservice.google.com",
        "pagead2.googlesyndication.com",
        "ads.facebook.com",
        "ad.doubleclick.net",
    ],
    "trackers": [
        "google-analytics.com",
        "googletagmanager.com",
        "facebook.com/tr",
        "connect.facebook.net",
        "analytics.twitter.com",
        "bat.bing.com",
        "pixel.quantserve.com",
        "scorecardresearch.com",
    ],
    "malware": [
        "malware-check.disconnect.me",
        "iloveyou.virus.test",
        "tracking.example.com",
        "badware.example.org",
        "phishing.example.net",
    ],
}


# ── Helpers ──────────────────────────────────────────────────────────

def _is_valid_domain(domain: str) -> bool:
    """Return ``True`` if *domain* looks like a valid domain name."""
    return _DOMAIN_RE.match(domain) is not None


def _load_blocklist() -> dict[str, Any]:
    """Load the blocklist from disk, returning defaults on failure."""
    try:
        with open(_BLOCKLIST_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Ensure essential keys exist
        if "domains" not in data or not isinstance(data["domains"], dict):
            data["domains"] = dict(_DEFAULT_BLOCKED)
        if "enabled" not in data:
            data["enabled"] = True
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {
            "enabled": True,
            "domains": {cat: list(doms) for cat, doms in _DEFAULT_BLOCKED.items()},
            "custom": [],
            "updated_at": time.time(),
        }


def _save_blocklist(blocklist: dict[str, Any]) -> None:
    """Persist the blocklist to disk."""
    blocklist["updated_at"] = time.time()
    _BLOCKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_BLOCKLIST_PATH, "w", encoding="utf-8") as fh:
        json.dump(blocklist, fh, indent=2)


def _all_domains(blocklist: dict[str, Any]) -> set[str]:
    """Return a flat set of every blocked domain."""
    domains: set[str] = set()
    for category_domains in blocklist.get("domains", {}).values():
        domains.update(category_domains)
    domains.update(blocklist.get("custom", []))
    return domains


# ── Skill implementation ─────────────────────────────────────────────

class DNSFilterSkill(Skill):
    """Block and unblock domains via a local DNS-level blocklist."""

    def __init__(self) -> None:
        super().__init__(
            name="dns_filter",
            description="Block and unblock domains via a local DNS-level blocklist",
            version="1.0.0",
            category="network",
        )
        self._blocklist = _load_blocklist()

    # ── public interface ─────────────────────────────────────────────

    def get_commands(self) -> list[SkillCommand]:
        return [
            SkillCommand(
                name="block_domain",
                description="Block a domain",
                patterns=[
                    r"block\s+(domain|site|website)\s+(?P<domain>\S+)",
                    r"(block|blacklist)\s+(?P<domain>[\w.-]+\.[a-z]{2,})",
                ],
                args_schema={
                    "domain": {"type": "string", "required": True},
                    "category": {"type": "string", "optional": True},
                },
                category="network",
                requires_confirmation=True,
            ),
            SkillCommand(
                name="unblock_domain",
                description="Unblock a domain",
                patterns=[
                    r"unblock\s+(domain|site|website)\s+(?P<domain>\S+)",
                ],
                args_schema={
                    "domain": {"type": "string", "required": True},
                },
                category="network",
                requires_confirmation=True,
            ),
            SkillCommand(
                name="list_blocked",
                description="List all blocked domains",
                patterns=[
                    r"(show|list)\s+blocked\s+(domains|sites|websites)",
                ],
                args_schema={},
                category="network",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="dns_status",
                description="Show DNS filter status and statistics",
                patterns=[
                    r"dns\s+(filter\s+)?status",
                    r"(is|check)\s+dns\s+filter",
                ],
                args_schema={},
                category="network",
                requires_confirmation=False,
            ),
        ]

    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        """Dispatch *command* to the appropriate private handler."""
        dispatch: dict[str, Any] = {
            "block_domain": self._block_domain,
            "unblock_domain": self._unblock_domain,
            "list_blocked": self._list_blocked,
            "dns_status": self._dns_status,
        }
        handler = dispatch.get(command)
        if handler is None:
            return SkillResult(
                success=False,
                message=f"Error: unknown command '{command}'",
                data={},
            )
        try:
            return handler(args)
        except Exception as exc:
            log.exception("DNSFilterSkill.%s failed: %s", command, exc)
            return SkillResult(
                success=False,
                message=f"Error: {exc}",
                data={},
            )

    # ── private handlers ─────────────────────────────────────────────

    def _block_domain(self, args: dict[str, Any]) -> SkillResult:
        """Add a domain to the blocklist."""
        domain: str = args.get("domain", "").strip().lower()
        category: str = args.get("category", "custom").strip().lower()

        if not domain:
            return SkillResult(
                success=False,
                message="Error: no domain provided",
                data={},
            )

        if not _is_valid_domain(domain):
            return SkillResult(
                success=False,
                message=f"Error: '{domain}' is not a valid domain name",
                data={},
            )

        # Check if already blocked
        if domain in _all_domains(self._blocklist):
            return SkillResult(
                success=True,
                message=f"Domain '{domain}' is already blocked",
                data={"domain": domain, "already_blocked": True},
            )

        # Add to the appropriate category
        if category == "custom":
            custom: list[str] = self._blocklist.setdefault("custom", [])
            custom.append(domain)
        else:
            categories: dict[str, list[str]] = self._blocklist.setdefault("domains", {})
            cat_list = categories.setdefault(category, [])
            cat_list.append(domain)

        _save_blocklist(self._blocklist)

        total = len(_all_domains(self._blocklist))
        return SkillResult(
            success=True,
            message=f"Blocked domain '{domain}' (category: {category})",
            data={
                "domain": domain,
                "category": category,
                "total_blocked": total,
            },
        )

    def _unblock_domain(self, args: dict[str, Any]) -> SkillResult:
        """Remove a domain from the blocklist."""
        domain: str = args.get("domain", "").strip().lower()

        if not domain:
            return SkillResult(
                success=False,
                message="Error: no domain provided",
                data={},
            )

        removed = False

        # Search in categorised domains
        for cat, dom_list in self._blocklist.get("domains", {}).items():
            if domain in dom_list:
                dom_list.remove(domain)
                removed = True
                break

        # Search in custom list
        custom: list[str] = self._blocklist.get("custom", [])
        if domain in custom:
            custom.remove(domain)
            removed = True

        if not removed:
            return SkillResult(
                success=False,
                message=f"Domain '{domain}' is not in the blocklist",
                data={"domain": domain},
            )

        _save_blocklist(self._blocklist)

        total = len(_all_domains(self._blocklist))
        return SkillResult(
            success=True,
            message=f"Unblocked domain '{domain}'",
            data={
                "domain": domain,
                "total_blocked": total,
            },
        )

    def _list_blocked(self, args: dict[str, Any]) -> SkillResult:
        """Return all blocked domains grouped by category."""
        categories: dict[str, list[str]] = self._blocklist.get("domains", {})
        custom: list[str] = self._blocklist.get("custom", [])

        all_doms = _all_domains(self._blocklist)

        # Build per-category summary
        summary: dict[str, int] = {}
        for cat, doms in categories.items():
            summary[cat] = len(doms)
        if custom:
            summary["custom"] = len(custom)

        return SkillResult(
            success=True,
            message=f"{len(all_doms)} domain(s) blocked across {len(summary)} category/categories",
            data={
                "domains": {cat: sorted(doms) for cat, doms in categories.items()},
                "custom": sorted(custom),
                "total_blocked": len(all_doms),
                "categories": summary,
            },
        )

    def _dns_status(self, args: dict[str, Any]) -> SkillResult:
        """Show DNS filter status and statistics."""
        enabled: bool = self._blocklist.get("enabled", True)
        all_doms = _all_domains(self._blocklist)

        categories: dict[str, list[str]] = self._blocklist.get("domains", {})
        category_counts: dict[str, int] = {
            cat: len(doms) for cat, doms in categories.items()
        }
        custom_count = len(self._blocklist.get("custom", []))
        if custom_count:
            category_counts["custom"] = custom_count

        updated_at = self._blocklist.get("updated_at")

        status = "active" if enabled else "inactive"
        return SkillResult(
            success=True,
            message=f"DNS filter is {status} — {len(all_doms)} domain(s) blocked",
            data={
                "enabled": enabled,
                "status": status,
                "total_blocked": len(all_doms),
                "categories": category_counts,
                "blocklist_path": str(_BLOCKLIST_PATH),
                "updated_at": updated_at,
            },
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Persist blocklist on shutdown."""
        try:
            _save_blocklist(self._blocklist)
        except OSError as exc:
            log.warning("Failed to save blocklist on cleanup: %s", exc)
