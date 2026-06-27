"""
firewall_manager.py — Firewall management skill for libaix.

Provides commands to manage host-based firewall rules:
  • Block an IP address (iptables on Linux, netsh on Windows)
  • Unblock a previously blocked IP address
  • List current firewall rules
  • Check whether the firewall is active

IP addresses are validated via the ``ipaddress`` module.  OS detection
uses ``platform.system()``.  All external commands run through
``subprocess.run`` with a 10-second timeout.

Uses only the Python standard library.  No external packages required.
"""

from __future__ import annotations

import ipaddress
import logging
import platform
import shutil
import subprocess
from typing import Any

from skill_registry import Skill, SkillCommand, SkillResult

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_SUBPROCESS_TIMEOUT = 10


# ── Helpers ──────────────────────────────────────────────────────────

def _validate_ip(ip: str) -> str | None:
    """Return the normalised IP string, or ``None`` if invalid."""
    try:
        return str(ipaddress.ip_address(ip))
    except ValueError:
        return None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run *cmd* with timeout and capture output."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )


# ── Skill implementation ─────────────────────────────────────────────

class FirewallManagerSkill(Skill):
    """Manage host-based firewall rules (block/unblock IPs, list rules)."""

    def __init__(self) -> None:
        super().__init__(
            name="firewall_manager",
            description="Manage host-based firewall rules (block/unblock IPs, list rules)",
            version="1.0.0",
            category="security",
        )
        self._system = platform.system()

    # ── public interface ─────────────────────────────────────────────

    def get_commands(self) -> list[SkillCommand]:
        return [
            SkillCommand(
                name="block_ip",
                description="Block an IP address",
                patterns=[
                    r"block\s+(ip\s+)?(?P<ip>\d+\.\d+\.\d+\.\d+)",
                    r"firewall\s+block\s+(?P<ip>\S+)",
                ],
                args_schema={
                    "ip": {"type": "string", "required": True},
                },
                category="security",
                requires_confirmation=True,
            ),
            SkillCommand(
                name="unblock_ip",
                description="Unblock an IP address",
                patterns=[
                    r"unblock\s+(ip\s+)?(?P<ip>\d+\.\d+\.\d+\.\d+)",
                ],
                args_schema={
                    "ip": {"type": "string", "required": True},
                },
                category="security",
                requires_confirmation=True,
            ),
            SkillCommand(
                name="list_rules",
                description="Show current firewall rules",
                patterns=[
                    r"(show|list)\s+firewall\s+rules",
                    r"firewall\s+status",
                ],
                args_schema={},
                category="security",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="firewall_status",
                description="Check if the firewall is active",
                patterns=[
                    r"(is|check)\s+(the\s+)?firewall\s+(on|active|enabled|running)",
                ],
                args_schema={},
                category="security",
                requires_confirmation=False,
            ),
        ]

    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        """Dispatch *command* to the appropriate private handler."""
        dispatch: dict[str, Any] = {
            "block_ip": self._block_ip,
            "unblock_ip": self._unblock_ip,
            "list_rules": self._list_rules,
            "firewall_status": self._firewall_status,
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
            log.exception("FirewallManagerSkill.%s failed: %s", command, exc)
            return SkillResult(
                success=False,
                message=f"Error: {exc}",
                data={},
            )

    # ── private handlers ─────────────────────────────────────────────

    def _block_ip(self, args: dict[str, Any]) -> SkillResult:
        """Block an IP address using the platform firewall."""
        raw_ip: str = args.get("ip", "")
        ip = _validate_ip(raw_ip)
        if ip is None:
            return SkillResult(
                success=False,
                message=f"Error: '{raw_ip}' is not a valid IP address",
                data={},
            )

        if self._system == "Windows":
            cmd = [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name=libaix_block_{ip}",
                "dir=in", "action=block",
                f"remoteip={ip}",
                "protocol=any",
            ]
        else:
            iptables = shutil.which("iptables")
            if iptables is None:
                return SkillResult(
                    success=False,
                    message="Error: 'iptables' not found on this system",
                    data={},
                )
            cmd = [iptables, "-A", "INPUT", "-s", ip, "-j", "DROP"]

        return self._exec_firewall_cmd(cmd, f"Blocked IP {ip}", ip=ip)

    def _unblock_ip(self, args: dict[str, Any]) -> SkillResult:
        """Unblock a previously blocked IP address."""
        raw_ip: str = args.get("ip", "")
        ip = _validate_ip(raw_ip)
        if ip is None:
            return SkillResult(
                success=False,
                message=f"Error: '{raw_ip}' is not a valid IP address",
                data={},
            )

        if self._system == "Windows":
            cmd = [
                "netsh", "advfirewall", "firewall", "delete", "rule",
                f"name=libaix_block_{ip}",
            ]
        else:
            iptables = shutil.which("iptables")
            if iptables is None:
                return SkillResult(
                    success=False,
                    message="Error: 'iptables' not found on this system",
                    data={},
                )
            cmd = [iptables, "-D", "INPUT", "-s", ip, "-j", "DROP"]

        return self._exec_firewall_cmd(cmd, f"Unblocked IP {ip}", ip=ip)

    def _list_rules(self, args: dict[str, Any]) -> SkillResult:
        """List current firewall rules."""
        if self._system == "Windows":
            cmd = [
                "netsh", "advfirewall", "firewall", "show", "rule",
                "name=all",
            ]
        else:
            iptables = shutil.which("iptables")
            if iptables is None:
                return SkillResult(
                    success=False,
                    message="Error: 'iptables' not found on this system",
                    data={},
                )
            cmd = [iptables, "-L", "-n", "--line-numbers"]

        return self._exec_firewall_cmd(cmd, "Firewall rules retrieved")

    def _firewall_status(self, args: dict[str, Any]) -> SkillResult:
        """Check whether the host firewall is active."""
        if self._system == "Windows":
            cmd = ["netsh", "advfirewall", "show", "allprofiles", "state"]
        else:
            # Try ufw first, then fall back to iptables rule count
            ufw = shutil.which("ufw")
            if ufw is not None:
                cmd = [ufw, "status"]
            else:
                iptables = shutil.which("iptables")
                if iptables is None:
                    return SkillResult(
                        success=False,
                        message="Error: neither 'ufw' nor 'iptables' found on this system",
                        data={},
                    )
                cmd = [iptables, "-L", "-n"]

        return self._exec_firewall_cmd(cmd, "Firewall status retrieved")

    # ── shared execution helper ──────────────────────────────────────

    def _exec_firewall_cmd(
        self,
        cmd: list[str],
        success_message: str,
        **extra_data: Any,
    ) -> SkillResult:
        """Run a firewall command and return a uniform ``SkillResult``."""
        try:
            proc = _run(cmd)
        except subprocess.TimeoutExpired:
            return SkillResult(
                success=False,
                message="Error: firewall command timed out",
                data={},
            )
        except PermissionError:
            return SkillResult(
                success=False,
                message="Error: insufficient permissions — try running as root/administrator",
                data={},
            )
        except FileNotFoundError as exc:
            return SkillResult(
                success=False,
                message=f"Error: command not found — {exc}",
                data={},
            )
        except OSError as exc:
            return SkillResult(
                success=False,
                message=f"Error: {exc}",
                data={},
            )

        if proc.returncode != 0:
            return SkillResult(
                success=False,
                message=f"Error: {proc.stderr.strip() or proc.stdout.strip()}",
                data={"returncode": proc.returncode, **extra_data},
            )

        return SkillResult(
            success=True,
            message=success_message,
            data={
                "output": proc.stdout.strip(),
                "returncode": proc.returncode,
                **extra_data,
            },
        )
