"""
vpn_manager.py — VPN management skill for libaix.

Provides commands to manage VPN connections:
  • Connect to a VPN (WireGuard via ``wg-quick`` or OpenVPN)
  • Disconnect from an active VPN session
  • Check VPN connection status
  • Show and manage VPN configuration files

VPN provider is auto-detected by checking for ``wg-quick`` and
``openvpn`` on the system PATH via ``shutil.which``.  Configuration
is persisted in ``data/vpn_config.json``.

Uses only the Python standard library.  No external packages required.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from skill_registry import Skill, SkillCommand, SkillResult

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_SUBPROCESS_TIMEOUT = 30
_CONFIG_PATH = Path("data/vpn_config.json")

_DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "auto",
    "interface": "wg0",
    "config_file": "",
    "auto_connect": False,
    "dns_leak_protection": True,
    "kill_switch": False,
}


# ── Helpers ──────────────────────────────────────────────────────────

def _load_config() -> dict[str, Any]:
    """Load VPN configuration from disk, returning defaults on failure."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Merge with defaults so new keys are always present
        merged = {**_DEFAULT_CONFIG, **data}
        return merged
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(_DEFAULT_CONFIG)


def _save_config(config: dict[str, Any]) -> None:
    """Persist VPN configuration to disk."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


def _detect_provider() -> str | None:
    """Return ``'wireguard'``, ``'openvpn'``, or ``None``."""
    if shutil.which("wg-quick") is not None:
        return "wireguard"
    if shutil.which("openvpn") is not None:
        return "openvpn"
    return None


def _run(cmd: list[str], timeout: int = _SUBPROCESS_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """Run *cmd* with timeout and capture output."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ── Skill implementation ─────────────────────────────────────────────

class VPNManagerSkill(Skill):
    """Connect, disconnect, and manage VPN sessions."""

    def __init__(self) -> None:
        super().__init__(
            name="vpn_manager",
            description="Connect, disconnect, and manage VPN sessions",
            version="1.0.0",
            category="network",
        )
        self._config = _load_config()
        self._system = platform.system()

    # ── public interface ─────────────────────────────────────────────

    def get_commands(self) -> list[SkillCommand]:
        return [
            SkillCommand(
                name="vpn_connect",
                description="Connect to a VPN",
                patterns=[
                    r"(connect|start)\s+(to\s+)?vpn",
                    r"vpn\s+(connect|start|on|up)",
                ],
                args_schema={
                    "interface": {"type": "string", "optional": True},
                    "config_file": {"type": "string", "optional": True},
                },
                category="network",
                requires_confirmation=True,
            ),
            SkillCommand(
                name="vpn_disconnect",
                description="Disconnect from VPN",
                patterns=[
                    r"(disconnect|stop)\s+(from\s+)?vpn",
                    r"vpn\s+(disconnect|stop|off|down)",
                ],
                args_schema={
                    "interface": {"type": "string", "optional": True},
                },
                category="network",
                requires_confirmation=True,
            ),
            SkillCommand(
                name="vpn_status",
                description="Check VPN connection status",
                patterns=[
                    r"vpn\s+status",
                    r"(is|check)\s+(the\s+)?vpn\s+(on|active|connected|running)",
                ],
                args_schema={},
                category="network",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="vpn_config",
                description="Show or manage VPN configuration",
                patterns=[
                    r"(show|list)\s+vpn\s+config",
                ],
                args_schema={},
                category="network",
                requires_confirmation=False,
            ),
        ]

    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        """Dispatch *command* to the appropriate private handler."""
        dispatch: dict[str, Any] = {
            "vpn_connect": self._vpn_connect,
            "vpn_disconnect": self._vpn_disconnect,
            "vpn_status": self._vpn_status,
            "vpn_config": self._vpn_config,
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
            log.exception("VPNManagerSkill.%s failed: %s", command, exc)
            return SkillResult(
                success=False,
                message=f"Error: {exc}",
                data={},
            )

    # ── private handlers ─────────────────────────────────────────────

    def _vpn_connect(self, args: dict[str, Any]) -> SkillResult:
        """Connect to a VPN using WireGuard or OpenVPN."""
        provider = _detect_provider()
        if provider is None:
            return SkillResult(
                success=False,
                message="Error: neither WireGuard (wg-quick) nor OpenVPN found on this system",
                data={"checked": ["wg-quick", "openvpn"]},
            )

        interface = args.get("interface", self._config.get("interface", "wg0"))
        config_file = args.get("config_file", self._config.get("config_file", ""))

        if provider == "wireguard":
            cmd = ["wg-quick", "up", interface]
        else:
            # OpenVPN requires a config file
            if not config_file:
                return SkillResult(
                    success=False,
                    message="Error: OpenVPN requires a config file — set 'config_file' in vpn_config",
                    data={"provider": provider},
                )
            if not Path(config_file).is_file():
                return SkillResult(
                    success=False,
                    message=f"Error: config file not found: {config_file}",
                    data={"provider": provider, "config_file": config_file},
                )
            cmd = ["openvpn", "--config", config_file, "--daemon"]

        return self._exec_vpn_cmd(
            cmd,
            f"VPN connected via {provider} (interface={interface})",
            provider=provider,
            interface=interface,
        )

    def _vpn_disconnect(self, args: dict[str, Any]) -> SkillResult:
        """Disconnect the active VPN session."""
        provider = _detect_provider()
        if provider is None:
            return SkillResult(
                success=False,
                message="Error: neither WireGuard (wg-quick) nor OpenVPN found on this system",
                data={},
            )

        interface = args.get("interface", self._config.get("interface", "wg0"))

        if provider == "wireguard":
            cmd = ["wg-quick", "down", interface]
        else:
            # OpenVPN daemon — find and signal the process
            killall = shutil.which("killall")
            if killall is not None:
                cmd = [killall, "openvpn"]
            else:
                # Fallback: use pkill-style approach via subprocess
                return SkillResult(
                    success=False,
                    message="Error: cannot stop OpenVPN — 'killall' not found",
                    data={"provider": provider},
                )

        return self._exec_vpn_cmd(
            cmd,
            f"VPN disconnected ({provider}, interface={interface})",
            provider=provider,
            interface=interface,
        )

    def _vpn_status(self, args: dict[str, Any]) -> SkillResult:
        """Check whether a VPN connection is currently active."""
        provider = _detect_provider()
        results: dict[str, Any] = {"provider": provider}

        if provider == "wireguard":
            wg = shutil.which("wg")
            if wg is not None:
                try:
                    proc = _run([wg, "show"])
                    output = proc.stdout.strip()
                    active = proc.returncode == 0 and len(output) > 0
                    results["active"] = active
                    results["output"] = output
                    message = "WireGuard VPN is active" if active else "WireGuard VPN is not active"
                    return SkillResult(success=True, message=message, data=results)
                except (subprocess.TimeoutExpired, PermissionError, OSError) as exc:
                    return SkillResult(
                        success=False,
                        message=f"Error checking WireGuard status: {exc}",
                        data=results,
                    )

        # Generic check: look for tun/tap interfaces
        if self._system != "Windows":
            try:
                proc = _run(["ip", "link", "show"])
                output = proc.stdout
                has_tun = "tun" in output or "wg" in output
                results["active"] = has_tun
                results["output"] = output.strip()
                message = "VPN tunnel interface detected" if has_tun else "No VPN tunnel interface detected"
                return SkillResult(success=True, message=message, data=results)
            except (subprocess.TimeoutExpired, PermissionError, FileNotFoundError, OSError):
                pass

        # Windows: check for active VPN adapters
        if self._system == "Windows":
            try:
                proc = _run(["ipconfig", "/all"])
                output = proc.stdout
                has_vpn = "TAP" in output or "WireGuard" in output or "Wintun" in output
                results["active"] = has_vpn
                results["output"] = output.strip()
                message = "VPN adapter detected" if has_vpn else "No VPN adapter detected"
                return SkillResult(success=True, message=message, data=results)
            except (subprocess.TimeoutExpired, PermissionError, FileNotFoundError, OSError):
                pass

        return SkillResult(
            success=False,
            message="Error: unable to determine VPN status on this system",
            data=results,
        )

    def _vpn_config(self, args: dict[str, Any]) -> SkillResult:
        """Return the current VPN configuration."""
        config = _load_config()
        provider = _detect_provider()
        available_tools: list[str] = []
        for tool in ("wg-quick", "wg", "openvpn"):
            if shutil.which(tool) is not None:
                available_tools.append(tool)

        return SkillResult(
            success=True,
            message="VPN configuration loaded",
            data={
                "config": config,
                "detected_provider": provider,
                "available_tools": available_tools,
                "config_path": str(_CONFIG_PATH),
            },
        )

    # ── shared execution helper ──────────────────────────────────────

    def _exec_vpn_cmd(
        self,
        cmd: list[str],
        success_message: str,
        **extra_data: Any,
    ) -> SkillResult:
        """Run a VPN command and return a uniform ``SkillResult``."""
        try:
            proc = _run(cmd)
        except subprocess.TimeoutExpired:
            return SkillResult(
                success=False,
                message="Error: VPN command timed out",
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
