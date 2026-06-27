"""
network_scanner.py — Network scanning skill for libaix.

Provides network scanning capabilities using Python stdlib only:
  • Discover devices on the local network (ARP scan)
  • Scan ports on a target host (socket connect)
  • List active network connections (netstat / ss)
  • Detect suspicious network activity (known-bad port checks)

All external tool invocations use subprocess with timeouts and
graceful error handling.  No third-party packages required.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import socket
import subprocess
from typing import Any

from skill_registry import Skill, SkillCommand, SkillResult

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

COMMON_PORTS: list[int] = [
    21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995,
    3306, 3389, 5432, 6379, 8080, 8443, 9090,
]

SUSPICIOUS_PORTS: set[int] = {1337, 4444, 5555, 6667, 31337}

PORT_SERVICE_NAMES: dict[int, str] = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    143: "IMAP",
    443: "HTTPS",
    993: "IMAPS",
    995: "POP3S",
    1337: "waste",
    3306: "MySQL",
    3389: "RDP",
    4444: "metasploit-default",
    5432: "PostgreSQL",
    5555: "adb",
    6379: "Redis",
    6667: "IRC",
    8080: "HTTP-Alt",
    8443: "HTTPS-Alt",
    9090: "Prometheus",
    31337: "Back Orifice",
}

_SUBPROCESS_TIMEOUT = 30
_PORT_SCAN_TIMEOUT = 0.5


# ── Skill implementation ─────────────────────────────────────────────

class NetworkScannerSkill(Skill):
    """Scan networks for devices, open ports, and security issues."""

    def __init__(self) -> None:
        super().__init__(
            name="network_scanner",
            description="Scan networks for devices, open ports, and security issues",
            version="1.0.0",
            category="security",
        )

    # ── public interface ─────────────────────────────────────────────

    def get_commands(self) -> list[SkillCommand]:
        return [
            SkillCommand(
                name="scan_network",
                description="Scan local network for connected devices",
                patterns=[
                    r"scan\s+(my\s+)?network",
                    r"find\s+devices?\s+(on|in)\s+(my\s+)?network",
                    r"who\s+is\s+on\s+my\s+network",
                ],
                args_schema={},
                category="security",
                requires_confirmation=True,
            ),
            SkillCommand(
                name="scan_ports",
                description="Scan ports on a target host",
                patterns=[
                    r"scan\s+ports?\s+(on\s+)?(?P<target>\S+)",
                    r"port\s+scan\s+(?P<target>\S+)",
                ],
                args_schema={
                    "target": {"type": "string", "default": "127.0.0.1"},
                    "ports": {"type": "array", "items": {"type": "integer"}, "optional": True},
                },
                category="security",
                requires_confirmation=True,
            ),
            SkillCommand(
                name="check_connections",
                description="Show active network connections",
                patterns=[
                    r"(show|list|check)\s+(active\s+)?connections",
                    r"what.+connected",
                ],
                args_schema={},
                category="security",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="detect_suspicious",
                description="Check for suspicious network activity",
                patterns=[
                    r"(detect|find|check)\s+suspicious\s+(activity|connections)",
                    r"am\s+i\s+(being\s+)?hacked",
                ],
                args_schema={},
                category="security",
                requires_confirmation=False,
            ),
        ]

    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        """Dispatch *command* to the appropriate private handler."""
        dispatch: dict[str, Any] = {
            "scan_network": self._scan_network,
            "scan_ports": self._scan_ports,
            "check_connections": self._check_connections,
            "detect_suspicious": self._detect_suspicious,
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
            log.exception("NetworkScannerSkill.%s failed: %s", command, exc)
            return SkillResult(
                success=False,
                message=f"Error: {exc}",
                data={},
            )

    # ── private handlers ─────────────────────────────────────────────

    def _scan_network(self, args: dict[str, Any]) -> SkillResult:
        """Discover devices on the local network via ``arp -a``."""
        arp_path = shutil.which("arp")
        if arp_path is None:
            return SkillResult(
                success=False,
                message="Error: 'arp' command not found on this system",
                data={},
            )

        try:
            proc = subprocess.run(
                [arp_path, "-a"],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return SkillResult(
                success=False,
                message="Error: ARP scan timed out",
                data={},
            )
        except PermissionError as exc:
            return SkillResult(
                success=False,
                message=f"Error: insufficient permissions — {exc}",
                data={},
            )
        except OSError as exc:
            return SkillResult(
                success=False,
                message=f"Error: {exc}",
                data={},
            )

        devices = _parse_arp_output(proc.stdout)
        return SkillResult(
            success=True,
            message=f"Found {len(devices)} device(s) on the network",
            data={"devices": devices, "raw_output": proc.stdout},
        )

    def _scan_ports(self, args: dict[str, Any]) -> SkillResult:
        """Scan TCP ports on *target* using ``socket.connect_ex``."""
        target: str = args.get("target", "127.0.0.1")
        ports: list[int] = args.get("ports") or COMMON_PORTS

        # Resolve hostname once
        try:
            resolved_ip = socket.gethostbyname(target)
        except socket.gaierror as exc:
            return SkillResult(
                success=False,
                message=f"Error: cannot resolve host '{target}' — {exc}",
                data={},
            )

        open_ports: list[dict[str, Any]] = []
        closed_count = 0

        for port in ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(_PORT_SCAN_TIMEOUT)
                    result = sock.connect_ex((resolved_ip, port))
                if result == 0:
                    service = PORT_SERVICE_NAMES.get(port, "unknown")
                    open_ports.append({"port": port, "service": service, "state": "open"})
                else:
                    closed_count += 1
            except OSError:
                closed_count += 1

        return SkillResult(
            success=True,
            message=(
                f"Scanned {len(ports)} port(s) on {target} ({resolved_ip}): "
                f"{len(open_ports)} open, {closed_count} closed/filtered"
            ),
            data={
                "target": target,
                "resolved_ip": resolved_ip,
                "open_ports": open_ports,
                "scanned_count": len(ports),
                "closed_count": closed_count,
            },
        )

    def _check_connections(self, args: dict[str, Any]) -> SkillResult:
        """List active network connections via ``ss`` or ``netstat``."""
        tool_path, tool_args = _find_connection_tool()
        if tool_path is None:
            return SkillResult(
                success=False,
                message="Error: neither 'ss' nor 'netstat' found on this system",
                data={},
            )

        try:
            proc = subprocess.run(
                [tool_path] + tool_args,
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return SkillResult(
                success=False,
                message="Error: connection listing timed out",
                data={},
            )
        except PermissionError as exc:
            return SkillResult(
                success=False,
                message=f"Error: insufficient permissions — {exc}",
                data={},
            )
        except OSError as exc:
            return SkillResult(
                success=False,
                message=f"Error: {exc}",
                data={},
            )

        connections = _parse_connection_output(proc.stdout)
        return SkillResult(
            success=True,
            message=f"Found {len(connections)} active connection(s)",
            data={"connections": connections, "raw_output": proc.stdout},
        )

    def _detect_suspicious(self, args: dict[str, Any]) -> SkillResult:
        """Flag connections to known-suspicious ports."""
        conn_result = self._check_connections(args)
        if not conn_result.success:
            return conn_result

        connections: list[dict[str, Any]] = conn_result.data.get("connections", [])
        suspicious: list[dict[str, Any]] = []

        for conn in connections:
            remote_port = conn.get("remote_port")
            if remote_port is not None and remote_port in SUSPICIOUS_PORTS:
                suspicious.append({
                    **conn,
                    "reason": (
                        f"Connection to suspicious port {remote_port} "
                        f"({PORT_SERVICE_NAMES.get(remote_port, 'unknown')})"
                    ),
                })

        if suspicious:
            message = f"WARNING: {len(suspicious)} suspicious connection(s) detected"
        else:
            message = "No suspicious connections detected"

        return SkillResult(
            success=True,
            message=message,
            data={
                "suspicious": suspicious,
                "total_connections": len(connections),
                "suspicious_count": len(suspicious),
            },
        )


# ── Helper functions ─────────────────────────────────────────────────

def _parse_arp_output(output: str) -> list[dict[str, str]]:
    """Parse the output of ``arp -a`` into a list of device dicts."""
    devices: list[dict[str, str]] = []
    # Common arp -a format:  hostname (ip) at mac [ether] on iface
    pattern = re.compile(
        r"^(?P<hostname>\S+)\s+\((?P<ip>[\d.]+)\)\s+at\s+(?P<mac>[\w:]+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(output):
        mac = m.group("mac")
        if mac == "<incomplete>":
            continue
        devices.append({
            "hostname": m.group("hostname"),
            "ip": m.group("ip"),
            "mac": mac,
        })
    return devices


def _find_connection_tool() -> tuple[str | None, list[str]]:
    """Return the path and default args for ``ss`` or ``netstat``."""
    ss_path = shutil.which("ss")
    if ss_path is not None:
        return ss_path, ["-tunap"]

    netstat_path = shutil.which("netstat")
    if netstat_path is not None:
        return netstat_path, ["-tunap"]

    return None, []


def _parse_connection_output(output: str) -> list[dict[str, Any]]:
    """Best-effort parse of ``ss`` / ``netstat`` output into dicts."""
    connections: list[dict[str, Any]] = []
    lines = output.strip().splitlines()

    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) < 5:
            continue

        proto = parts[0]
        local_addr = parts[3] if len(parts) > 3 else ""
        remote_addr = parts[4] if len(parts) > 4 else ""

        local_ip, local_port = _split_address(local_addr)
        remote_ip, remote_port = _split_address(remote_addr)

        connections.append({
            "protocol": proto,
            "local_address": local_addr,
            "local_ip": local_ip,
            "local_port": local_port,
            "remote_address": remote_addr,
            "remote_ip": remote_ip,
            "remote_port": remote_port,
            "state": parts[5] if len(parts) > 5 else "",
        })

    return connections


def _split_address(addr: str) -> tuple[str, int | None]:
    """Split ``ip:port`` or ``[ipv6]:port`` into ``(ip, port)``."""
    if not addr:
        return ("", None)
    # IPv6: [::1]:port
    if addr.startswith("["):
        bracket_end = addr.rfind("]")
        if bracket_end != -1 and bracket_end + 1 < len(addr) and addr[bracket_end + 1] == ":":
            ip = addr[1:bracket_end]
            port_str = addr[bracket_end + 2:]
            try:
                return (ip, int(port_str))
            except ValueError:
                return (ip, None)
    # IPv4: last colon separates ip and port
    sep = addr.rfind(":")
    if sep != -1:
        ip = addr[:sep]
        port_str = addr[sep + 1:]
        try:
            return (ip, int(port_str))
        except ValueError:
            return (ip, None)
    return (addr, None)
