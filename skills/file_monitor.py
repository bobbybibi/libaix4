"""
file_monitor.py — File and process monitoring skill for libaix.

Provides commands to:
  • List running processes and flag suspicious ones
  • Monitor directories for file changes via SHA-256 snapshots
  • Compare current directory state against stored snapshots
  • Report system information (OS, Python version, disk usage, etc.)

Uses only the Python standard library.  No external packages required.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from skill_registry import Skill, SkillCommand, SkillResult

log = logging.getLogger(__name__)

# Limits
_MAX_FILES = 1000
_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
_SUBPROCESS_TIMEOUT = 30

# Process names / keywords commonly associated with suspicious activity
_SUSPICIOUS_NAMES = frozenset({
    "nc", "ncat", "netcat", "nmap", "msfconsole", "msfvenom",
    "mimikatz", "john", "hashcat", "hydra", "sqlmap",
    "cryptominer", "xmrig", "minerd", "coinhive",
    "reverse_shell", "bind_shell", "payload",
})

# Ports that may indicate suspicious listeners
_SUSPICIOUS_PORTS = frozenset({
    4444, 4445, 5555, 1337, 31337, 6666, 6667, 9001,
})


class FileMonitorSkill(Skill):
    """Monitor files and processes for suspicious activity."""

    def __init__(self) -> None:
        super().__init__(
            name="file_monitor",
            description="Monitor files and processes for suspicious activity",
            version="1.0.0",
            category="security",
        )
        self._snapshots: dict[str, dict[str, str]] = {}

    # ── Skill interface ──────────────────────────────────────────────

    def get_commands(self) -> list[SkillCommand]:
        return [
            SkillCommand(
                name="check_processes",
                description="List running processes and flag suspicious ones",
                patterns=[
                    r"(check|list|show)\s+(running\s+)?processes",
                    r"what.+running\s+(on\s+)?(my\s+)?(computer|machine|system)",
                ],
                args_schema={},
                category="security",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="watch_directory",
                description="Monitor a directory for changes",
                patterns=[
                    r"(watch|monitor)\s+(?P<path>\S+)\s*(directory|folder)?",
                    r"(watch|monitor)\s+(directory|folder)\s+(?P<path>\S+)",
                ],
                args_schema={"path": {"type": "string", "required": True}},
                category="security",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="check_directory_changes",
                description="Compare current directory state against stored snapshot",
                patterns=[
                    r"check\s+(for\s+)?changes\s+(in\s+)?(?P<path>\S+)",
                    r"(any|what)\s+changes\s+(in\s+)?(?P<path>\S+)",
                ],
                args_schema={"path": {"type": "string", "required": True}},
                category="security",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="system_info",
                description="Show system information",
                patterns=[
                    r"(show|get)\s+system\s+info",
                    r"system\s+(information|status|health)",
                ],
                args_schema={},
                category="security",
                requires_confirmation=False,
            ),
        ]

    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        dispatch = {
            "check_processes": self._check_processes,
            "watch_directory": self._watch_directory,
            "check_directory_changes": self._check_directory_changes,
            "system_info": self._system_info,
        }
        handler = dispatch.get(command)
        if handler is None:
            return SkillResult(
                success=False,
                message=f"Unknown command: {command}",
            )
        try:
            return handler(args)
        except Exception as exc:
            log.exception("file_monitor command '%s' failed: %s", command, exc)
            return SkillResult(
                success=False,
                message=f"Command '{command}' failed: {exc}",
            )

    # ── Command implementations ──────────────────────────────────────

    def _check_processes(self, args: dict[str, Any]) -> SkillResult:
        """List running processes and flag suspicious ones."""
        system = platform.system()

        if system == "Windows":
            cmd = ["tasklist"]
        else:
            if shutil.which("ps") is None:
                return SkillResult(
                    success=False,
                    message="'ps' command not found on this system.",
                )
            cmd = ["ps", "aux"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return SkillResult(
                success=False,
                message="Process listing timed out.",
            )
        except OSError as exc:
            return SkillResult(
                success=False,
                message=f"Failed to run process listing: {exc}",
            )

        if result.returncode != 0:
            return SkillResult(
                success=False,
                message=f"Process listing failed: {result.stderr.strip()}",
            )

        processes = self._parse_ps_output(result.stdout, system)
        suspicious = self._flag_suspicious(processes)

        return SkillResult(
            success=True,
            message=f"Found {len(processes)} process(es), {len(suspicious)} flagged as suspicious.",
            data={
                "processes": processes,
                "suspicious": suspicious,
                "total": len(processes),
                "suspicious_count": len(suspicious),
            },
        )

    def _watch_directory(self, args: dict[str, Any]) -> SkillResult:
        """Take a SHA-256 snapshot of a directory's files."""
        path_str = args.get("path", "")
        if not path_str:
            return SkillResult(
                success=False,
                message="No directory path provided.",
            )

        dir_path = Path(path_str).resolve()

        if not dir_path.exists():
            return SkillResult(
                success=False,
                message=f"Directory not found: {dir_path}",
            )
        if not dir_path.is_dir():
            return SkillResult(
                success=False,
                message=f"Path is not a directory: {dir_path}",
            )

        try:
            snapshot, skipped = self._build_snapshot(dir_path)
        except PermissionError as exc:
            return SkillResult(
                success=False,
                message=f"Permission denied: {exc}",
            )

        key = str(dir_path)
        self._snapshots[key] = snapshot

        return SkillResult(
            success=True,
            message=f"Watching {len(snapshot)} file(s) in {dir_path}.",
            data={
                "path": key,
                "file_count": len(snapshot),
                "files": sorted(snapshot.keys()),
                "skipped": skipped,
            },
        )

    def _check_directory_changes(self, args: dict[str, Any]) -> SkillResult:
        """Compare current directory state against stored snapshot."""
        path_str = args.get("path", "")
        if not path_str:
            return SkillResult(
                success=False,
                message="No directory path provided.",
            )

        dir_path = Path(path_str).resolve()
        key = str(dir_path)

        if key not in self._snapshots:
            return SkillResult(
                success=False,
                message=f"No snapshot stored for {dir_path}. Run watch_directory first.",
            )

        if not dir_path.exists():
            return SkillResult(
                success=False,
                message=f"Directory not found: {dir_path}",
            )

        try:
            current, _ = self._build_snapshot(dir_path)
        except PermissionError as exc:
            return SkillResult(
                success=False,
                message=f"Permission denied: {exc}",
            )

        old = self._snapshots[key]

        old_files = set(old.keys())
        new_files = set(current.keys())

        added = sorted(new_files - old_files)
        deleted = sorted(old_files - new_files)
        modified = sorted(
            f for f in old_files & new_files if old[f] != current[f]
        )

        has_changes = bool(added or deleted or modified)
        summary_parts: list[str] = []
        if added:
            summary_parts.append(f"{len(added)} new")
        if modified:
            summary_parts.append(f"{len(modified)} modified")
        if deleted:
            summary_parts.append(f"{len(deleted)} deleted")

        message = (
            f"Changes detected: {', '.join(summary_parts)}."
            if has_changes
            else "No changes detected."
        )

        return SkillResult(
            success=True,
            message=message,
            data={
                "path": key,
                "added": added,
                "modified": modified,
                "deleted": deleted,
                "has_changes": has_changes,
            },
        )

    def _system_info(self, args: dict[str, Any]) -> SkillResult:
        """Gather and return system information."""
        info: dict[str, Any] = {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        }

        # Uptime (Linux / macOS)
        uptime = self._get_uptime()
        if uptime is not None:
            info["uptime_seconds"] = uptime

        # Disk usage for root / current drive
        try:
            usage = shutil.disk_usage("/")
            info["disk"] = {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "used_percent": round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
            }
        except OSError:
            info["disk"] = None

        return SkillResult(
            success=True,
            message=f"System: {info['os']} {info['os_release']} ({info['architecture']})",
            data=info,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_ps_output(raw: str, system: str) -> list[dict[str, str]]:
        """Parse ``ps aux`` (Unix) or ``tasklist`` (Windows) output."""
        processes: list[dict[str, str]] = []
        lines = raw.strip().splitlines()

        if system == "Windows":
            # tasklist output: Image Name, PID, Session Name, Session#, Mem Usage
            for line in lines[3:]:  # skip header / separator
                parts = line.split()
                if len(parts) >= 2:
                    processes.append({
                        "name": parts[0],
                        "pid": parts[1],
                        "raw": line.strip(),
                    })
        else:
            # ps aux header: USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND
            for line in lines[1:]:
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    processes.append({
                        "user": parts[0],
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "command": parts[10],
                        "raw": line.strip(),
                    })
                elif parts:
                    processes.append({
                        "pid": parts[1] if len(parts) > 1 else "?",
                        "command": parts[-1],
                        "raw": line.strip(),
                    })

        return processes

    @staticmethod
    def _flag_suspicious(processes: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return processes whose name matches known-suspicious patterns."""
        flagged: list[dict[str, str]] = []
        for proc in processes:
            cmd = proc.get("command", proc.get("name", "")).lower()
            basename = cmd.split("/")[-1].split()[0] if cmd else ""

            reasons: list[str] = []
            if basename in _SUSPICIOUS_NAMES:
                reasons.append(f"suspicious process name: {basename}")

            # Check for listening on suspicious ports (e.g. "-p 4444" or ":4444")
            for port in _SUSPICIOUS_PORTS:
                if f":{port}" in cmd or f"-p {port}" in cmd or f"--port {port}" in cmd:
                    reasons.append(f"suspicious port: {port}")
                    break

            if reasons:
                flagged.append({**proc, "reasons": "; ".join(reasons)})

        return flagged

    def _build_snapshot(
        self, dir_path: Path
    ) -> tuple[dict[str, str], list[str]]:
        """Hash files in *dir_path*, returning ``(hashes, skipped)``."""
        hashes: dict[str, str] = {}
        skipped: list[str] = []
        count = 0

        for root, _dirs, files in os.walk(dir_path):
            for fname in files:
                if count >= _MAX_FILES:
                    skipped.append(f"(limit reached — stopped at {_MAX_FILES} files)")
                    return hashes, skipped

                fpath = Path(root) / fname
                rel = str(fpath.relative_to(dir_path))

                try:
                    size = fpath.stat().st_size
                except OSError:
                    skipped.append(rel)
                    continue

                if size > _MAX_FILE_SIZE:
                    skipped.append(rel)
                    continue

                try:
                    digest = self._hash_file(fpath)
                except (PermissionError, OSError):
                    skipped.append(rel)
                    continue

                hashes[rel] = digest
                count += 1

        return hashes, skipped

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Return the SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _get_uptime() -> float | None:
        """Return system uptime in seconds, or *None* if unavailable."""
        # Linux: /proc/uptime
        try:
            with open("/proc/uptime", "r") as f:
                return float(f.read().split()[0])
        except (FileNotFoundError, OSError, ValueError, IndexError):
            pass

        # macOS: sysctl kern.boottime
        if platform.system() == "Darwin" and shutil.which("sysctl"):
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "kern.boottime"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # Output: "{ sec = 1719370000, usec = 0 } ..."
                if result.returncode == 0:
                    import re
                    match = re.search(r"sec\s*=\s*(\d+)", result.stdout)
                    if match:
                        boot = int(match.group(1))
                        return time.time() - boot
            except (subprocess.TimeoutExpired, OSError):
                pass

        return None
