"""
smart_home.py — Smart home device control skill for libaix.

Provides commands to:
  • Turn devices on or off (HTTP, MQTT, Home Assistant)
  • List all registered smart devices
  • Check the status of a specific device
  • Register new devices into the local device registry

Device registry is persisted as JSON in ``data/smart_devices.json``.
Uses only the Python standard library.  No external packages required.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from skill_registry import Skill, SkillCommand, SkillResult

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_DATA_DIR = Path("data")
_DEVICES_FILE = _DATA_DIR / "smart_devices.json"
_HTTP_TIMEOUT = 10

_EXAMPLE_DEVICES: list[dict[str, str]] = [
    {
        "name": "living room lights",
        "type": "light",
        "protocol": "http",
        "endpoint": "http://192.168.1.100/api/light",
        "state": "off",
    },
    {
        "name": "TV",
        "type": "tv",
        "protocol": "http",
        "endpoint": "http://192.168.1.101/api/power",
        "state": "off",
    },
    {
        "name": "thermostat",
        "type": "thermostat",
        "protocol": "homeassistant",
        "endpoint": "http://homeassistant.local:8123/api/services/climate/set_hvac_mode",
        "state": "unknown",
    },
    {
        "name": "bedroom lights",
        "type": "light",
        "protocol": "mqtt",
        "endpoint": "home/bedroom/lights/set",
        "state": "off",
    },
]


# ── Skill implementation ─────────────────────────────────────────────

class SmartHomeSkill(Skill):
    """Control smart home devices — lights, TV, thermostat, etc."""

    def __init__(self) -> None:
        super().__init__(
            name="smart_home",
            description="Control smart home devices — lights, TV, thermostat, etc.",
            version="1.0.0",
            category="device",
        )
        self._devices: list[dict[str, str]] = self._load_devices()

    # ── public interface ─────────────────────────────────────────────

    def get_commands(self) -> list[SkillCommand]:
        return [
            SkillCommand(
                name="device_on",
                description="Turn a device on",
                patterns=[
                    r"turn\s+on\s+(the\s+)?(?P<device>.+)",
                    r"(switch|power)\s+on\s+(the\s+)?(?P<device>.+)",
                    r"(?P<device>.+)\s+on\s*$",
                ],
                args_schema={"device": {"type": "string", "required": True}},
                category="device",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="device_off",
                description="Turn a device off",
                patterns=[
                    r"turn\s+off\s+(the\s+)?(?P<device>.+)",
                    r"(switch|power)\s+off\s+(the\s+)?(?P<device>.+)",
                ],
                args_schema={"device": {"type": "string", "required": True}},
                category="device",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="list_devices",
                description="List known devices",
                patterns=[
                    r"(list|show)\s+(my\s+)?(smart\s+)?devices",
                    r"what\s+devices?\s+(do\s+i\s+have|are\s+connected)",
                ],
                args_schema={},
                category="device",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="device_status",
                description="Check a device's status",
                patterns=[
                    r"(status|state)\s+of\s+(the\s+)?(?P<device>.+)",
                    r"is\s+(the\s+)?(?P<device>.+)\s+(on|off|running)",
                ],
                args_schema={"device": {"type": "string", "required": True}},
                category="device",
                requires_confirmation=False,
            ),
            SkillCommand(
                name="add_device",
                description="Register a new device",
                patterns=[
                    r"add\s+(smart\s+)?device\s+(?P<device>.+)",
                    r"register\s+(smart\s+)?device\s+(?P<device>.+)",
                ],
                args_schema={
                    "device": {"type": "string", "required": True},
                    "type": {"type": "string", "default": "generic"},
                    "protocol": {"type": "string", "default": "http"},
                    "endpoint": {"type": "string", "default": ""},
                },
                category="device",
                requires_confirmation=False,
            ),
        ]

    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        """Dispatch *command* to the appropriate private handler."""
        dispatch: dict[str, Any] = {
            "device_on": self._device_on,
            "device_off": self._device_off,
            "list_devices": self._list_devices,
            "device_status": self._device_status,
            "add_device": self._add_device,
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
            log.exception("smart_home command '%s' failed: %s", command, exc)
            return SkillResult(
                success=False,
                message=f"Command '{command}' failed: {exc}",
            )

    # ── command implementations ──────────────────────────────────────

    def _device_on(self, args: dict[str, Any]) -> SkillResult:
        """Turn a device on."""
        return self._set_device_state(args, target_state="on")

    def _device_off(self, args: dict[str, Any]) -> SkillResult:
        """Turn a device off."""
        return self._set_device_state(args, target_state="off")

    def _set_device_state(
        self, args: dict[str, Any], *, target_state: str
    ) -> SkillResult:
        """Shared logic for turning a device on or off."""
        device_name = args.get("device", "").strip().lower()
        if not device_name:
            return SkillResult(
                success=False,
                message="No device name provided.",
            )

        device = self._find_device(device_name)
        if device is None:
            return SkillResult(
                success=False,
                message=f"Device '{device_name}' not found in registry.",
                data={"known_devices": [d["name"] for d in self._devices]},
            )

        protocol = device.get("protocol", "http")
        endpoint = device.get("endpoint", "")

        if protocol == "mqtt":
            # Generate the MQTT payload but note that an MQTT client is not
            # yet available — the caller should route this through a broker.
            payload = json.dumps({"state": target_state.upper()})
            device["state"] = target_state
            self._save_devices()
            return SkillResult(
                success=True,
                message=(
                    f"MQTT command prepared for '{device['name']}': "
                    f"topic={endpoint}, payload={payload}. "
                    "Note: MQTT client not yet available — publish manually."
                ),
                data={
                    "device": device["name"],
                    "protocol": "mqtt",
                    "topic": endpoint,
                    "payload": payload,
                    "state": target_state,
                },
            )

        if protocol == "homeassistant":
            # Build the Home Assistant REST API request.
            ha_payload = json.dumps({
                "entity_id": f"switch.{device_name.replace(' ', '_')}",
                "state": target_state,
            }).encode("utf-8")
            return self._send_http_command(
                device=device,
                endpoint=endpoint,
                payload=ha_payload,
                target_state=target_state,
                extra_headers={"Content-Type": "application/json"},
            )

        # Default: plain HTTP
        http_payload = json.dumps({"state": target_state}).encode("utf-8")
        return self._send_http_command(
            device=device,
            endpoint=endpoint,
            payload=http_payload,
            target_state=target_state,
            extra_headers={"Content-Type": "application/json"},
        )

    def _send_http_command(
        self,
        *,
        device: dict[str, str],
        endpoint: str,
        payload: bytes,
        target_state: str,
        extra_headers: dict[str, str] | None = None,
    ) -> SkillResult:
        """Send an HTTP POST to control a device."""
        if not endpoint:
            return SkillResult(
                success=False,
                message=f"No endpoint configured for device '{device['name']}'.",
            )

        req = urllib.request.Request(
            endpoint,
            data=payload,
            method="POST",
        )
        for key, value in (extra_headers or {}).items():
            req.add_header(key, value)

        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                status = resp.status
                body = resp.read(4096).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return SkillResult(
                success=False,
                message=(
                    f"HTTP error controlling '{device['name']}': "
                    f"{exc.code} {exc.reason}"
                ),
                data={"device": device["name"], "http_status": exc.code},
            )
        except urllib.error.URLError as exc:
            return SkillResult(
                success=False,
                message=(
                    f"Connection error for '{device['name']}': {exc.reason}"
                ),
                data={"device": device["name"]},
            )
        except OSError as exc:
            return SkillResult(
                success=False,
                message=f"Network error for '{device['name']}': {exc}",
                data={"device": device["name"]},
            )

        device["state"] = target_state
        self._save_devices()

        return SkillResult(
            success=True,
            message=f"'{device['name']}' turned {target_state} (HTTP {status}).",
            data={
                "device": device["name"],
                "state": target_state,
                "http_status": status,
                "response": body,
            },
        )

    def _list_devices(self, args: dict[str, Any]) -> SkillResult:
        """List all registered smart home devices."""
        if not self._devices:
            return SkillResult(
                success=True,
                message="No smart devices registered.",
                data={"devices": [], "count": 0},
            )

        summaries = [
            {
                "name": d["name"],
                "type": d.get("type", "generic"),
                "protocol": d.get("protocol", "http"),
                "state": d.get("state", "unknown"),
            }
            for d in self._devices
        ]

        return SkillResult(
            success=True,
            message=f"{len(summaries)} device(s) registered.",
            data={"devices": summaries, "count": len(summaries)},
        )

    def _device_status(self, args: dict[str, Any]) -> SkillResult:
        """Return the current known status of a device."""
        device_name = args.get("device", "").strip().lower()
        if not device_name:
            return SkillResult(
                success=False,
                message="No device name provided.",
            )

        device = self._find_device(device_name)
        if device is None:
            return SkillResult(
                success=False,
                message=f"Device '{device_name}' not found in registry.",
                data={"known_devices": [d["name"] for d in self._devices]},
            )

        return SkillResult(
            success=True,
            message=(
                f"'{device['name']}' ({device.get('type', 'generic')}) "
                f"is {device.get('state', 'unknown')}."
            ),
            data={
                "name": device["name"],
                "type": device.get("type", "generic"),
                "protocol": device.get("protocol", "http"),
                "endpoint": device.get("endpoint", ""),
                "state": device.get("state", "unknown"),
            },
        )

    def _add_device(self, args: dict[str, Any]) -> SkillResult:
        """Register a new device in the local registry."""
        device_name = args.get("device", "").strip().lower()
        if not device_name:
            return SkillResult(
                success=False,
                message="No device name provided.",
            )

        # Check for duplicates
        if self._find_device(device_name) is not None:
            return SkillResult(
                success=False,
                message=f"Device '{device_name}' is already registered.",
            )

        new_device: dict[str, str] = {
            "name": device_name,
            "type": args.get("type", "generic"),
            "protocol": args.get("protocol", "http"),
            "endpoint": args.get("endpoint", ""),
            "state": "unknown",
        }
        self._devices.append(new_device)
        self._save_devices()

        return SkillResult(
            success=True,
            message=f"Device '{device_name}' registered successfully.",
            data=new_device,
        )

    # ── helpers ──────────────────────────────────────────────────────

    def _find_device(self, name: str) -> dict[str, str] | None:
        """Find a device by name (case-insensitive partial match)."""
        name_lower = name.lower()
        # Exact match first
        for d in self._devices:
            if d["name"].lower() == name_lower:
                return d
        # Partial match fallback
        for d in self._devices:
            if name_lower in d["name"].lower():
                return d
        return None

    def _load_devices(self) -> list[dict[str, str]]:
        """Load the device registry from disk, seeding with examples if absent."""
        if _DEVICES_FILE.exists():
            try:
                with open(_DEVICES_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return data
                log.warning("smart_devices.json has unexpected format — using defaults.")
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to load smart_devices.json: %s — using defaults.", exc)

        # Seed with example devices
        devices = list(_EXAMPLE_DEVICES)
        self._save_devices_to(devices)
        return devices

    def _save_devices(self) -> None:
        """Persist the current device registry to disk."""
        self._save_devices_to(self._devices)

    @staticmethod
    def _save_devices_to(devices: list[dict[str, str]]) -> None:
        """Write *devices* list to the JSON file."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(_DEVICES_FILE, "w", encoding="utf-8") as fh:
                json.dump(devices, fh, indent=2)
        except OSError as exc:
            log.error("Failed to save smart_devices.json: %s", exc)
