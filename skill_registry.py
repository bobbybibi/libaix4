"""
skill_registry.py — Plugin/Skill framework and registry for libaix.

Provides the foundation for the agent skill system:
  • SkillResult — standard result envelope for every skill execution
  • Skill ABC — abstract base class that every skill must inherit from
  • SkillCommand — declarative description of a command a skill can handle
  • IntentMatch — result of matching user input to a skill command
  • SkillRegistry — central registry that manages skills, intent matching,
    and command dispatch
  • get_registry() — singleton accessor for the global registry instance

Skills register themselves with the registry and declare regex patterns
for intent matching.  The registry compiles all patterns at registration
time and scores matches by specificity (longer match → higher confidence).

No external packages are used — stdlib only.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class SkillResult:
    """Result envelope returned by every skill execution."""
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    background_task_id: str | None = None


@dataclass
class SkillCommand:
    """Describes a single command that a skill can handle."""
    name: str
    description: str
    patterns: list[str]
    args_schema: dict[str, Any] = field(default_factory=dict)
    category: str = ""
    requires_confirmation: bool = False


@dataclass
class IntentMatch:
    """Result of matching user input against registered skill patterns."""
    skill_name: str
    command_name: str
    confidence: float
    extracted_args: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


# ── Compiled pattern wrapper (internal) ──────────────────────────────

@dataclass
class _CompiledPattern:
    """Internal wrapper that links a compiled regex back to its skill/command."""
    regex: re.Pattern[str]
    raw_pattern: str
    skill_name: str
    command_name: str
    requires_confirmation: bool


# ── Skill ABC ────────────────────────────────────────────────────────

class Skill(ABC):
    """Abstract base class that every libaix skill must inherit from.

    Subclasses *must* implement :meth:`execute` and :meth:`get_commands`.
    They *should* set the four identity properties in ``__init__``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "0.1.0",
        category: str = "general",
    ) -> None:
        self._name = name
        self._description = description
        self._version = version
        self._category = category

    # ── identity properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def version(self) -> str:
        return self._version

    @property
    def category(self) -> str:
        return self._category

    # ── abstract methods ─────────────────────────────────────────────

    @abstractmethod
    def execute(self, command: str, args: dict[str, Any]) -> SkillResult:
        """Execute a command with the given arguments."""
        ...

    @abstractmethod
    def get_commands(self) -> list[SkillCommand]:
        """Return the list of commands this skill can handle."""
        ...

    # ── optional overrides ───────────────────────────────────────────

    def get_status(self) -> dict:
        """Return health / status information for this skill."""
        return {
            "name": self._name,
            "version": self._version,
            "category": self._category,
            "status": "ok",
            "timestamp": time.time(),
        }

    def cleanup(self) -> None:
        """Release any resources held by this skill.  Override as needed."""


# ── SkillRegistry ────────────────────────────────────────────────────

class SkillRegistry:
    """Central registry that manages skills, intent matching, and dispatch."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._compiled: list[_CompiledPattern] = []

    # ── registration ─────────────────────────────────────────────────

    def register(self, skill: Skill) -> None:
        """Register a skill and compile its command patterns."""
        if skill.name in self._skills:
            log.warning("Skill '%s' is already registered — replacing.", skill.name)
            self.unregister(skill.name)

        self._skills[skill.name] = skill

        for cmd in skill.get_commands():
            for pattern in cmd.patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    log.error(
                        "Invalid regex in skill '%s' command '%s': %s",
                        skill.name, cmd.name, exc,
                    )
                    continue
                self._compiled.append(
                    _CompiledPattern(
                        regex=compiled,
                        raw_pattern=pattern,
                        skill_name=skill.name,
                        command_name=cmd.name,
                        requires_confirmation=cmd.requires_confirmation,
                    )
                )
        log.info(
            "Registered skill '%s' v%s (%s) with %d command(s).",
            skill.name, skill.version, skill.category,
            len(skill.get_commands()),
        )

    def unregister(self, name: str) -> None:
        """Remove a skill and its compiled patterns from the registry."""
        skill = self._skills.pop(name, None)
        if skill is None:
            log.warning("Attempted to unregister unknown skill '%s'.", name)
            return
        skill.cleanup()
        self._compiled = [
            cp for cp in self._compiled if cp.skill_name != name
        ]
        log.info("Unregistered skill '%s'.", name)

    # ── lookup ───────────────────────────────────────────────────────

    def get_skill(self, name: str) -> Skill | None:
        """Return a registered skill by name, or ``None``."""
        return self._skills.get(name)

    def list_skills(self) -> list[dict]:
        """Return summary dicts for every registered skill and its commands."""
        result: list[dict] = []
        for skill in self._skills.values():
            commands = [
                {
                    "name": cmd.name,
                    "description": cmd.description,
                    "category": cmd.category,
                    "requires_confirmation": cmd.requires_confirmation,
                }
                for cmd in skill.get_commands()
            ]
            result.append({
                "name": skill.name,
                "description": skill.description,
                "version": skill.version,
                "category": skill.category,
                "status": skill.get_status(),
                "commands": commands,
            })
        return result

    def get_all_commands(self) -> list[SkillCommand]:
        """Return every command from every registered skill."""
        commands: list[SkillCommand] = []
        for skill in self._skills.values():
            commands.extend(skill.get_commands())
        return commands

    # ── intent matching ──────────────────────────────────────────────

    def match_intent(self, user_input: str) -> IntentMatch | None:
        """Match *user_input* against all compiled patterns.

        Confidence is derived from match quality: the ratio of the matched
        span length to the total input length gives a base score, capped
        at ``1.0``.  When multiple patterns match, the one with the
        longest matched span (highest confidence) wins.

        Named groups in the regex pattern are extracted as arguments.

        Returns ``None`` when no pattern matches.
        """
        if not user_input or not self._compiled:
            return None

        best: IntentMatch | None = None
        best_confidence: float = 0.0
        input_len = len(user_input)

        for cp in self._compiled:
            m = cp.regex.search(user_input)
            if m is None:
                continue

            matched_len = m.end() - m.start()
            confidence = min(matched_len / max(input_len, 1), 1.0)

            if confidence > best_confidence:
                # Extract named groups as args (drop None values)
                extracted = {
                    k: v for k, v in m.groupdict().items() if v is not None
                }
                best_confidence = confidence
                best = IntentMatch(
                    skill_name=cp.skill_name,
                    command_name=cp.command_name,
                    confidence=round(confidence, 4),
                    extracted_args=extracted,
                    requires_confirmation=cp.requires_confirmation,
                )

        return best

    # ── execution ────────────────────────────────────────────────────

    def execute(self, skill_name: str, command: str, args: dict[str, Any]) -> SkillResult:
        """Dispatch *command* with *args* to the named skill."""
        skill = self._skills.get(skill_name)
        if skill is None:
            return SkillResult(
                success=False,
                message=f"Skill '{skill_name}' is not registered.",
            )
        try:
            return skill.execute(command, args)
        except Exception as exc:
            log.exception(
                "Skill '%s' raised while executing '%s': %s",
                skill_name, command, exc,
            )
            return SkillResult(
                success=False,
                message=f"Skill '{skill_name}' failed: {exc}",
            )


# ── Singleton accessor ───────────────────────────────────────────────

_registry_instance: SkillRegistry | None = None

def get_registry() -> SkillRegistry:
    """Return the global :class:`SkillRegistry` instance (created on first call)."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SkillRegistry()
    return _registry_instance
