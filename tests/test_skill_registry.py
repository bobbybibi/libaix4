"""Tests for skill_registry.py."""
from __future__ import annotations


import pytest

from skill_registry import (
    IntentMatch,
    Skill,
    SkillCommand,
    SkillRegistry,
    SkillResult,
    get_registry,
)


# ── Helper: concrete Skill subclass for testing ─────────────────────

class MockSkill(Skill):
    """Minimal concrete skill for testing the registry."""

    def __init__(
        self,
        name: str = "mock_skill",
        description: str = "A mock skill",
        version: str = "0.1.0",
        category: str = "testing",
        commands: list[SkillCommand] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, version=version, category=category)
        self._commands = commands or [
            SkillCommand(
                name="greet",
                description="Say hello",
                patterns=[r"hello", r"hi\s+there"],
                category="testing",
            ),
            SkillCommand(
                name="echo",
                description="Echo a message",
                patterns=[r"echo\s+(?P<message>.+)"],
                category="testing",
            ),
        ]
        self.cleanup_called = False

    def execute(self, command: str, args: dict) -> SkillResult:
        if command == "greet":
            return SkillResult(success=True, message="Hello!")
        if command == "echo":
            msg = args.get("message", "")
            return SkillResult(success=True, message=msg, data={"echo": msg})
        if command == "fail":
            raise RuntimeError("boom")
        return SkillResult(success=False, message=f"Unknown command: {command}")

    def get_commands(self) -> list[SkillCommand]:
        return self._commands

    def cleanup(self) -> None:
        self.cleanup_called = True


# ── SkillResult tests ────────────────────────────────────────────────

class TestSkillResult:
    def test_creation_with_all_fields(self):
        r = SkillResult(success=True, message="ok", data={"key": "val"}, background_task_id="t1")
        assert r.success is True
        assert r.message == "ok"
        assert r.data == {"key": "val"}
        assert r.background_task_id == "t1"

    def test_defaults(self):
        r = SkillResult(success=False, message="err")
        assert r.data == {}
        assert r.background_task_id is None

    def test_data_default_is_independent(self):
        r1 = SkillResult(success=True, message="a")
        r2 = SkillResult(success=True, message="b")
        r1.data["x"] = 1
        assert "x" not in r2.data

    def test_success_true(self):
        r = SkillResult(success=True, message="done")
        assert r.success is True

    def test_success_false(self):
        r = SkillResult(success=False, message="fail")
        assert r.success is False


# ── SkillCommand tests ───────────────────────────────────────────────

class TestSkillCommand:
    def test_creation(self):
        cmd = SkillCommand(name="test", description="desc", patterns=[r"test"])
        assert cmd.name == "test"
        assert cmd.description == "desc"
        assert cmd.patterns == [r"test"]

    def test_defaults(self):
        cmd = SkillCommand(name="x", description="y", patterns=[])
        assert cmd.args_schema == {}
        assert cmd.category == ""
        assert cmd.requires_confirmation is False

    def test_requires_confirmation(self):
        cmd = SkillCommand(name="rm", description="delete", patterns=[], requires_confirmation=True)
        assert cmd.requires_confirmation is True

    def test_args_schema(self):
        schema = {"ip": {"type": "string"}}
        cmd = SkillCommand(name="a", description="b", patterns=[], args_schema=schema)
        assert cmd.args_schema == schema

    def test_category(self):
        cmd = SkillCommand(name="a", description="b", patterns=[], category="security")
        assert cmd.category == "security"


# ── IntentMatch tests ────────────────────────────────────────────────

class TestIntentMatch:
    def test_creation(self):
        m = IntentMatch(skill_name="s", command_name="c", confidence=0.9)
        assert m.skill_name == "s"
        assert m.command_name == "c"
        assert m.confidence == 0.9

    def test_defaults(self):
        m = IntentMatch(skill_name="s", command_name="c", confidence=0.5)
        assert m.extracted_args == {}
        assert m.requires_confirmation is False

    def test_with_extracted_args(self):
        m = IntentMatch(skill_name="s", command_name="c", confidence=0.8, extracted_args={"ip": "1.2.3.4"})
        assert m.extracted_args == {"ip": "1.2.3.4"}

    def test_requires_confirmation_flag(self):
        m = IntentMatch(skill_name="s", command_name="c", confidence=1.0, requires_confirmation=True)
        assert m.requires_confirmation is True


# ── Skill ABC tests ─────────────────────────────────────────────────

class TestSkillABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Skill(name="x", description="y")  # type: ignore[abstract]

    def test_concrete_subclass(self):
        s = MockSkill()
        assert s.name == "mock_skill"
        assert s.description == "A mock skill"
        assert s.version == "0.1.0"
        assert s.category == "testing"

    def test_get_commands(self):
        s = MockSkill()
        cmds = s.get_commands()
        assert len(cmds) == 2
        assert cmds[0].name == "greet"

    def test_execute(self):
        s = MockSkill()
        r = s.execute("greet", {})
        assert r.success is True
        assert r.message == "Hello!"

    def test_get_status(self):
        s = MockSkill()
        st = s.get_status()
        assert st["name"] == "mock_skill"
        assert st["version"] == "0.1.0"
        assert st["status"] == "ok"
        assert "timestamp" in st

    def test_cleanup_default(self):
        s = MockSkill()
        s.cleanup()  # should not raise
        assert s.cleanup_called is True


# ── SkillRegistry tests ─────────────────────────────────────────────

class TestSkillRegistry:
    def _make_registry(self) -> SkillRegistry:
        return SkillRegistry()

    # ── register / unregister ────────────────────────────────────────

    def test_register_skill(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        assert reg.get_skill("mock_skill") is not None

    def test_unregister_skill(self):
        reg = self._make_registry()
        sk = MockSkill()
        reg.register(sk)
        reg.unregister("mock_skill")
        assert reg.get_skill("mock_skill") is None
        assert sk.cleanup_called is True

    def test_unregister_unknown_is_safe(self):
        reg = self._make_registry()
        reg.unregister("no_such_skill")  # no error

    def test_reregistration_replaces(self):
        reg = self._make_registry()
        old = MockSkill(version="1.0.0")
        new = MockSkill(version="2.0.0")
        reg.register(old)
        reg.register(new)
        assert reg.get_skill("mock_skill").version == "2.0.0"
        assert old.cleanup_called is True

    # ── get_skill ────────────────────────────────────────────────────

    def test_get_skill_existing(self):
        reg = self._make_registry()
        sk = MockSkill()
        reg.register(sk)
        assert reg.get_skill("mock_skill") is sk

    def test_get_skill_missing(self):
        reg = self._make_registry()
        assert reg.get_skill("nope") is None

    # ── list_skills ──────────────────────────────────────────────────

    def test_list_skills_empty(self):
        reg = self._make_registry()
        assert reg.list_skills() == []

    def test_list_skills_populated(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        listing = reg.list_skills()
        assert len(listing) == 1
        entry = listing[0]
        assert entry["name"] == "mock_skill"
        assert entry["version"] == "0.1.0"
        assert "commands" in entry
        assert len(entry["commands"]) == 2

    def test_list_skills_multiple(self):
        reg = self._make_registry()
        reg.register(MockSkill(name="a", description="A"))
        reg.register(MockSkill(name="b", description="B"))
        assert len(reg.list_skills()) == 2

    # ── get_all_commands ─────────────────────────────────────────────

    def test_get_all_commands_empty(self):
        reg = self._make_registry()
        assert reg.get_all_commands() == []

    def test_get_all_commands(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        cmds = reg.get_all_commands()
        assert len(cmds) == 2
        names = {c.name for c in cmds}
        assert names == {"greet", "echo"}

    # ── match_intent ─────────────────────────────────────────────────

    def test_match_intent_no_skills(self):
        reg = self._make_registry()
        assert reg.match_intent("hello") is None

    def test_match_intent_empty_input(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        assert reg.match_intent("") is None

    def test_match_intent_basic(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        m = reg.match_intent("hello")
        assert m is not None
        assert m.skill_name == "mock_skill"
        assert m.command_name == "greet"
        assert m.confidence > 0

    def test_match_intent_case_insensitive(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        m = reg.match_intent("HELLO")
        assert m is not None
        assert m.command_name == "greet"

    def test_match_intent_no_match(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        assert reg.match_intent("xyzzy foobar nothing") is None

    def test_match_intent_named_group_extraction(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        m = reg.match_intent("echo foobar")
        assert m is not None
        assert m.command_name == "echo"
        assert m.extracted_args.get("message") == "foobar"

    def test_match_intent_confidence_longer_match_wins(self):
        """Longer matched span should produce higher confidence."""
        reg = self._make_registry()
        reg.register(MockSkill())
        m_short = reg.match_intent("hello world how are you")
        m_long = reg.match_intent("hello")
        # "hello" matching all 5 chars out of 5 → 1.0
        # "hello" matching 5 chars out of 23 → lower
        assert m_long is not None
        assert m_short is not None
        assert m_long.confidence >= m_short.confidence

    def test_match_intent_confidence_capped_at_one(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        m = reg.match_intent("hello")
        assert m is not None
        assert m.confidence <= 1.0

    def test_match_intent_best_match_selected(self):
        """When two patterns match, the higher-confidence one wins."""
        reg = self._make_registry()
        reg.register(MockSkill())
        m = reg.match_intent("hi there")
        assert m is not None
        assert m.command_name == "greet"

    def test_match_intent_requires_confirmation_propagated(self):
        cmd = SkillCommand(
            name="danger",
            description="dangerous",
            patterns=[r"danger"],
            requires_confirmation=True,
        )
        sk = MockSkill(name="danger_skill", commands=[cmd])
        reg = self._make_registry()
        reg.register(sk)
        m = reg.match_intent("danger zone")
        assert m is not None
        assert m.requires_confirmation is True

    # ── execute ──────────────────────────────────────────────────────

    def test_execute_success(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        r = reg.execute("mock_skill", "greet", {})
        assert r.success is True
        assert r.message == "Hello!"

    def test_execute_unknown_skill(self):
        reg = self._make_registry()
        r = reg.execute("no_such_skill", "cmd", {})
        assert r.success is False
        assert "not registered" in r.message

    def test_execute_unknown_command(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        r = reg.execute("mock_skill", "nonexistent", {})
        assert r.success is False

    def test_execute_with_args(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        r = reg.execute("mock_skill", "echo", {"message": "test123"})
        assert r.success is True
        assert r.data["echo"] == "test123"

    def test_execute_handles_exception(self):
        reg = self._make_registry()
        reg.register(MockSkill())
        r = reg.execute("mock_skill", "fail", {})
        assert r.success is False
        assert "failed" in r.message.lower() or "boom" in r.message.lower()

    # ── invalid regex in pattern ─────────────────────────────────────

    def test_invalid_regex_pattern_skipped(self):
        cmd = SkillCommand(
            name="bad",
            description="bad regex",
            patterns=[r"(unclosed"],
        )
        sk = MockSkill(name="bad_regex_skill", commands=[cmd])
        reg = self._make_registry()
        reg.register(sk)
        # Skill is registered but the bad pattern is simply skipped
        assert reg.get_skill("bad_regex_skill") is not None
        # No patterns compiled, so no match
        assert reg.match_intent("unclosed") is None


# ── get_registry singleton tests ─────────────────────────────────────

class TestGetRegistry:
    def test_returns_instance(self):
        r = get_registry()
        assert isinstance(r, SkillRegistry)

    def test_returns_same_instance(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
