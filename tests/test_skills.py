"""Tests for all skills in skills/."""
from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from skill_registry import Skill, SkillCommand, SkillResult


# ── Shared helpers ───────────────────────────────────────────────────

def _assert_valid_skill(skill: Skill) -> None:
    """Assert that a skill satisfies the Skill ABC contract."""
    assert isinstance(skill.name, str) and len(skill.name) > 0
    assert isinstance(skill.description, str) and len(skill.description) > 0
    assert isinstance(skill.version, str)
    assert isinstance(skill.category, str)

    cmds = skill.get_commands()
    assert isinstance(cmds, list)
    assert len(cmds) > 0

    for cmd in cmds:
        assert isinstance(cmd, SkillCommand)
        assert isinstance(cmd.name, str) and len(cmd.name) > 0
        assert isinstance(cmd.patterns, list) and len(cmd.patterns) > 0
        # All patterns must compile
        for pat in cmd.patterns:
            compiled = re.compile(pat, re.IGNORECASE)
            assert compiled is not None


def _assert_unknown_command_fails(skill: Skill) -> None:
    """Executing an unknown command should return a failing SkillResult."""
    r = skill.execute("__nonexistent_command__", {})
    assert isinstance(r, SkillResult)
    assert r.success is False


# ── NetworkScannerSkill ──────────────────────────────────────────────

class TestNetworkScannerSkill:
    @pytest.fixture
    def skill(self):
        from skills.network_scanner import NetworkScannerSkill
        return NetworkScannerSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "network_scanner"
        assert skill.category == "security"
        assert skill.version == "1.0.0"

    def test_get_commands_non_empty(self, skill):
        assert len(skill.get_commands()) >= 4

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_scan_ports_localhost(self, skill):
        r = skill.execute("scan_ports", {"target": "127.0.0.1", "ports": [1]})
        assert isinstance(r, SkillResult)
        assert r.success is True
        assert "scanned_count" in r.data or "Scanned" in r.message

    def test_scan_network(self, skill):
        r = skill.execute("scan_network", {})
        assert isinstance(r, SkillResult)
        # May fail if arp is missing — that's OK
        if not r.success:
            assert "Error" in r.message or "error" in r.message.lower()

    def test_check_connections(self, skill):
        r = skill.execute("check_connections", {})
        assert isinstance(r, SkillResult)

    def test_detect_suspicious(self, skill):
        r = skill.execute("detect_suspicious", {})
        assert isinstance(r, SkillResult)

    def test_scan_ports_bad_host(self, skill):
        r = skill.execute("scan_ports", {"target": "this.host.does.not.exist.example.invalid", "ports": [80]})
        assert isinstance(r, SkillResult)
        assert r.success is False


# ── FileMonitorSkill ─────────────────────────────────────────────────

class TestFileMonitorSkill:
    @pytest.fixture
    def skill(self):
        from skills.file_monitor import FileMonitorSkill
        return FileMonitorSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "file_monitor"
        assert skill.category == "security"
        assert skill.version == "1.0.0"

    def test_get_commands(self, skill):
        assert len(skill.get_commands()) >= 4

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_system_info(self, skill):
        r = skill.execute("system_info", {})
        assert r.success is True
        assert "os" in r.data
        assert "python_version" in r.data

    def test_watch_directory(self, skill, tmp_path):
        r = skill.execute("watch_directory", {"path": str(tmp_path)})
        assert r.success is True
        assert r.data["path"] == str(tmp_path)

    def test_watch_directory_nonexistent(self, skill):
        r = skill.execute("watch_directory", {"path": "/nonexistent/path/xyz"})
        assert r.success is False

    def test_check_directory_changes_no_snapshot(self, skill, tmp_path):
        r = skill.execute("check_directory_changes", {"path": str(tmp_path)})
        assert r.success is False
        assert "snapshot" in r.message.lower() or "watch" in r.message.lower()

    def test_check_directory_changes_after_watch(self, skill, tmp_path):
        skill.execute("watch_directory", {"path": str(tmp_path)})
        r = skill.execute("check_directory_changes", {"path": str(tmp_path)})
        assert r.success is True
        assert r.data["has_changes"] is False

    def test_check_directory_changes_detects_new_file(self, skill, tmp_path):
        skill.execute("watch_directory", {"path": str(tmp_path)})
        # Add a file
        (tmp_path / "new_file.txt").write_text("hello")
        r = skill.execute("check_directory_changes", {"path": str(tmp_path)})
        assert r.success is True
        assert r.data["has_changes"] is True
        assert len(r.data["added"]) == 1

    def test_watch_directory_no_path(self, skill):
        r = skill.execute("watch_directory", {})
        assert r.success is False

    def test_check_processes(self, skill):
        r = skill.execute("check_processes", {})
        assert isinstance(r, SkillResult)
        # May succeed or fail depending on OS — just check return type


# ── MalwareScannerSkill ──────────────────────────────────────────────

class TestMalwareScannerSkill:
    @pytest.fixture
    def skill(self):
        from skills.malware_scanner import MalwareScannerSkill
        return MalwareScannerSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "malware_scanner"
        assert skill.category == "security"

    def test_get_commands(self, skill):
        assert len(skill.get_commands()) >= 4

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_scan_clean_file(self, skill, tmp_path):
        test_file = tmp_path / "clean.txt"
        test_file.write_text("This is a perfectly safe file.")
        r = skill.execute("scan_file", {"path": str(test_file)})
        assert r.success is True
        assert r.data["infected"] is False

    def test_scan_nonexistent_file(self, skill):
        r = skill.execute("scan_file", {"path": "/nonexistent/file.txt"})
        assert r.success is False

    def test_scan_file_no_path(self, skill):
        r = skill.execute("scan_file", {})
        assert r.success is False

    def test_add_signature_and_detect(self, skill, tmp_path):
        test_file = tmp_path / "evil.txt"
        content = "unique evil content for test"
        test_file.write_text(content)

        # Hash the file
        import hashlib
        digest = hashlib.sha256(content.encode()).hexdigest()

        # Add signature
        skill.add_signature(digest, "TestMalware", "high", "Test malware")

        # Now scan should detect it
        r = skill.execute("scan_file", {"path": str(test_file)})
        assert r.success is True
        assert r.data["infected"] is True
        assert r.data["threat"]["name"] == "TestMalware"

    def test_update_signatures(self, skill):
        r = skill.execute("update_signatures", {})
        assert r.success is True
        assert r.data["signature_count"] >= 1

    def test_scan_directory(self, skill, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        r = skill.execute("scan_directory", {"path": str(tmp_path)})
        assert r.success is True
        assert r.data["scanned"] >= 2


# ── FirewallManagerSkill ─────────────────────────────────────────────

class TestFirewallManagerSkill:
    @pytest.fixture
    def skill(self):
        from skills.firewall_manager import FirewallManagerSkill
        return FirewallManagerSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "firewall_manager"
        assert skill.category == "security"

    def test_get_commands(self, skill):
        assert len(skill.get_commands()) >= 4

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_firewall_status(self, skill):
        r = skill.execute("firewall_status", {})
        assert isinstance(r, SkillResult)
        # May fail (no iptables/ufw) — that's acceptable

    def test_list_rules(self, skill):
        r = skill.execute("list_rules", {})
        assert isinstance(r, SkillResult)
        # May fail due to permissions or missing tools

    def test_block_ip_invalid(self, skill):
        r = skill.execute("block_ip", {"ip": "not_an_ip"})
        assert r.success is False
        assert "not a valid" in r.message or "Error" in r.message

    def test_unblock_ip_invalid(self, skill):
        r = skill.execute("unblock_ip", {"ip": "invalid"})
        assert r.success is False


# ── VPNManagerSkill ──────────────────────────────────────────────────

class TestVPNManagerSkill:
    @pytest.fixture
    def skill(self):
        from skills.vpn_manager import VPNManagerSkill
        return VPNManagerSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "vpn_manager"
        assert skill.category == "network"

    def test_get_commands(self, skill):
        assert len(skill.get_commands()) >= 4

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_vpn_status(self, skill):
        r = skill.execute("vpn_status", {})
        assert isinstance(r, SkillResult)
        # May fail or succeed depending on system — just check type

    def test_vpn_config(self, skill):
        r = skill.execute("vpn_config", {})
        assert isinstance(r, SkillResult)
        # Config should always succeed
        assert r.success is True
        assert "config" in r.data


# ── DNSFilterSkill ───────────────────────────────────────────────────

class TestDNSFilterSkill:
    @pytest.fixture
    def skill(self):
        from skills.dns_filter import DNSFilterSkill
        return DNSFilterSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "dns_filter"
        assert skill.category == "network"

    def test_get_commands(self, skill):
        assert len(skill.get_commands()) >= 4

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_list_blocked(self, skill):
        r = skill.execute("list_blocked", {})
        assert r.success is True
        assert "total_blocked" in r.data

    def test_dns_status(self, skill):
        r = skill.execute("dns_status", {})
        assert r.success is True
        assert r.data["enabled"] is True
        assert "total_blocked" in r.data

    def test_block_domain(self, skill):
        r = skill.execute("block_domain", {"domain": "test-block-xyz.example.com"})
        assert r.success is True
        assert r.data["domain"] == "test-block-xyz.example.com"

    def test_block_invalid_domain(self, skill):
        r = skill.execute("block_domain", {"domain": "not a domain!!"})
        assert r.success is False

    def test_unblock_domain(self, skill):
        # Block first, then unblock
        skill.execute("block_domain", {"domain": "test-unblock-xyz.example.com"})
        r = skill.execute("unblock_domain", {"domain": "test-unblock-xyz.example.com"})
        assert r.success is True

    def test_unblock_unknown_domain(self, skill):
        r = skill.execute("unblock_domain", {"domain": "never-blocked-xyz123.example.com"})
        assert r.success is False

    def test_block_no_domain(self, skill):
        r = skill.execute("block_domain", {})
        assert r.success is False

    def test_block_already_blocked(self, skill):
        skill.execute("block_domain", {"domain": "dup-test.example.com"})
        r = skill.execute("block_domain", {"domain": "dup-test.example.com"})
        assert r.success is True
        assert r.data.get("already_blocked") is True


# ── SmartHomeSkill ───────────────────────────────────────────────────

class TestSmartHomeSkill:
    @pytest.fixture
    def skill(self):
        from skills.smart_home import SmartHomeSkill
        return SmartHomeSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "smart_home"
        assert skill.category == "device"

    def test_get_commands(self, skill):
        assert len(skill.get_commands()) >= 5

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_list_devices(self, skill):
        r = skill.execute("list_devices", {})
        assert r.success is True
        assert "devices" in r.data
        assert "count" in r.data

    def test_add_device(self, skill):
        r = skill.execute("add_device", {"device": "test_toaster_xyz"})
        assert isinstance(r, SkillResult)
        # Either succeeds or says already registered
        if r.success:
            assert r.data["name"] == "test_toaster_xyz"

    def test_add_device_no_name(self, skill):
        r = skill.execute("add_device", {})
        assert r.success is False

    def test_device_status_known(self, skill):
        # The skill comes with example devices; "TV" is one of them
        r = skill.execute("device_status", {"device": "TV"})
        assert isinstance(r, SkillResult)
        # Should find TV via partial match
        if r.success:
            assert "state" in r.data

    def test_device_status_unknown(self, skill):
        r = skill.execute("device_status", {"device": "nonexistent_gadget_xyz"})
        assert r.success is False

    def test_device_on_unknown(self, skill):
        r = skill.execute("device_on", {"device": "nonexistent_gadget_xyz"})
        assert r.success is False


# ── WebAutomationSkill ───────────────────────────────────────────────

class TestWebAutomationSkill:
    @pytest.fixture
    def skill(self):
        from skills.web_automation import WebAutomationSkill
        return WebAutomationSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "web_automation"
        assert skill.category == "automation"

    def test_get_commands(self, skill):
        assert len(skill.get_commands()) >= 4

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_fetch_page_no_url(self, skill):
        r = skill.execute("fetch_page", {})
        assert r.success is False

    def test_fetch_page_invalid_url(self, skill):
        r = skill.execute("fetch_page", {"url": "http://this.host.does.not.exist.example.invalid"})
        assert isinstance(r, SkillResult)
        assert r.success is False

    def test_monitor_page_no_url(self, skill):
        r = skill.execute("monitor_page", {})
        assert r.success is False

    def test_monitor_page_mock(self, skill):
        """Test monitor_page with a mocked HTTP response."""
        with patch.object(type(skill), '_safe_fetch', return_value=("<html>Hello</html>", 200)):
            r = skill.execute("monitor_page", {"url": "http://example.com"})
        assert r.success is True
        assert r.data["first_check"] is True

    def test_monitor_page_change_detection(self, skill):
        """Second call should detect no change with same content."""
        with patch.object(type(skill), '_safe_fetch', return_value=("<html>Hello</html>", 200)):
            skill.execute("monitor_page", {"url": "http://test-change.example.com"})
            r = skill.execute("monitor_page", {"url": "http://test-change.example.com"})
        assert r.success is True
        assert r.data["changed"] is False

    def test_normalise_url_adds_scheme(self, skill):
        assert skill._normalise_url("example.com") == "https://example.com"
        assert skill._normalise_url("http://x.com") == "http://x.com"
        assert skill._normalise_url("") == ""

    def test_search_web_no_query(self, skill):
        r = skill.execute("search_web", {})
        assert r.success is False


# ── ResearchAgentSkill ───────────────────────────────────────────────

class TestResearchAgentSkill:
    @pytest.fixture
    def skill(self):
        from skills.research_agent import ResearchAgentSkill
        return ResearchAgentSkill()

    def test_instantiation(self, skill):
        _assert_valid_skill(skill)

    def test_properties(self, skill):
        assert skill.name == "research_agent"
        assert skill.category == "automation"

    def test_get_commands(self, skill):
        assert len(skill.get_commands()) >= 4

    def test_patterns_compile(self, skill):
        _assert_valid_skill(skill)

    def test_unknown_command(self, skill):
        _assert_unknown_command_fails(skill)

    def test_research_topic_no_topic(self, skill):
        r = skill.execute("research_topic", {})
        assert r.success is False

    def test_research_topic_mock(self, skill):
        """Test with mocked Wikipedia response."""
        fake_response = {
            "type": "standard",
            "title": "Python",
            "extract": "Python is a programming language.",
            "description": "Programming language",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}},
            "thumbnail": {"source": ""},
        }
        import json
        with patch.object(type(skill), '_fetch_url', return_value=json.dumps(fake_response)):
            r = skill.execute("research_topic", {"topic": "Python"})
        assert r.success is True
        assert r.data["topic"] == "Python"
        assert "Python" in r.data["summary"]

    def test_research_topic_caching(self, skill):
        """Second call with same topic should use cache."""
        import json
        fake_response = {
            "type": "standard",
            "title": "Cache Test",
            "extract": "Cached content.",
            "description": "",
            "content_urls": {"desktop": {"page": ""}},
            "thumbnail": {},
        }
        with patch.object(type(skill), '_fetch_url', return_value=json.dumps(fake_response)):
            skill.execute("research_topic", {"topic": "cache_test_xyz"})

        # Second call — _fetch_url should NOT be called (cache hit)
        with patch.object(type(skill), '_fetch_url', side_effect=AssertionError("should not be called")):
            r2 = skill.execute("research_topic", {"topic": "cache_test_xyz"})

        assert r2.success is True
        assert "cached" in r2.message.lower()

    def test_fact_check_no_claim(self, skill):
        r = skill.execute("fact_check", {})
        assert r.success is False

    def test_fact_check_with_claim(self, skill):
        """fact_check should return results even if Wikipedia is unreachable."""
        import urllib.error
        with patch.object(type(skill), '_fetch_url', side_effect=urllib.error.URLError("no network")):
            r = skill.execute("fact_check", {"claim": "The earth is round"})
        assert r.success is True
        assert "disclaimer" in r.data

    def test_compare_topics_no_args(self, skill):
        r = skill.execute("compare_topics", {})
        assert r.success is False

    def test_summarize_url_no_url(self, skill):
        r = skill.execute("summarize_url", {})
        assert r.success is False


# ── Cross-cutting: all skills get_status ─────────────────────────────

class TestAllSkillsGetStatus:
    SKILL_CLASSES = []

    @pytest.fixture(autouse=True)
    def _load_classes(self):
        from skills.network_scanner import NetworkScannerSkill
        from skills.file_monitor import FileMonitorSkill
        from skills.malware_scanner import MalwareScannerSkill
        from skills.firewall_manager import FirewallManagerSkill
        from skills.vpn_manager import VPNManagerSkill
        from skills.dns_filter import DNSFilterSkill
        from skills.smart_home import SmartHomeSkill
        from skills.web_automation import WebAutomationSkill
        from skills.research_agent import ResearchAgentSkill

        self.SKILL_CLASSES = [
            NetworkScannerSkill,
            FileMonitorSkill,
            MalwareScannerSkill,
            FirewallManagerSkill,
            VPNManagerSkill,
            DNSFilterSkill,
            SmartHomeSkill,
            WebAutomationSkill,
            ResearchAgentSkill,
        ]

    def test_all_skills_have_valid_status(self):
        for cls in self.SKILL_CLASSES:
            skill = cls()
            st = skill.get_status()
            assert st["status"] == "ok", f"{cls.__name__}.get_status() did not return 'ok'"
            assert "name" in st
            assert "version" in st
