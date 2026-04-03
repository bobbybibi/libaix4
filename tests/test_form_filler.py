"""Tests for the form_filler module."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import form_filler  # noqa: E402
from form_filler import (
    FillProfile,
    FormField,
    FormTemplate,
    classify_field,
    extract_csrf_token,
    extract_forms,
    extract_validation_rules,
    fill_form,
    get_fill_history,
    load_form_templates,
    load_profiles,
    parse_fill_prompt,
    save_form_template,
    save_profile,
)


# ── Helpers ──────────────────────────────────────────────────────────

SIMPLE_HTML = """
<html><body>
<form action="/submit" method="POST">
  <label for="name_field">Name</label>
  <input type="text" name="name" id="name_field" required placeholder="Your name">
  <input type="email" name="email" id="email_field" placeholder="you@example.com">
  <input type="hidden" name="csrf_token" id="csrf_token" value="abc123">
  <textarea name="message" id="msg" placeholder="Your message"></textarea>
  <select name="country" id="country_field">
    <option value="us">US</option>
    <option value="uk">UK</option>
  </select>
  <input type="submit" value="Send">
</form>
</body></html>
"""

MULTI_FORM_HTML = """
<html><body>
<form action="/login" method="POST">
  <input type="text" name="username" required>
  <input type="password" name="password" required>
</form>
<form action="/search" method="GET">
  <input type="text" name="q" placeholder="Search…">
</form>
</body></html>
"""


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Redirect form-filler paths to tmp_path."""
    monkeypatch.setattr(form_filler, "FORM_CONFIG_PATH", tmp_path / "form_config.json")
    monkeypatch.setattr(form_filler, "FORM_HISTORY_PATH", tmp_path / "form_history.json")
    monkeypatch.setattr(form_filler, "FORM_TEMPLATES_PATH", tmp_path / "form_templates.json")
    monkeypatch.setattr(form_filler, "FORM_PROFILES_PATH", tmp_path / "form_profiles.json")


# ── extract_forms tests ──────────────────────────────────────────────


class TestExtractForms:
    def test_extract_single_form(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        assert len(forms) == 1

    def test_form_action_resolved(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        assert forms[0].action == "https://example.com/submit"

    def test_form_method(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        assert forms[0].method == "POST"

    def test_form_has_fields(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        field_names = [f.name for f in forms[0].fields]
        assert "name" in field_names
        assert "email" in field_names
        assert "csrf_token" in field_names
        assert "message" in field_names

    def test_textarea_detected(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        types = {f.name: f.type for f in forms[0].fields}
        assert types.get("message") == "textarea"

    def test_select_detected(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        sel = [f for f in forms[0].fields if f.name == "country"]
        assert len(sel) == 1
        assert sel[0].type == "select"
        assert "us" in sel[0].options

    def test_required_field(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        name_field = [f for f in forms[0].fields if f.name == "name"][0]
        assert name_field.required is True

    def test_csrf_field_detected(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        assert forms[0].csrf_field == "csrf_token"

    def test_extract_multiple_forms(self):
        forms = extract_forms(MULTI_FORM_HTML, "https://example.com")
        assert len(forms) == 2

    def test_empty_html_returns_empty(self):
        forms = extract_forms("", "")
        assert forms == []


# ── classify_field tests ─────────────────────────────────────────────


class TestClassifyField:
    def test_email_by_type(self):
        f = FormField(name="user_mail", type="email")
        assert classify_field(f) == "email"

    def test_email_by_name(self):
        f = FormField(name="email_address", type="text")
        assert classify_field(f) == "email"

    def test_phone_by_type(self):
        f = FormField(name="phone_num", type="tel")
        assert classify_field(f) == "phone"

    def test_password_by_type(self):
        f = FormField(name="pass", type="password")
        assert classify_field(f) == "password"

    def test_name_by_name(self):
        f = FormField(name="first_name", type="text")
        assert classify_field(f) == "name"

    def test_address_by_name(self):
        f = FormField(name="street_address", type="text")
        assert classify_field(f) == "address"

    def test_date_by_type(self):
        f = FormField(name="start", type="date")
        assert classify_field(f) == "date"

    def test_fallback_to_type(self):
        f = FormField(name="x", type="text")
        assert classify_field(f) == "text"


# ── parse_fill_prompt tests ──────────────────────────────────────────


class TestParseFillPrompt:
    def test_simple_kv(self):
        result = parse_fill_prompt('name=John email=john@test.com')
        assert result["name"] == "John"
        assert result["email"] == "john@test.com"

    def test_quoted_values(self):
        result = parse_fill_prompt('name="John Doe" email=j@x.com')
        assert result["name"] == "John Doe"

    def test_single_quoted(self):
        result = parse_fill_prompt("name='Jane Doe'")
        assert result["name"] == "Jane Doe"

    def test_empty_prompt(self):
        result = parse_fill_prompt("")
        assert result == {}


# ── FillProfile dataclass ───────────────────────────────────────────


class TestFillProfile:
    def test_create_profile(self):
        p = FillProfile(name="default", field_mappings={"email": "me@test.com"})
        assert p.name == "default"
        assert p.field_mappings["email"] == "me@test.com"
        assert p.created_at  # auto-populated


# ── Profile persistence ─────────────────────────────────────────────


class TestProfilePersistence:
    def test_save_and_load_profile(self, tmp_path):
        p = FillProfile(name="work", field_mappings={"email": "work@co.com"})
        save_profile(p)
        profiles = load_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "work"
        assert profiles[0].field_mappings["email"] == "work@co.com"

    def test_overwrite_profile_by_name(self, tmp_path):
        save_profile(FillProfile(name="x", field_mappings={"a": "1"}))
        save_profile(FillProfile(name="x", field_mappings={"a": "2"}))
        profiles = load_profiles()
        assert len(profiles) == 1
        assert profiles[0].field_mappings["a"] == "2"


# ── Template persistence ────────────────────────────────────────────


class TestTemplatePersistence:
    def test_save_and_load_template(self, tmp_path):
        tmpl = FormTemplate(url="https://example.com", method="POST", action="/submit")
        save_form_template(tmpl, "my_form")
        templates = load_form_templates()
        assert len(templates) == 1
        assert templates[0]["template_name"] == "my_form"

    def test_load_templates_empty(self, tmp_path):
        assert load_form_templates() == []


# ── Fill history ─────────────────────────────────────────────────────


class TestFillHistory:
    def test_history_empty_by_default(self, tmp_path):
        assert get_fill_history() == []


# ── Validation rules ────────────────────────────────────────────────


class TestValidationRules:
    def test_extract_validation_rules(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        rules = extract_validation_rules(forms[0])
        assert "name" in rules
        assert rules["name"]["required"] is True

    def test_email_type_captured(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        rules = extract_validation_rules(forms[0])
        assert rules.get("email", {}).get("type") == "email"


# ── CSRF token extraction ───────────────────────────────────────────


class TestCsrfDetection:
    def test_extract_csrf_token_from_html(self):
        token = extract_csrf_token(SIMPLE_HTML)
        assert token == "abc123"

    def test_no_csrf_in_plain_form(self):
        html = '<form><input type="text" name="q"></form>'
        assert extract_csrf_token(html) is None


# ── fill_form ────────────────────────────────────────────────────────


class TestFillForm:
    def test_fill_with_values(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        filled = fill_form(forms[0], {"name": "Alice", "email": "a@b.c"})
        assert filled["name"] == "Alice"
        assert filled["email"] == "a@b.c"
        # Hidden CSRF token keeps original value
        assert filled["csrf_token"] == "abc123"

    def test_fill_with_profile_fallback(self):
        forms = extract_forms(SIMPLE_HTML, "https://example.com")
        profile = FillProfile(name="p", field_mappings={"email": "profile@x.com"})
        filled = fill_form(forms[0], {}, profile=profile)
        assert filled["email"] == "profile@x.com"
