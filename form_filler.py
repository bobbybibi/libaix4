"""
form_filler.py — Automated form detection, analysis, and filling for libaix.

HTML form extraction, field classification, profile-based filling, prompt-
directed filling, CSRF token handling, submission, history, and templates.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────

FORM_CONFIG_PATH = Path("data/form_config.json")
FORM_HISTORY_PATH = Path("data/form_fill_history.json")
FORM_TEMPLATES_PATH = Path("data/form_templates.json")
FORM_PROFILES_PATH = Path("data/form_profiles.json")

# Patterns that hint at CSRF token fields
_CSRF_PATTERNS = re.compile(
    r"csrf|xsrf|_token|authenticity.token|__RequestVerificationToken",
    re.IGNORECASE,
)

# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class FormField:
    name: str
    type: str = "text"
    id: str = ""
    label: str = ""
    required: bool = False
    options: list[str] = field(default_factory=list)
    value: str = ""
    placeholder: str = ""
    pattern: str = ""


@dataclass
class FormTemplate:
    url: str
    method: str = "GET"
    action: str = ""
    fields: list[FormField] = field(default_factory=list)
    csrf_field: str = ""
    encoding: str = "application/x-www-form-urlencoded"


@dataclass
class FillProfile:
    name: str
    field_mappings: dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── HTML form parser ─────────────────────────────────────────────────


class _FormExtractor(HTMLParser):
    """Parse HTML to extract <form> elements and their child fields."""

    def __init__(self) -> None:
        super().__init__()
        self.forms: list[dict[str, Any]] = []
        self.labels: dict[str, str] = {}   # id → label text
        self._form: dict[str, Any] | None = None
        self._select: dict[str, Any] | None = None
        self._label_for: str = ""
        self._label_text: str = ""
        self._in_label = self._in_option = self._in_textarea = False
        self._ta_text: str = ""
        self._ta_attrs: dict[str, str] = {}

    @staticmethod
    def _ad(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {k: (v or "") for k, v in attrs}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        ad = self._ad(attrs)
        if tag == "form":
            self._form = {"action": ad.get("action", ""),
                          "method": ad.get("method", "GET").upper(),
                          "enctype": ad.get("enctype", "application/x-www-form-urlencoded"),
                          "fields": []}
        elif tag == "label":
            self._label_for = ad.get("for", "")
            self._label_text = ""
            self._in_label = True
        elif self._form is None:
            return
        elif tag == "input":
            self._form["fields"].append({
                "name": ad.get("name", ""), "type": ad.get("type", "text").lower(),
                "id": ad.get("id", ""), "required": "required" in ad,
                "value": ad.get("value", ""), "placeholder": ad.get("placeholder", ""),
                "pattern": ad.get("pattern", "")})
        elif tag == "textarea":
            self._in_textarea = True
            self._ta_text = ""
            self._ta_attrs = ad
        elif tag == "select":
            self._select = {"name": ad.get("name", ""), "id": ad.get("id", ""),
                            "required": "required" in ad, "options": []}
        elif tag == "option" and self._select is not None:
            self._in_option = True
            if "value" in ad:
                self._select["options"].append(ad["value"])

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "form" and self._form is not None:
            self.forms.append(self._form)
            self._form = None
        elif tag == "label" and self._in_label:
            self._in_label = False
            if self._label_for and self._label_text:
                self.labels[self._label_for] = self._label_text.strip()
        elif tag == "textarea" and self._in_textarea:
            self._in_textarea = False
            if self._form is not None:
                self._form["fields"].append({
                    "name": self._ta_attrs.get("name", ""), "type": "textarea",
                    "id": self._ta_attrs.get("id", ""), "required": "required" in self._ta_attrs,
                    "value": self._ta_text.strip(), "placeholder": self._ta_attrs.get("placeholder", ""),
                    "pattern": ""})
        elif tag == "select" and self._select is not None:
            if self._form is not None:
                self._form["fields"].append({
                    "name": self._select["name"], "type": "select",
                    "id": self._select["id"], "required": self._select["required"],
                    "value": "", "placeholder": "", "pattern": "",
                    "options": self._select["options"]})
            self._select = None
        elif tag == "option":
            self._in_option = False

    def handle_data(self, data: str) -> None:
        if self._in_label:
            self._label_text += data
        if self._in_textarea:
            self._ta_text += data


# ── Config management ────────────────────────────────────────────────


def _default_form_config() -> dict[str, Any]:
    return {
        "auto_csrf": True,
        "default_method": "POST",
        "timeout": 20,
        "max_history": 500,
        "use_anon_by_default": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_form_config() -> dict[str, Any]:
    """Load form-filler configuration from disk."""
    if FORM_CONFIG_PATH.exists():
        try:
            return json.loads(FORM_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return _default_form_config()


def save_form_config(config: dict[str, Any]) -> None:
    """Persist form-filler configuration to disk."""
    FORM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORM_CONFIG_PATH.write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ── Form extraction ──────────────────────────────────────────────────


def extract_forms(html: str, base_url: str = "") -> list[FormTemplate]:
    """Parse *html* and return :class:`FormTemplate` objects."""
    parser = _FormExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass

    templates: list[FormTemplate] = []
    for raw in parser.forms:
        action = raw.get("action", "")
        if base_url and action and not action.startswith(("http://", "https://")):
            action = urllib.parse.urljoin(base_url, action)
        elif not action:
            action = base_url

        fields: list[FormField] = []
        csrf_field = ""
        for rf in raw.get("fields", []):
            ff = FormField(
                name=rf.get("name", ""), type=rf.get("type", "text"),
                id=rf.get("id", ""), label=parser.labels.get(rf.get("id", ""), ""),
                required=rf.get("required", False), options=rf.get("options", []),
                value=rf.get("value", ""), placeholder=rf.get("placeholder", ""),
                pattern=rf.get("pattern", ""))
            fields.append(ff)
            combined = f"{ff.name} {ff.id}".lower()
            if ff.type == "hidden" and _CSRF_PATTERNS.search(combined):
                csrf_field = ff.name

        templates.append(FormTemplate(
            url=base_url, method=raw.get("method", "GET"), action=action,
            fields=fields, csrf_field=csrf_field,
            encoding=raw.get("enctype", "application/x-www-form-urlencoded")))
    return templates


# ── Field classification ─────────────────────────────────────────────

_FIELD_HEURISTICS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"e[\-_]?mail", re.I), "email"),
    (re.compile(r"(first|last|full|fname|lname)[_\-]?name", re.I), "name"),
    (re.compile(r"\bname\b", re.I), "name"),
    (re.compile(r"phone|tel(ephone)?|mobile|cell", re.I), "phone"),
    (re.compile(r"addr|street|city|zip|postal|state|province|country", re.I), "address"),
    (re.compile(r"pass(word)?", re.I), "password"),
    (re.compile(r"user(name)?|login|uid", re.I), "username"),
    (re.compile(r"company|org(anization)?|firm", re.I), "organization"),
    (re.compile(r"comment|message|note|body|content|desc(ription)?", re.I), "message"),
    (re.compile(r"subject|title|topic", re.I), "subject"),
    (re.compile(r"url|website|homepage|link", re.I), "url"),
    (re.compile(r"date|dob|birth", re.I), "date"),
    (re.compile(r"age", re.I), "age"),
    (re.compile(r"gender|sex", re.I), "gender"),
    (re.compile(r"card|cc|credit", re.I), "payment"),
    (re.compile(r"search|query|q\b", re.I), "search"),
]


def classify_field(field_obj: FormField) -> str:
    """Guess the semantic meaning of *field_obj* using heuristics.

    Returns a short label (``"email"``, ``"name"``, ``"phone"`` etc.) or the
    raw HTML ``type`` attribute when no heuristic matches.
    """
    # Fast-path for explicit HTML5 types
    _type_map = {"email": "email", "tel": "phone", "password": "password",
                 "url": "url", "number": "number"}
    if field_obj.type in _type_map:
        return _type_map[field_obj.type]
    if field_obj.type in ("date", "datetime-local", "month", "week", "time"):
        return "date"

    hints = " ".join([field_obj.name, field_obj.id, field_obj.placeholder, field_obj.label])
    if not hints.strip():
        return field_obj.type
    for pattern, semantic in _FIELD_HEURISTICS:
        if pattern.search(hints):
            return semantic
    return field_obj.type


# ── Prompt parsing ───────────────────────────────────────────────────

_KV_RE = re.compile(
    r"""
    (\w[\w\-]*)            # key (field name)
    \s*=\s*                # equals sign
    (?:                    # value: quoted or unquoted
        "([^"]*)"          # double-quoted
        |'([^']*)'         # single-quoted
        |(\S+)             # bare word
    )
    """,
    re.VERBOSE,
)


def parse_fill_prompt(prompt: str) -> dict[str, str]:
    """Parse ``"fill name=John email=john@ex.com"`` into a dict.

    Supports ``key=value``, ``key="value with spaces"``, and ``key='value'``.
    """
    return {m.group(1): (m.group(2) or m.group(3) or m.group(4) or "")
            for m in _KV_RE.finditer(prompt)}


# ── Form filling ─────────────────────────────────────────────────────


def fill_form(
    template: FormTemplate,
    values: dict[str, str],
    profile: FillProfile | None = None,
) -> dict[str, str]:
    """Fill a :class:`FormTemplate` with *values* and optional *profile* fallback.

    Hidden fields (CSRF tokens) keep their original values unless overridden.
    Returns ``{field_name: filled_value}`` ready for submission.
    """
    profile_map: dict[str, str] = dict(profile.field_mappings) if profile else {}
    filled: dict[str, str] = {}
    for fld in template.fields:
        if not fld.name:
            continue
        if fld.name in values:
            filled[fld.name] = values[fld.name]
        elif classify_field(fld) in profile_map:
            filled[fld.name] = profile_map[classify_field(fld)]
        elif fld.name in profile_map:
            filled[fld.name] = profile_map[fld.name]
        elif fld.value:
            filled[fld.name] = fld.value
        else:
            filled[fld.name] = ""
    return filled


# ── Form submission ──────────────────────────────────────────────────


def submit_form(
    template: FormTemplate,
    filled_values: dict[str, str],
    use_anon: bool = True,
) -> dict[str, Any]:
    """Submit a filled form via HTTP. Returns dict with status/text/url/error."""
    action = template.action or template.url
    if not action:
        return {"status": 0, "text": "", "url": "", "error": "No action URL"}

    method = template.method.upper()
    encoded_data = urllib.parse.urlencode(filled_values)
    _record_history(action, method, filled_values)

    # Anonymous path via anon_crawler
    if use_anon:
        try:
            from anon_crawler import anon_fetch  # type: ignore[import-untyped]
            if method == "GET":
                sep = "&" if "?" in action else "?"
                return anon_fetch(f"{action}{sep}{encoded_data}")
            return anon_fetch(action)
        except ImportError:
            pass

    # Direct urllib fallback
    timeout = load_form_config().get("timeout", 20)
    try:
        if method == "GET":
            sep = "&" if "?" in action else "?"
            req = urllib.request.Request(f"{action}{sep}{encoded_data}", method="GET")
        else:
            req = urllib.request.Request(
                action, data=encoded_data.encode("utf-8"), method="POST",
                headers={"Content-Type": template.encoding})
        req.add_header("User-Agent", "libaix-formfiller/1.0")

        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
            return {"status": resp.status, "text": body, "url": resp.url, "error": ""}
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
        except Exception:
            pass
        return {"status": exc.code, "text": body, "url": action, "error": str(exc.reason)}
    except Exception as exc:
        return {"status": 0, "text": "", "url": action, "error": str(exc)}


# ── Profile management ───────────────────────────────────────────────


def _load_profiles_raw() -> list[dict[str, Any]]:
    if FORM_PROFILES_PATH.exists():
        try:
            return json.loads(FORM_PROFILES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_profiles_raw(profiles: list[dict[str, Any]]) -> None:
    FORM_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORM_PROFILES_PATH.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def save_profile(profile: FillProfile) -> None:
    """Save or update a named fill profile."""
    profiles = [p for p in _load_profiles_raw() if p.get("name") != profile.name]
    profiles.append(asdict(profile))
    _save_profiles_raw(profiles)


def load_profiles() -> list[FillProfile]:
    """Load all saved fill profiles."""
    result: list[FillProfile] = []
    for e in _load_profiles_raw():
        try:
            result.append(FillProfile(name=e["name"], field_mappings=e.get("field_mappings", {}),
                                      created_at=e.get("created_at", "")))
        except (KeyError, TypeError):
            continue
    return result


def delete_profile(name: str) -> bool:
    """Delete a profile by name. Returns ``True`` if found and deleted."""
    profiles = _load_profiles_raw()
    filtered = [p for p in profiles if p.get("name") != name]
    if len(filtered) < len(profiles):
        _save_profiles_raw(filtered)
        return True
    return False


# ── Template management ──────────────────────────────────────────────


def save_form_template(template: FormTemplate, name: str) -> None:
    """Persist a form template under a friendly *name* for reuse."""
    templates = [t for t in _load_templates_raw() if t.get("template_name") != name]
    entry = asdict(template)
    entry["template_name"] = name
    entry["saved_at"] = datetime.now(timezone.utc).isoformat()
    templates.append(entry)
    _save_templates_raw(templates)


def load_form_templates() -> list[dict[str, Any]]:
    """Return all saved form templates as dicts."""
    return _load_templates_raw()


def _load_templates_raw() -> list[dict[str, Any]]:
    if FORM_TEMPLATES_PATH.exists():
        try:
            return json.loads(FORM_TEMPLATES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_templates_raw(templates: list[dict[str, Any]]) -> None:
    FORM_TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORM_TEMPLATES_PATH.write_text(
        json.dumps(templates, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ── Fill history ─────────────────────────────────────────────────────


def _record_history(url: str, method: str, values: dict[str, str]) -> None:
    """Append a fill event to the history log (capped by config max_history)."""
    max_history = load_form_config().get("max_history", 500)
    history = _load_history_raw()
    # Redact sensitive-looking values
    safe = {k: ("***REDACTED***" if re.search(r"pass|secret|token|credit|cvv|ssn", k, re.I) else v)
            for k, v in values.items()}
    history.append({"url": url, "method": method, "fields": safe,
                    "filled_at": datetime.now(timezone.utc).isoformat()})
    if len(history) > max_history:
        history = history[-max_history:]
    _save_history_raw(history)


def _load_history_raw() -> list[dict[str, Any]]:
    if FORM_HISTORY_PATH.exists():
        try:
            return json.loads(FORM_HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history_raw(history: list[dict[str, Any]]) -> None:
    FORM_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    FORM_HISTORY_PATH.write_text(
        json.dumps(history, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def get_fill_history(n: int = 50) -> list[dict[str, Any]]:
    """Return the last *n* fill-history entries (most recent last)."""
    history = _load_history_raw()
    return history[-n:]


# ── CSRF helpers ─────────────────────────────────────────────────────


def extract_csrf_token(html: str) -> str | None:
    """Extract a CSRF token value from raw HTML hidden inputs."""
    for form in extract_forms(html):
        for fld in form.fields:
            if fld.type == "hidden" and fld.value and _CSRF_PATTERNS.search(f"{fld.name} {fld.id}"):
                return fld.value
    return None


# ── Validation extraction ────────────────────────────────────────────


def extract_validation_rules(template: FormTemplate) -> dict[str, dict[str, Any]]:
    """Extract client-side validation rules from a form template."""
    rules: dict[str, dict[str, Any]] = {}
    for fld in template.fields:
        if not fld.name:
            continue
        entry: dict[str, Any] = {}
        if fld.required:
            entry["required"] = True
        if fld.pattern:
            entry["pattern"] = fld.pattern
        if fld.type in ("email", "url", "number", "tel", "date"):
            entry["type"] = fld.type
        if entry:
            rules[fld.name] = entry
    return rules


# ── Multi-step (wizard) form support ─────────────────────────────────


def detect_wizard_steps(html: str, base_url: str = "") -> list[FormTemplate]:
    """Detect multiple form steps in a single page (wizard forms)."""
    return extract_forms(html, base_url)


def submit_wizard(
    steps: list[FormTemplate],
    values: dict[str, str],
    profile: FillProfile | None = None,
    use_anon: bool = True,
) -> list[dict[str, Any]]:
    """Fill and submit each step of a wizard form in order."""
    results: list[dict[str, Any]] = []
    carry: dict[str, str] = dict(values)
    for step in steps:
        filled = fill_form(step, carry, profile)
        resp = submit_form(step, filled, use_anon=use_anon)
        results.append(resp)
        # Carry forward hidden values from response (e.g. updated CSRF)
        for nf in extract_forms(resp.get("text", ""), step.action or step.url):
            for fld in nf.fields:
                if fld.type == "hidden" and fld.value and fld.name:
                    carry[fld.name] = fld.value
    return results
