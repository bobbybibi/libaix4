"""Tests for file_processor.py — text extraction, domain classification, Q&A generation."""
from __future__ import annotations



from file_processor import (
    classify_domain,
    extract_text_from_file,
    extract_text_from_string,
    generate_qa_from_text,
    process_file,
    process_pasted_text,
    split_into_chunks,
    _clean_answer,
    _extract_key_terms,
    _strip_tags,
)


# ── classify_domain ──────────────────────────────────────────────────

class TestClassifyDomain:
    def test_networking_text(self):
        assert classify_domain("The TCP protocol uses IP addresses for routing") == "networking"

    def test_security_text(self):
        assert classify_domain("A firewall blocks unauthorized access using IDS rules") == "security"

    def test_wifi_text(self):
        assert classify_domain("802.11ax supports OFDMA and MU-MIMO for wifi") == "wifi"

    def test_unknown_text_returns_general(self):
        assert classify_domain("The quick brown fox jumps over the lazy dog") == "general"

    def test_empty_text_returns_general(self):
        assert classify_domain("") == "general"


# ── extract_text_from_file ───────────────────────────────────────────

class TestExtractText:
    def test_txt_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello world content", encoding="utf-8")
        result = extract_text_from_file(f)
        assert "Hello world content" in result

    def test_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\nfoo,bar\n", encoding="utf-8")
        result = extract_text_from_file(f)
        assert "foo" in result
        assert "bar" in result

    def test_html_file(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<p>Hello <b>world</b></p>", encoding="utf-8")
        result = extract_text_from_file(f)
        assert "Hello" in result
        assert "world" in result
        assert "<p>" not in result

    def test_md_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\nSome content", encoding="utf-8")
        result = extract_text_from_file(f)
        assert "Title" in result


class TestExtractTextFromString:
    def test_collapses_whitespace(self):
        result = extract_text_from_string("hello     world")
        assert result == "hello world"

    def test_collapses_newlines(self):
        result = extract_text_from_string("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_strips_edges(self):
        result = extract_text_from_string("  hello  ")
        assert result == "hello"


# ── strip_tags ───────────────────────────────────────────────────────

class TestStripTags:
    def test_removes_tags(self):
        assert "Hello" in _strip_tags("<b>Hello</b>")
        assert "<b>" not in _strip_tags("<b>Hello</b>")

    def test_preserves_text(self):
        assert _strip_tags("no tags here") == "no tags here"


# ── split_into_chunks ────────────────────────────────────────────────

class TestSplitIntoChunks:
    def test_splits_paragraphs(self):
        text = ("A" * 60) + "\n\n" + ("B" * 60)
        chunks = split_into_chunks(text)
        assert len(chunks) == 2

    def test_skips_short_paragraphs(self):
        text = "short\n\n" + ("A" * 60)
        chunks = split_into_chunks(text)
        assert len(chunks) == 1


# ── generate_qa_from_text ────────────────────────────────────────────

class TestGenerateQA:
    def test_definition_pattern(self):
        text = "TCP is a transport protocol that provides reliable ordered data delivery between applications over IP networks."
        entries = generate_qa_from_text(text)
        assert any("What is" in e["question"] for e in entries)

    def test_capability_pattern(self):
        text = "A firewall provides protection against unauthorized network access by filtering incoming and outgoing traffic based on rules."
        entries = generate_qa_from_text(text)
        assert len(entries) >= 1

    def test_deduplicates_questions(self):
        text = (
            "TCP is a transport protocol that ensures reliable data delivery.\n\n"
            "TCP is a transport protocol that ensures reliable data delivery."
        )
        entries = generate_qa_from_text(text)
        questions = [e["question"].lower() for e in entries]
        assert len(questions) == len(set(questions))

    def test_domain_hint_used(self):
        text = "OSPF is a routing protocol used for finding shortest paths in networks."
        entries = generate_qa_from_text(text, domain_hint="networking")
        for e in entries:
            assert e["domain"] == "networking"

    def test_empty_text(self):
        assert generate_qa_from_text("") == []


# ── _clean_answer ────────────────────────────────────────────────────

class TestCleanAnswer:
    def test_adds_period(self):
        assert _clean_answer("Hello world") == "Hello world."

    def test_keeps_existing_period(self):
        assert _clean_answer("Hello world.") == "Hello world."

    def test_strips_whitespace(self):
        assert _clean_answer("  Hello.  ") == "Hello."


# ── _extract_key_terms ───────────────────────────────────────────────

class TestExtractKeyTerms:
    def test_finds_acronyms(self):
        terms = _extract_key_terms("The TCP and UDP protocols are important.")
        assert any("TCP" in t for t in terms)

    def test_empty_text(self):
        assert _extract_key_terms("hello world") == []


# ── process_file / process_pasted_text ───────────────────────────────

class TestProcessFile:
    def test_returns_entries_and_preview(self, tmp_path):
        f = tmp_path / "test.txt"
        content = "TCP is a transport protocol that provides reliable ordered data delivery between applications over IP networks."
        f.write_text(content, encoding="utf-8")
        entries, preview = process_file(f)
        assert isinstance(entries, list)
        assert "TCP" in preview


class TestProcessPastedText:
    def test_returns_entries(self):
        text = "DNS is a system that translates domain names to IP addresses for internet communication purposes."
        entries = process_pasted_text(text)
        assert isinstance(entries, list)
