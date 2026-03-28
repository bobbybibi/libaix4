"""Unit tests for the knowledge base."""

import tempfile
from pathlib import Path


from knowledge_base import (
    KNOWLEDGE,
    get_answers,
    get_domain_labels,
    get_domains,
    get_questions,
    load_extra_knowledge,
)


class TestKnowledgeBase:
    def test_knowledge_not_empty(self):
        assert len(KNOWLEDGE) > 50

    def test_entry_format(self):
        for q, a, d in KNOWLEDGE:
            assert isinstance(q, str) and len(q) > 0
            assert isinstance(a, str) and len(a) > 0
            assert isinstance(d, str) and len(d) > 0

    def test_domains_present(self):
        domains = get_domains()
        assert "networking" in domains
        assert "internet" in domains
        assert "intranet" in domains
        assert "security" in domains
        assert "general" in domains

    def test_get_questions_length(self):
        assert len(get_questions()) == len(KNOWLEDGE)

    def test_get_answers_length(self):
        assert len(get_answers()) == len(KNOWLEDGE)

    def test_get_domain_labels_length(self):
        assert len(get_domain_labels()) == len(KNOWLEDGE)

    def test_load_extra_knowledge(self):
        data = [
            {"question": "What is WiFi?", "answer": "Wireless networking.", "domain": "networking"},
            {"question": "What is Bluetooth?", "answer": "Short-range wireless.", "domain": "networking"},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "extra.json"
            import json
            path.write_text(json.dumps(data))
            extra = load_extra_knowledge(path)
        assert len(extra) == 2
        assert extra[0][0] == "What is WiFi?"
        assert extra[0][2] == "networking"
