"""Tests for crawler de-duplication helpers in file_processor."""

from __future__ import annotations

import json

from file_processor import dedupe_new_entries, existing_qa_keys


def test_existing_qa_keys_empty_or_missing_dir(tmp_path):
    assert existing_qa_keys(tmp_path) == set()
    assert existing_qa_keys(tmp_path / "does_not_exist") == set()


def test_existing_qa_keys_reads_saved_pairs(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps([{"question": "Q1", "answer": "A1", "domain": "x"}]),
        encoding="utf-8",
    )
    keys = existing_qa_keys(tmp_path)
    assert ("Q1", "A1") in keys


def test_dedupe_filters_already_saved(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps([{"question": "Q1", "answer": "A1", "domain": "x"}]),
        encoding="utf-8",
    )
    new = dedupe_new_entries(
        [
            {"question": "Q1", "answer": "A1", "domain": "x"},  # already saved
            {"question": "Q2", "answer": "A2", "domain": "x"},  # genuinely new
        ],
        tmp_path,
    )
    assert [e["question"] for e in new] == ["Q2"]


def test_dedupe_drops_within_batch_duplicates(tmp_path):
    new = dedupe_new_entries(
        [
            {"question": "Q", "answer": "A", "domain": "x"},
            {"question": "Q", "answer": "A", "domain": "y"},  # same (q, a)
        ],
        tmp_path,
    )
    assert len(new) == 1


def test_dedupe_tolerates_corrupt_files(tmp_path):
    (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
    new = dedupe_new_entries([{"question": "Q", "answer": "A", "domain": "x"}], tmp_path)
    assert len(new) == 1


def test_dedupe_all_new_when_dir_empty(tmp_path):
    entries = [
        {"question": "Q1", "answer": "A1", "domain": "x"},
        {"question": "Q2", "answer": "A2", "domain": "x"},
    ]
    assert dedupe_new_entries(entries, tmp_path) == entries
