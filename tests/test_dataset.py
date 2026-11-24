"""Tests for golden-dataset validity."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
DOCS_DIR = ROOT / "data" / "sample_docs"

REQUIRED_FIELDS = {"id", "question", "expected_answer", "context", "source", "page"}


def load_golden() -> list[dict]:
    with GOLDEN.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_golden_size_in_range():
    rows = load_golden()
    assert 30 <= len(rows) <= 50, f"golden set has {len(rows)} items, expected 30-50"


def test_golden_ids_unique():
    rows = load_golden()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids in golden set"


def test_golden_required_fields():
    for row in load_golden():
        assert REQUIRED_FIELDS <= set(row), f"{row.get('id')}: missing fields"
        assert row["question"].strip()
        assert row["expected_answer"].strip()
        assert row["context"].strip()
        assert isinstance(row["page"], int)


def test_golden_contexts_verbatim_in_source_docs():
    """Every context must appear word-for-word in its source document."""
    docs = {p.name: p.read_text(encoding="utf-8") for p in DOCS_DIR.glob("*.md")}
    for row in load_golden():
        assert row["source"] in docs, f"{row['id']}: unknown source {row['source']}"
        assert row["context"] in docs[row["source"]], (
            f"{row['id']}: context not found verbatim in {row['source']}"
        )
