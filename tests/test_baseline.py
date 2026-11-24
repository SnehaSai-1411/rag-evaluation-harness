"""Tests for the offline baseline RAG (no model, no key)."""

import pytest

from eval_harness.baseline_rag import BaselineRAG, chunk_text, load_docs
from eval_harness.metrics import ABSTAIN_TEXT

DOCS_DIR = "data/sample_docs"


def test_load_docs():
    docs = load_docs(DOCS_DIR)
    assert len(docs) == 4
    assert all(d["text"].strip() for d in docs)


def test_chunk_text_respects_chunk_size():
    text = " ".join(["word"] * 500)
    for size in (100, 400, 800):
        chunks = chunk_text(text, size, overlap=20)
        assert chunks
        assert all(len(c) <= size for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", 400) == []


def test_chunk_text_invalid_args():
    with pytest.raises(ValueError):
        chunk_text("hello", 0)
    with pytest.raises(ValueError):
        chunk_text("hello", 100, overlap=100)


def test_retrieval_returns_top_k():
    rag = BaselineRAG(DOCS_DIR, chunk_size=800, top_k=3)
    results = rag.retrieve("What is chunking?")
    assert len(results) == 3
    assert all("text" in r and "source" in r and "score" in r for r in results)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_finds_relevant_doc():
    rag = BaselineRAG(DOCS_DIR, chunk_size=800)
    results = rag.retrieve("What is cosine similarity?")
    assert results[0]["source"] == "embeddings.md"


def test_answer_is_extractive():
    rag = BaselineRAG(DOCS_DIR, chunk_size=800)
    answer, contexts = rag.answer("What is chunk overlap?")
    assert answer != ABSTAIN_TEXT
    assert "overlap" in answer.lower()
    # Extractive: the answer's sentences come from the retrieved contexts.
    assert any(answer.split(".")[0].strip() in ctx for ctx in contexts)


def test_answer_abstains_on_unrelated_query():
    rag = BaselineRAG(DOCS_DIR, chunk_size=800)
    answer, _ = rag.answer("zebra quasar snorkel flibberty")
    assert answer == ABSTAIN_TEXT


def test_chunk_size_changes_chunk_count():
    small = BaselineRAG(DOCS_DIR, chunk_size=400)
    large = BaselineRAG(DOCS_DIR, chunk_size=800)
    assert len(small.chunks) > len(large.chunks)


def test_fake_embedder_interface_accepted():
    class FakeEmbedder:
        def embed(self, texts):
            return [[0.0] * 4 for _ in texts]

    rag = BaselineRAG(DOCS_DIR, chunk_size=800, embedder=FakeEmbedder())
    answer, _ = rag.answer("What is chunking?")
    assert answer != ABSTAIN_TEXT
