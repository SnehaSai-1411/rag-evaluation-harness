"""Tests for the deterministic metrics (known pass/fail cases)."""

from eval_harness.metrics import (
    ABSTAIN_TEXT,
    answer_relevance,
    context_precision,
    context_recall,
    evaluate_one,
    faithfulness,
)

CTX = [
    "Chunking splits long documents into smaller pieces so a retrieval system can return the most relevant passage.",
    "Overlap repeats the last characters of one chunk at the start of the next chunk.",
]


def test_faithfulness_fully_supported():
    answer = "Chunking splits long documents into smaller pieces."
    assert faithfulness(answer, CTX) == 1.0


def test_faithfulness_half_supported():
    # One sentence supported, one invented -> 0.5.
    answer = (
        "Chunking splits long documents into smaller pieces. "
        "Quantum entanglement powers the flux capacitor."
    )
    assert faithfulness(answer, CTX) == 0.5


def test_faithfulness_unsupported():
    answer = "Quantum entanglement powers the flux capacitor."
    assert faithfulness(answer, CTX) == 0.0


def test_faithfulness_abstention_is_faithful():
    # An abstention makes no claims, so it cannot be unfaithful.
    assert faithfulness(ABSTAIN_TEXT, CTX) == 1.0


def test_faithfulness_empty_answer():
    assert faithfulness("", CTX) == 0.0


def test_answer_relevance_high():
    question = "What is chunk overlap?"
    answer = "Overlap repeats the last characters of one chunk at the start of the next chunk."
    # question content tokens: {chunk, overlap}; both appear in the answer.
    assert answer_relevance(question, answer) == 1.0


def test_answer_relevance_zero():
    question = "What is chunk overlap?"
    answer = "Embeddings are vectors that represent meaning."
    assert answer_relevance(question, answer) == 0.0


def test_answer_relevance_abstention_is_zero():
    assert answer_relevance("What is chunking?", ABSTAIN_TEXT) == 0.0


def test_context_precision_all_relevant():
    # "chunking", "long", "documents" all appear in CTX[0] (>= 2 shared).
    question = "How does chunking split long documents?"
    assert context_precision(question, [CTX[0]]) == 1.0


def test_context_precision_none_relevant():
    question = "What is chunk overlap?"
    assert context_precision(question, ["Quantum entanglement powers the flux capacitor."]) == 0.0


def test_context_precision_empty():
    assert context_precision("What is chunking?", []) == 0.0


def test_context_recall_full():
    expected = "Chunking splits long documents into smaller pieces."
    assert context_recall(expected, CTX) == 1.0


def test_context_recall_none():
    expected = "Quantum entanglement powers the flux capacitor."
    assert context_recall(expected, CTX) == 0.0


def test_evaluate_one_returns_all_metrics():
    scores = evaluate_one(
        question="What is chunk overlap?",
        answer="Overlap repeats the last characters of one chunk at the start of the next chunk.",
        contexts=CTX,
        expected_answer="Overlap repeats the last characters of one chunk.",
    )
    assert set(scores) == {
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    }
    assert all(0.0 <= v <= 1.0 for v in scores.values())
