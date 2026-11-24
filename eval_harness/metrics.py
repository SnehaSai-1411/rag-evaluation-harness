"""Deterministic (heuristic) RAG evaluation metrics.

These metrics are token-overlap proxies, not the model-based metrics from
RAGAS or similar frameworks. They are deliberately simple so the whole eval
runs offline with no API key:

- faithfulness: fraction of answer sentences whose content tokens are
  (mostly) covered by the retrieved context.
- answer_relevance: fraction of the question's content tokens that appear
  in the answer.
- context_precision: fraction of retrieved chunks that share at least two
  content tokens with the question.
- context_recall: fraction of the expected answer's content tokens covered
  by the retrieved contexts.

All scores are in [0, 1]. Higher is better.

Known limitations (see README): token overlap misses paraphrases and
synonyms, so these scores understate quality when an answer rewords the
context. The optional LLM-as-judge path (eval_harness/judge.py) exists
for exactly that gap.
"""

import re

# Small stopword list so content tokens carry the signal. Not exhaustive;
# the goal is to keep common function words out of the overlap math.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "by", "at", "from", "as",
    "and", "or", "but", "so", "if", "than", "then", "that", "this",
    "these", "those", "it", "its", "it’s", "they", "them", "their",
    "he", "she", "his", "her", "we", "our", "you", "your", "i", "my",
    "do", "does", "did", "can", "could", "will", "would", "should",
    "may", "might", "must", "shall", "has", "have", "had", "not", "no",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "there", "here", "into", "over", "under", "between", "through",
    "such", "also", "more", "most", "other", "some", "any", "each",
    "all", "both", "either", "neither", "only", "very", "just", "per",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# The baseline's abstention text. An abstention makes no claims, so it is
# faithful by definition (1.0) but scores 0.0 on answer relevance.
ABSTAIN_TEXT = "I don't know based on the provided documents."


def content_tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens with stopwords removed."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? followed by whitespace."""
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def is_abstention(answer: str) -> bool:
    return answer.strip() == ABSTAIN_TEXT


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences supported by the retrieved context.

    A sentence is supported when at least half of its content tokens appear
    in the union of the retrieved contexts. An abstention makes no claims
    and scores 1.0. An empty answer scores 0.0.
    """
    if is_abstention(answer):
        return 1.0
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    context_tokens: set[str] = set()
    for ctx in contexts:
        context_tokens |= content_tokens(ctx)
    supported = 0
    for sentence in sentences:
        tokens = content_tokens(sentence)
        if not tokens:
            supported += 1  # no claims to check
            continue
        covered = len(tokens & context_tokens) / len(tokens)
        if covered >= 0.5:
            supported += 1
    return supported / len(sentences)


def answer_relevance(question: str, answer: str) -> float:
    """Fraction of the question's content tokens that appear in the answer.

    An abstention scores 0.0: it may be honest, but it does not answer
    the question.
    """
    if is_abstention(answer):
        return 0.0
    q_tokens = content_tokens(question)
    if not q_tokens:
        return 0.0
    a_tokens = content_tokens(answer)
    return len(q_tokens & a_tokens) / len(q_tokens)


def context_precision(question: str, contexts: list[str]) -> float:
    """Fraction of retrieved chunks relevant to the question.

    A chunk counts as relevant when it shares at least two content tokens
    with the question. No retrieved contexts scores 0.0.
    """
    if not contexts:
        return 0.0
    q_tokens = content_tokens(question)
    relevant = sum(
        1 for ctx in contexts if len(q_tokens & content_tokens(ctx)) >= 2
    )
    return relevant / len(contexts)


def context_recall(expected_answer: str, contexts: list[str]) -> float:
    """Fraction of the expected answer's content tokens covered by the
    retrieved contexts."""
    e_tokens = content_tokens(expected_answer)
    if not e_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for ctx in contexts:
        context_tokens |= content_tokens(ctx)
    return len(e_tokens & context_tokens) / len(e_tokens)


def evaluate_one(
    question: str,
    answer: str,
    contexts: list[str],
    expected_answer: str,
) -> dict[str, float]:
    """Compute all four heuristic metrics for one question."""
    return {
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevance": answer_relevance(question, answer),
        "context_precision": context_precision(question, contexts),
        "context_recall": context_recall(expected_answer, contexts),
    }
