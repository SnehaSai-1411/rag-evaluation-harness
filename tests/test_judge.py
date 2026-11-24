"""Tests for the optional LLM-as-judge path.

No real API calls are made: the "no key" case is tested with env vars
cleared, and the "key present" case uses a mocked Anthropic client.
"""

import os

import pytest

from eval_harness.judge import LLMJudge, _parse_scores


def test_parse_scores_valid():
    text = 'Here are the scores: {"faithfulness": 0.9, "answer_relevance": 0.8} done.'
    assert _parse_scores(text) == {"faithfulness": 0.9, "answer_relevance": 0.8}


def test_parse_scores_invalid():
    assert _parse_scores("no json here") is None
    assert _parse_scores('{"faithfulness": 1.5, "answer_relevance": 0.5}') is None
    assert _parse_scores('{"faithfulness": 0.5}') is None


def test_judge_unavailable_without_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    judge = LLMJudge()
    assert judge.available is False
    assert judge.score("q", "a", ["ctx"]) is None


def test_judge_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    judge = LLMJudge()
    assert judge.provider == "anthropic"


def test_judge_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    judge = LLMJudge()
    assert judge.provider == "openai"


class _FakeMessage:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeMessage(text)]


class _FakeAnthropicClient:
    def __init__(self, *args, **kwargs):
        pass

    class messages:
        @staticmethod
        def create(**kwargs):
            return _FakeResponse('{"faithfulness": 0.9, "answer_relevance": 0.75}')


def test_judge_score_with_mocked_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import eval_harness.judge as judge_module

    monkeypatch.setattr(judge_module.anthropic, "Anthropic", _FakeAnthropicClient)
    judge = LLMJudge()
    assert judge.available is True
    scores = judge.score("What is chunking?", "Chunking splits documents.", ["Chunking splits documents."])
    assert scores == {"faithfulness": 0.9, "answer_relevance": 0.75}
