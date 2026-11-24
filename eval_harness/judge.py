"""Optional LLM-as-judge scoring (Anthropic or OpenAI, key from .env).

The judge scores faithfulness and answer relevance on a 0-1 scale against
a short rubric. It is strictly optional: when no API key is set, the
judge reports itself unavailable and the CLI skips it with a clear note
instead of failing. The heuristic metrics in metrics.py always run.

Provider selection: ANTHROPIC_API_KEY wins if set, otherwise
OPENAI_API_KEY. JUDGE_MODEL overrides the default model for either
provider.
"""

import json
import os
import re

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

_JUDGE_PROMPT = """You are grading a RAG answer. Score it on two metrics, each from 0.0 to 1.0:

- faithfulness: every claim in the answer is supported by the retrieved context (1.0 = fully supported, 0.0 = contradicts or invents facts).
- answer_relevance: the answer addresses the question asked (1.0 = directly answers it, 0.0 = answers something else or says nothing useful).

Question: {question}

Retrieved context:
{context}

Answer: {answer}

Reply with JSON only, exactly like: {{"faithfulness": 0.9, "answer_relevance": 0.8}}
"""

_JSON_RE = re.compile(r"\{[^}]*\}")


def _parse_scores(text: str) -> dict[str, float] | None:
    """Pull {"faithfulness": x, "answer_relevance": y} out of model output."""
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        faith = float(data["faithfulness"])
        rel = float(data["answer_relevance"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= faith <= 1.0 and 0.0 <= rel <= 1.0):
        return None
    return {"faithfulness": faith, "answer_relevance": rel}


class LLMJudge:
    """LLM-as-judge. ``available`` is False when no API key is configured."""

    def __init__(self, model: str | None = None) -> None:
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if self.anthropic_key:
            self.provider = "anthropic"
            self.model = model or os.getenv("JUDGE_MODEL") or DEFAULT_ANTHROPIC_MODEL
        elif self.openai_key:
            self.provider = "openai"
            self.model = model or os.getenv("JUDGE_MODEL") or DEFAULT_OPENAI_MODEL
        else:
            self.provider = None
            self.model = model or os.getenv("JUDGE_MODEL") or DEFAULT_ANTHROPIC_MODEL

    @property
    def available(self) -> bool:
        return self.provider is not None

    def score(
        self, question: str, answer: str, contexts: list[str]
    ) -> dict[str, float] | None:
        """Score one answer. Returns None when the judge is unavailable
        or the model reply cannot be parsed."""
        if not self.available:
            return None
        prompt = _JUDGE_PROMPT.format(
            question=question,
            context="\n---\n".join(contexts),
            answer=answer,
        )
        try:
            if self.provider == "anthropic":
                client = anthropic.Anthropic(api_key=self.anthropic_key)
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text
            else:
                client = openai.OpenAI(api_key=self.openai_key)
                resp = client.chat.completions.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.choices[0].message.content or ""
        except Exception:
            return None
        return _parse_scores(text)
