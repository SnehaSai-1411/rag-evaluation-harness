"""A tiny offline baseline RAG, so the eval runs with no model and no key.

Retrieval is BM25 keyword search (rank-bm25) over fixed-size character
chunks of the sample docs. Answering is extractive: the sentences from the
retrieved chunks with the highest token overlap with the question, joined
together. If no retrieved sentence shares any content token with the
question, the baseline abstains instead of guessing.

The baseline accepts an optional ``embedder`` with an ``embed(texts)``
method so a vector retriever can be plugged in later; when none is given
(the default) it is purely keyword-based and needs no model download.

``chunk_size`` is configurable so chunking strategies can be compared
with the same eval (see docs/comparison.md).
"""

from pathlib import Path

from rank_bm25 import BM25Okapi

from eval_harness.metrics import ABSTAIN_TEXT, content_tokens, split_sentences


class Embedder:
    """Interface a vector embedder must satisfy to plug into the baseline.

    Not used by the keyword baseline itself; kept so a sentence-transformers
    or API embedder can be dropped in without changing the RAG interface.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def load_docs(docs_dir: str | Path) -> list[dict[str, str]]:
    """Read all .md files in a directory. Returns [{source, text}]."""
    docs = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Strip Markdown heading markers but keep the heading text.
        lines = [line.lstrip("# ").rstrip() for line in text.splitlines()]
        docs.append({"source": path.name, "text": "\n".join(lines)})
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character windows with overlap.

    Every returned chunk is at most ``chunk_size`` characters long.
    Chunks are cut on whitespace where possible so words are not split.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    text = " ".join(text.split())  # collapse all whitespace
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer to break at the last space inside the window.
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
        # Skip ahead past any whitespace at the new start.
        while start < len(text) and text[start] == " ":
            start += 1
    return [c for c in chunks if c]


class BaselineRAG:
    """BM25 retrieval + extractive answering over a folder of docs."""

    def __init__(
        self,
        docs_dir: str | Path,
        chunk_size: int = 800,
        overlap: int = 50,
        top_k: int = 3,
        embedder: Embedder | None = None,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.embedder = embedder  # reserved for a future vector retriever
        self.chunks: list[dict[str, str]] = []
        for doc in load_docs(self.docs_dir):
            for chunk in chunk_text(doc["text"], chunk_size, overlap):
                self.chunks.append({"text": chunk, "source": doc["source"]})
        if not self.chunks:
            raise ValueError(f"no chunks built from {docs_dir}")
        tokenized = [sorted(content_tokens(c["text"])) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k chunks for a query, each with text/source/score."""
        k = top_k if top_k is not None else self.top_k
        query_tokens = sorted(content_tokens(query))
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(self.chunks)), key=lambda i: scores[i], reverse=True
        )
        return [
            {
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "score": round(float(scores[i]), 4),
            }
            for i in ranked[:k]
        ]

    def answer(self, query: str) -> tuple[str, list[str]]:
        """Extractive answer plus the retrieved context texts.

        Picks the two retrieved sentences with the highest question-token
        overlap. Abstains when nothing overlaps.
        """
        retrieved = self.retrieve(query)
        contexts = [r["text"] for r in retrieved]
        q_tokens = content_tokens(query)
        candidates: list[tuple[int, str]] = []
        seen: set[str] = set()
        for ctx in contexts:
            for sentence in split_sentences(ctx):
                if sentence in seen:
                    continue
                seen.add(sentence)
                overlap = len(q_tokens & content_tokens(sentence))
                if overlap > 0:
                    candidates.append((overlap, sentence))
        if not candidates:
            return ABSTAIN_TEXT, contexts
        candidates.sort(key=lambda c: c[0], reverse=True)
        return " ".join(s for _, s in candidates[:2]), contexts
