"""Build data/golden.jsonl from the sample docs.

Each golden item's ``context`` is extracted verbatim from the source
document (one paragraph), so the dataset can never drift out of sync with
the docs. Questions and expected answers are authored here; the context
is never hand-typed.

Usage:
    python scripts/build_golden.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "sample_docs"
OUT = ROOT / "data" / "golden.jsonl"

# (source file, paragraph index, question, expected answer)
# Paragraph index counts blank-line-separated blocks, skipping headings.
QA = [
    # --- chunking.md ---
    ("chunking.md", 0, "What is chunking in a RAG system?",
     "Chunking splits long documents into smaller pieces so a retrieval system can return the most relevant passage instead of an entire file."),
    ("chunking.md", 1, "How does fixed-size chunking work?",
     "Fixed-size chunking cuts text every N characters regardless of sentence or paragraph boundaries."),
    ("chunking.md", 2, "How does recursive chunking work?",
     "Recursive chunking tries separators in order, such as double newlines, single newlines, spaces, and finally individual characters, keeping paragraphs and sentences intact when possible."),
    ("chunking.md", 3, "What is chunk overlap and why use it?",
     "Overlap repeats the last N characters of one chunk at the start of the next chunk, preserving context across chunk boundaries."),
    ("chunking.md", 4, "What are the trade-offs of small chunks?",
     "Small chunks, around 200 to 400 characters, give precise retrieval because each chunk covers a narrow topic, but a chunk may be too short to contain a complete answer."),
    ("chunking.md", 5, "What are the trade-offs of large chunks?",
     "Large chunks, around 800 to 1500 characters, carry more context and are more likely to contain a complete answer, but with lower precision."),
    ("chunking.md", 6, "What is semantic chunking?",
     "Semantic chunking groups sentences by meaning instead of by character count, starting a new chunk when the similarity between neighboring sentences drops."),
    ("chunking.md", 7, "What metadata should a chunk record?",
     "Chunk metadata should record the source document, the page or section, and the chunk's position in the original document."),
    ("chunking.md", 8, "Why should text be cleaned before chunking?",
     "Headers, footers, page numbers, and boilerplate should be removed first, because otherwise they appear in many chunks and pollute retrieval results."),
    ("chunking.md", 9, "Is there a single best chunk size?",
     "No. The right choice depends on the documents, the questions, and the embedding model, so it should be measured with an evaluation harness on a golden dataset rather than guessed."),
    # --- embeddings.md ---
    ("embeddings.md", 0, "What is an embedding?",
     "An embedding is a list of numbers, called a vector, that represents the meaning of a piece of text."),
    ("embeddings.md", 1, "How are embedding models trained?",
     "Embedding models are trained so that texts with similar meanings end up close together in vector space, using large amounts of paired text such as questions and their answers."),
    ("embeddings.md", 2, "What is embedding dimensionality?",
     "Dimensionality is the number of values in an embedding vector; common sizes range from 384 dimensions for small local models to 1536 dimensions for large API models."),
    ("embeddings.md", 3, "What is cosine similarity?",
     "Cosine similarity measures the angle between two vectors; a value of 1 means the vectors point in exactly the same direction, while 0 means they are unrelated."),
    ("embeddings.md", 4, "What is sentence-transformers?",
     "Sentence-transformers is a Python library for running embedding models locally on CPU; a popular small model is all-MiniLM-L6-v2, which produces 384-dimensional embeddings."),
    ("embeddings.md", 5, "What are the trade-offs of local embedding models?",
     "Local embedding models have no per-request cost and keep your documents on your own machine, but they are lower quality than the largest API models."),
    ("embeddings.md", 6, "What are the trade-offs of API embedding models?",
     "API embedding models are usually more accurate than small local models, but you pay per token and every document and query leaves your machine."),
    ("embeddings.md", 7, "Why must the same embedding model be used for indexing and querying?",
     "If documents are embedded with one model and the question with another, the vectors live in different spaces and similarity scores become meaningless."),
    ("embeddings.md", 8, "How does domain affect embedding quality?",
     "A model trained on general web text may struggle with specialized vocabulary, so for a niche corpus it is worth testing a domain-tuned model."),
    ("embeddings.md", 9, "Why normalize embeddings before storage?",
     "Normalizing each vector to unit length turns cosine similarity into a simple dot product, which is faster to compute at query time."),
    # --- vector-search.md ---
    ("vector-search.md", 0, "What is vector search?",
     "Vector search finds the documents whose embeddings are closest to the embedding of a query, matching meaning instead of keywords."),
    ("vector-search.md", 1, "How does exact nearest-neighbor search work?",
     "Exact nearest-neighbor search compares the query vector against every vector in the index; it is perfectly accurate but slow for large collections."),
    ("vector-search.md", 2, "What is approximate nearest neighbor search?",
     "Approximate nearest neighbor search trades a small amount of accuracy for much faster queries by organizing vectors so the search can skip most of the index."),
    ("vector-search.md", 3, "What is HNSW?",
     "HNSW, short for Hierarchical Navigable Small World, is a popular approximate nearest neighbor algorithm that builds a layered graph of vectors connected to their nearest neighbors."),
    ("vector-search.md", 4, "What does a vector index store?",
     "An index stores vectors together with metadata such as the source document and the chunk text."),
    ("vector-search.md", 5, "What is top-k retrieval?",
     "Top-k retrieval returns the k closest vectors to the query; a small k returns only the most relevant chunks, while a large k returns more context."),
    ("vector-search.md", 6, "What is Chroma?",
     "Chroma is a lightweight open-source vector database that runs locally with no server to manage and stores embeddings in a persistent directory on disk."),
    ("vector-search.md", 7, "What is metadata filtering?",
     "Metadata filtering narrows a vector search to a subset of the index, such as chunks from a single source document, before ranking by similarity."),
    ("vector-search.md", 8, "What is a weakness of vector search?",
     "Vector search struggles with exact terms such as product codes, names, and numbers, which is why many systems combine it with keyword search in a hybrid retriever."),
    ("vector-search.md", 9, "What determines relevance in vector search?",
     "Relevance in vector search is only as good as the embeddings and the chunking behind it; if chunks are too large or the model is a poor fit, the index returns the wrong passages."),
    # --- retrieval-evaluation.md ---
    ("retrieval-evaluation.md", 0, "What is retrieval in a RAG system?",
     "Retrieval is the step in a RAG system that selects which chunks to show the language model, and its quality sets an upper bound on answer quality."),
    ("retrieval-evaluation.md", 1, "How does top-k affect precision and recall?",
     "Increasing k raises recall, since more of the relevant material is included, but it lowers precision, since more irrelevant material is included too."),
    ("retrieval-evaluation.md", 2, "What is reranking?",
     "Reranking runs a second, more expensive model over the retrieved chunks to reorder them by relevance, typically narrowing twenty candidates down to the best five."),
    ("retrieval-evaluation.md", 3, "What is a golden dataset?",
     "A golden dataset is a small set of questions with known correct answers and the source passages that support them, versioned in the repository like code."),
    ("retrieval-evaluation.md", 4, "What is context precision?",
     "Context precision measures the fraction of retrieved chunks that are actually relevant to the question."),
    ("retrieval-evaluation.md", 5, "What is context recall?",
     "Context recall measures the fraction of the relevant material that the retriever found."),
    ("retrieval-evaluation.md", 6, "What is faithfulness?",
     "Faithfulness checks whether the answer is supported by the retrieved context; an answer is faithful when every claim in it can be traced back to a retrieved passage."),
    ("retrieval-evaluation.md", 7, "What is answer relevance?",
     "Answer relevance measures how well the answer addresses the question that was asked."),
    ("retrieval-evaluation.md", 8, "What is LLM-as-judge evaluation?",
     "LLM-as-judge evaluation asks a language model to score an answer against a rubric; it catches nuance that token-overlap metrics miss, but it costs money per evaluation and its scores can vary between runs."),
    ("retrieval-evaluation.md", 9, "Why put evaluation in continuous integration?",
     "A CI gate that fails the build when mean scores drop below a threshold turns evaluation from a one-off experiment into a regression test."),
]


def paragraphs_of(doc_path: Path) -> list[str]:
    """Blank-line-separated paragraphs, skipping Markdown headings."""
    text = doc_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n")]
    return [b for b in blocks if b and not b.startswith("#")]


def main() -> None:
    docs = {}
    for doc in sorted(DOCS_DIR.glob("*.md")):
        paras = paragraphs_of(doc)
        assert len(paras) == 10, f"{doc.name}: expected 10 paragraphs, got {len(paras)}"
        docs[doc.name] = paras

    rows = []
    for i, (source, idx, question, expected) in enumerate(QA, start=1):
        context = docs[source][idx]
        rows.append({
            "id": f"{Path(source).stem}-{idx + 1:02d}",
            "question": question,
            "expected_answer": expected,
            "context": context,
            "source": source,
            "page": 1,
        })

    assert 30 <= len(rows) <= 50, f"golden set must have 30-50 items, got {len(rows)}"
    assert len({r['id'] for r in rows}) == len(rows), "duplicate ids"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} items to {OUT}")


if __name__ == "__main__":
    main()
