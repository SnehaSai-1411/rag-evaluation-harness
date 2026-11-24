# rag-evaluation-harness

A small, offline RAG evaluation harness: a golden dataset, four heuristic metrics, an optional LLM-as-judge, and a CI gate that fails the build when scores regress.

## What it does

- Scores a RAG system on **faithfulness, answer relevance, context precision, and context recall** over a versioned 40-question golden dataset (`data/golden.jsonl`).
- Runs **fully offline** — a built-in BM25 + extractive-answering baseline needs no model download and no API key, so the eval works in CI for free.
- Offers an **optional LLM-as-judge** path (Anthropic or OpenAI via `.env`) for the nuance token-overlap metrics miss; when no key is set, it is skipped with a clear note.
- Compares configurations with one flag — e.g. `--chunk-size 400` vs `800` — and writes machine-readable results to `results/results.json`.
- **Gates CI on quality**: GitHub Actions runs pytest plus the eval on every push/PR and fails if mean faithfulness or answer relevance drops below `config/thresholds.json`.

## Architecture

```
data/golden.jsonl ──┐
                    ▼
data/sample_docs/ ──► BaselineRAG (BM25 retrieval + extractive answers)
                    │        │
                    │        ▼
                    │   eval_harness/metrics.py ──► score table + results/results.json
                    │        │
                    │        └── (optional) LLM-as-judge ──► judge scores
                    ▼
              config/thresholds.json ──► CI gate (pass / fail)
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m eval_harness.run --dataset data/golden.jsonl
.venv/bin/python -m pytest -q
```

## Example

```bash
$ python -m eval_harness.run --dataset data/golden.jsonl --chunk-size 800

id                            faith  relev   prec recall
--------------------------------------------------------
chunking-01                   1.000  1.000  0.667  1.000
chunking-02                   1.000  0.750  1.000  1.000
chunking-03                   1.000  0.667  0.333  0.941
chunking-04                   1.000  1.000  1.000  0.923
chunking-05                   1.000  0.500  0.667  1.000
... (35 more rows)
--------------------------------------------------------
MEAN                          1.000  0.896  0.617  0.942

Wrote results/results.json
```

The table above is the real output of the default run (chunk size 800, 40 questions). Per-question rows and the full config are also saved in `results/results.json`; a copy of a real run is committed as `results/baseline.example.json`. See `docs/comparison.md` for a measured chunk-size comparison (400 vs 800).

## Design decisions & trade-offs

- **Heuristic metrics + optional judge instead of RAGAS.** RAGAS pulls in heavy dependencies and needs an API key for every run; token-overlap proxies run offline in CI for free and deterministically, and the LLM judge is there when a key is available and nuance matters. The trade-off is honest: overlap metrics miss paraphrases, so they understate quality when an answer rewords the context.
- **Extractive keyword baseline, not a generative one.** BM25 retrieval plus sentence extraction needs no model and no key, and it isolates chunking/retrieval effects from LLM variance — which is what you want when comparing configurations. A vector retriever can be plugged in via the `embedder` interface on `BaselineRAG` (sentence-transformers + Chroma are the intended fit); no torch dependency is included because the offline baseline doesn't need it.
- **Thresholds were set from the measured baseline, not guessed.** The default run measured faithfulness 1.000 and answer relevance 0.896, so `config/thresholds.json` gates at 0.90 and 0.80 — a small margin below the real baseline, tight enough to catch regressions.
- **Golden contexts are generated, not hand-typed.** `scripts/build_golden.py` extracts each context verbatim from the sample docs, so the dataset can never drift out of sync with its sources (a test enforces this).

## What you'd add next

- Plug in a real vector retriever (sentence-transformers + Chroma) and re-run the chunk-size comparison against it.
- Add an LLM-generated answering mode alongside the extractive baseline, so the judge scores something that can actually be unfaithful.
- Grow the golden set past 40 questions and add harder questions (multi-hop, unanswerable) so small metric differences stop being noise.
- Gate on all four metrics once the retriever is real, and track scores over time instead of only pass/fail.
