# Chunk-size comparison: 400 vs 800 characters

Measured with the offline baseline (BM25 retrieval, top-k 3, extractive
answering) over the 40-question golden set in `data/golden.jsonl`.
Both runs used the same dataset, retriever, and metrics — only
`--chunk-size` changed.

Commands:

```bash
python -m eval_harness.run --dataset data/golden.jsonl --chunk-size 400
python -m eval_harness.run --dataset data/golden.jsonl --chunk-size 800
```

## Results

| Metric (mean over 40 questions) | chunk 400 | chunk 800 | Δ (800 − 400) |
| --- | --- | --- | --- |
| faithfulness | 1.000 | 1.000 | 0.000 |
| answer_relevance | 0.895 | 0.896 | +0.001 |
| context_precision | 0.600 | 0.617 | +0.017 |
| context_recall | 0.854 | 0.942 | +0.088 |

## Interpretation

Chunk size 800 wins on context recall (+0.088): with top-k fixed at 3, the
larger chunks more often contain the full expected answer, while 400-char
chunks split answers across chunk boundaries. Faithfulness is 1.000 either
way because the baseline answers extractively — its sentences come verbatim
from the retrieved context, so chunk size cannot make it unfaithful. Answer
relevance (+0.001) and context precision (+0.017) are effectively unchanged;
on a 40-question set, differences that small are within noise.

Two caveats: this is a small golden set over four short documents, so only
the recall gap is a clear signal, and this comparison uses the keyword
baseline — chunk-size effects may look different with a vector retriever,
which is the next comparison to run once one is plugged in.
