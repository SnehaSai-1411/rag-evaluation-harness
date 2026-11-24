"""Eval CLI: run the baseline RAG over the golden set and score it.

Usage:
    python -m eval_harness.run --dataset data/golden.jsonl
    python -m eval_harness.run --dataset data/golden.jsonl --chunk-size 400
    python -m eval_harness.run --dataset data/golden.jsonl --judge llm
    python -m eval_harness.run --dataset data/golden.jsonl --check-thresholds

Prints a score table (mean per metric + one row per question) and writes
results/results.json. With --check-thresholds, exits non-zero when mean
faithfulness or answer_relevance falls below config/thresholds.json
(this is what CI runs).
"""

import argparse
import json
import sys
from pathlib import Path

from eval_harness.baseline_rag import BaselineRAG
from eval_harness.judge import LLMJudge
from eval_harness.metrics import evaluate_one

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
DEFAULT_THRESHOLDS = Path(__file__).resolve().parent.parent / "config" / "thresholds.json"


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_thresholds(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def check_thresholds(mean_scores: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    """Return a list of failure messages (empty = gate passed).

    Only faithfulness and answer_relevance gate the build; the other two
    metrics are reported but not gated.
    """
    failures = []
    for metric in ("faithfulness", "answer_relevance"):
        threshold = thresholds.get(metric)
        if threshold is None:
            continue
        if mean_scores.get(metric, 0.0) < threshold:
            failures.append(
                f"{metric}: mean {mean_scores[metric]:.3f} < threshold {threshold:.3f}"
            )
    return failures


def run_eval(
    dataset: list[dict],
    rag: BaselineRAG,
    judge: LLMJudge | None = None,
) -> dict:
    """Run the RAG over the dataset and compute per-question + mean scores."""
    rows = []
    judge_rows = []
    for item in dataset:
        answer, contexts = rag.answer(item["question"])
        scores = evaluate_one(
            question=item["question"],
            answer=answer,
            contexts=contexts,
            expected_answer=item["expected_answer"],
        )
        row = {"id": item["id"], "question": item["question"], **scores}
        rows.append(row)
        if judge is not None:
            judge_scores = judge.score(item["question"], answer, contexts)
            if judge_scores is not None:
                judge_rows.append({"id": item["id"], **judge_scores})
    mean = {m: round(sum(r[m] for r in rows) / len(rows), 4) for m in METRICS} if rows else {}
    result: dict = {"mean": mean, "per_question": rows}
    if judge is not None:
        if judge_rows:
            result["judge_mean"] = {
                m: round(sum(r[m] for r in judge_rows) / len(judge_rows), 4)
                for m in ("faithfulness", "answer_relevance")
            }
            result["judge_per_question"] = judge_rows
            result["judge_scored"] = len(judge_rows)
        else:
            result["judge_scored"] = 0
    return result


def print_table(result: dict) -> None:
    header = f"{'id':<28} {'faith':>6} {'relev':>6} {'prec':>6} {'recall':>6}"
    print(header)
    print("-" * len(header))
    for row in result["per_question"]:
        print(
            f"{row['id']:<28} {row['faithfulness']:>6.3f} {row['answer_relevance']:>6.3f} "
            f"{row['context_precision']:>6.3f} {row['context_recall']:>6.3f}"
        )
    print("-" * len(header))
    mean = result["mean"]
    print(
        f"{'MEAN':<28} {mean['faithfulness']:>6.3f} {mean['answer_relevance']:>6.3f} "
        f"{mean['context_precision']:>6.3f} {mean['context_recall']:>6.3f}"
    )
    if "judge_mean" in result:
        jm = result["judge_mean"]
        print(
            f"\nLLM judge mean (n={result['judge_scored']}): "
            f"faithfulness={jm['faithfulness']:.3f}, answer_relevance={jm['answer_relevance']:.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument("--dataset", default="data/golden.jsonl")
    parser.add_argument("--docs-dir", default="data/sample_docs")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--judge", choices=["none", "llm"], default="none")
    parser.add_argument("--output", default="results/results.json")
    parser.add_argument("--check-thresholds", action="store_true")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    args = parser.parse_args(argv)

    dataset = load_dataset(Path(args.dataset))
    rag = BaselineRAG(args.docs_dir, chunk_size=args.chunk_size, top_k=args.top_k)

    judge = None
    if args.judge == "llm":
        candidate = LLMJudge()
        if candidate.available:
            judge = candidate
            print(f"LLM judge enabled ({candidate.provider}, {candidate.model}).")
        else:
            print(
                "NOTE: --judge llm requested but no ANTHROPIC_API_KEY or "
                "OPENAI_API_KEY is set; LLM judge skipped, heuristic metrics only."
            )

    result = run_eval(dataset, rag, judge)
    result["config"] = {
        "dataset": args.dataset,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "judge": args.judge if judge is not None else "none",
        "n_questions": len(dataset),
    }
    print_table(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")

    if args.check_thresholds:
        thresholds = load_thresholds(Path(args.thresholds))
        failures = check_thresholds(result["mean"], thresholds)
        if failures:
            print("\nTHRESHOLD GATE FAILED:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("\nThreshold gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
