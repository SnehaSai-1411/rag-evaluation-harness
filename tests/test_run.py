"""End-to-end tests for the eval CLI and the threshold gate.

The CLI runs fully offline with the mock (heuristic) judge; no API key
and no network access are involved.
"""

import json
from pathlib import Path

from eval_harness import run as run_module
from eval_harness.baseline_rag import BaselineRAG
from eval_harness.run import check_thresholds, load_dataset, main, run_eval

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden.jsonl"
DOCS_DIR = ROOT / "data" / "sample_docs"


def test_run_eval_end_to_end_offline():
    dataset = load_dataset(GOLDEN)
    rag = BaselineRAG(DOCS_DIR, chunk_size=800)
    result = run_eval(dataset, rag, judge=None)
    assert len(result["per_question"]) == len(dataset)
    assert set(result["mean"]) == {
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
    }
    assert all(0.0 <= v <= 1.0 for v in result["mean"].values())
    # The extractive baseline should be reasonably faithful on its own docs.
    assert result["mean"]["faithfulness"] > 0.8


def test_cli_writes_results_json(tmp_path):
    output = tmp_path / "results.json"
    exit_code = main(
        [
            "--dataset", str(GOLDEN),
            "--docs-dir", str(DOCS_DIR),
            "--chunk-size", "400",
            "--output", str(output),
        ]
    )
    assert exit_code == 0
    assert output.exists()
    data = json.loads(output.read_text())
    assert data["config"]["chunk_size"] == 400
    assert data["config"]["n_questions"] == 40
    assert len(data["per_question"]) == 40


def test_threshold_gate_passes_when_scores_clear():
    mean = {"faithfulness": 0.95, "answer_relevance": 0.60}
    assert check_thresholds(mean, {"faithfulness": 0.85, "answer_relevance": 0.50}) == []


def test_threshold_gate_fails_when_scores_below():
    mean = {"faithfulness": 0.70, "answer_relevance": 0.60}
    failures = check_thresholds(mean, {"faithfulness": 0.85, "answer_relevance": 0.50})
    assert len(failures) == 1
    assert "faithfulness" in failures[0]

    mean = {"faithfulness": 0.95, "answer_relevance": 0.30}
    failures = check_thresholds(mean, {"faithfulness": 0.85, "answer_relevance": 0.50})
    assert len(failures) == 1
    assert "answer_relevance" in failures[0]


def test_threshold_gate_ignores_ungated_metrics():
    # context_precision / context_recall are reported, not gated.
    mean = {
        "faithfulness": 0.95,
        "answer_relevance": 0.60,
        "context_precision": 0.10,
        "context_recall": 0.10,
    }
    assert check_thresholds(mean, {"faithfulness": 0.85, "answer_relevance": 0.50}) == []


def test_cli_check_thresholds_exit_code(tmp_path):
    # A thresholds file the baseline cannot clear must exit non-zero.
    strict = tmp_path / "thresholds.json"
    strict.write_text(json.dumps({"faithfulness": 1.01, "answer_relevance": 0.0}))
    exit_code = main(
        [
            "--dataset", str(GOLDEN),
            "--docs-dir", str(DOCS_DIR),
            "--output", str(tmp_path / "results.json"),
            "--check-thresholds",
            "--thresholds", str(strict),
        ]
    )
    assert exit_code == 1

    # And the repo's own thresholds must pass (they were set from the
    # measured baseline with margin).
    exit_code = main(
        [
            "--dataset", str(GOLDEN),
            "--docs-dir", str(DOCS_DIR),
            "--output", str(tmp_path / "results2.json"),
            "--check-thresholds",
        ]
    )
    assert exit_code == 0
