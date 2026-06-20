"""Tests for artifact-only Streamlit report parsing and interpretation."""

import json

import pandas as pd
import pytest

from evaluation.reporting import (
    load_candidate_recall_summary,
    load_evaluation_summary,
    summarize_metric_winners,
)


def _evaluation_result(recall, mrr, ndcg, coverage):
    return {
        "average_latency_ms": 1.0,
        "metrics": {
            "20": {
                "recall": recall,
                "mrr": mrr,
                "ndcg": ndcg,
                "coverage": coverage,
            }
        },
    }


def test_missing_split_report_returns_none(tmp_path) -> None:
    assert load_evaluation_summary(tmp_path / "results_test.json", "test") is None


def test_evaluation_report_rejects_mislabeled_split(tmp_path) -> None:
    path = tmp_path / "results_test.json"
    path.write_text(json.dumps({"splits": {"validation": {}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain only the 'test' split"):
        load_evaluation_summary(path, "test")


def test_metric_interpretation_and_neural_recall_selection() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "Markov",
                "Recall@20": 0.3,
                "MRR@20": 0.4,
                "NDCG@20": 0.35,
                "Coverage@20": 0.2,
            },
            {
                "model": "Item-KNN",
                "Recall@20": 0.4,
                "MRR@20": 0.3,
                "NDCG@20": 0.4,
                "Coverage@20": 0.8,
            },
            {
                "model": "GRU4Rec",
                "Recall@20": 0.1,
                "MRR@20": 0.1,
                "NDCG@20": 0.1,
                "Coverage@20": 0.1,
            },
            {
                "model": "GRU4Rec + diversity (MMR)",
                "Recall@20": 0.2,
                "MRR@20": 0.2,
                "NDCG@20": 0.2,
                "Coverage@20": 0.3,
            },
        ]
    )

    winners = summarize_metric_winners(summary)

    assert winners["Recall@20"][0] == "Item-KNN"
    assert winners["MRR@20"][0] == "Markov"
    assert winners["Best neural variant"][0] == "GRU4Rec + diversity (MMR)"


def test_candidate_recall_parser_validates_schema(tmp_path) -> None:
    path = tmp_path / "candidate_recall_test.json"
    path.write_text(
        json.dumps({"split": "test", "cutoffs": [100], "recall": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid schema"):
        load_candidate_recall_summary(path, "test")
