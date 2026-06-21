"""Tests for artifact-only Streamlit report parsing and interpretation."""

import json

import pandas as pd
import pytest

from evaluation.reporting import (
    load_candidate_gru_summary,
    load_candidate_recall_summary,
    load_evaluation_summary,
    load_markdown_report,
    load_quota_candidate_recall_summary,
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


def test_candidate_gru_report_parser_returns_controlled_rows(tmp_path) -> None:
    path = tmp_path / "candidate_gru_validation.json"
    models = {
        "Current GRU - quota_combined": _evaluation_result(0.1, 0.04, 0.05, 0.14),
        "Candidate-aware GRU - quota_combined": _evaluation_result(0.22, 0.09, 0.12, 0.51),
        "Candidate-aware GRU - quota_combined + MMR": _evaluation_result(
            0.21, 0.08, 0.11, 0.49
        ),
    }
    path.write_text(
        json.dumps(
            {
                "metadata": {"split": "validation", "report_status": "validation"},
                "models": models,
            }
        ),
        encoding="utf-8",
    )

    summary = load_candidate_gru_summary(path, "validation")

    assert summary["model"].tolist() == [
        "Current GRU on quota pool",
        "Candidate-aware GRU",
        "Candidate-aware GRU + MMR",
    ]
    assert summary.loc[1, "MRR@20"] == pytest.approx(0.09)


def test_candidate_gru_report_parser_rejects_non_observed_test(tmp_path) -> None:
    path = tmp_path / "candidate_gru_test.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {"split": "test", "report_status": "test"},
                "models": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="observed_test"):
        load_candidate_gru_summary(path, "test")


def test_quota_candidate_recall_parser_compares_saved_pools(tmp_path) -> None:
    path = tmp_path / "candidate_recall_quota_validation.json"
    path.write_text(
        json.dumps(
            {
                "split": "validation",
                "candidate_mode": "quota_combined",
                "cutoffs": [100, 500, 1000],
                "recall": {
                    "combined": {"100": 0.30, "500": 0.39, "1000": 0.43},
                    "quota_combined": {"100": 0.35, "500": 0.42, "1000": 0.45},
                },
            }
        ),
        encoding="utf-8",
    )

    summary = load_quota_candidate_recall_summary(path, "validation")

    assert summary["K"].tolist() == [100, 500, 1000]
    assert summary.loc[0, "improvement"] == pytest.approx(0.05)


def test_markdown_report_is_artifact_only_and_rejects_empty_files(tmp_path) -> None:
    path = tmp_path / "diagnostics.md"
    assert load_markdown_report(path) is None
    path.write_text("# Diagnostics\n", encoding="utf-8")
    assert load_markdown_report(path) == "# Diagnostics"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_markdown_report(path)
