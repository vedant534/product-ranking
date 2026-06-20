"""Tests for unified multi-cutoff evaluation and reporting."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from evaluation.evaluate import (
    evaluate_recommender,
    filter_seen_target_examples,
    reconstruct_sessions,
    save_evaluation_result,
)


class OrderedCatalogRecommender:
    def recommend(self, prefix_items: list[int], k: int) -> list[int]:
        return list(range(1, k + 1))


def test_seen_known_targets_are_filtered_but_unknown_targets_remain() -> None:
    examples = pd.DataFrame(
        {
            "prefix_items": [[1, 2], [1, 2], [0, 2]],
            "target_item": [2, 3, 0],
        }
    )

    filtered = filter_seen_target_examples(examples)

    assert filtered["target_item"].tolist() == [3, 0]


def test_unified_evaluator_computes_hand_checked_metrics_at_all_cutoffs() -> None:
    examples = pd.DataFrame(
        {
            "prefix_items": [[10], [20]],
            "target_item": [2, 7],
        }
    )

    result = evaluate_recommender(
        OrderedCatalogRecommender(),
        examples,
        total_items=20,
        warmup_queries=0,
    )

    assert tuple(result["metrics"]) == ("5", "10", "20")
    assert result["metrics"]["5"]["recall"] == pytest.approx(0.5)
    assert result["metrics"]["5"]["hitrate"] == pytest.approx(0.5)
    assert result["metrics"]["5"]["mrr"] == pytest.approx(0.25)
    assert result["metrics"]["5"]["ndcg"] == pytest.approx((1 / math.log2(3)) / 2)
    assert result["metrics"]["5"]["coverage"] == pytest.approx(0.25)
    assert result["metrics"]["10"]["recall"] == pytest.approx(1.0)
    assert result["metrics"]["10"]["mrr"] == pytest.approx((1 / 2 + 1 / 7) / 2)
    assert result["metrics"]["20"]["coverage"] == pytest.approx(1.0)
    assert result["average_latency_ms"] >= 0.0


def test_reconstruct_sessions_does_not_repeat_expanding_prefixes() -> None:
    examples = pd.DataFrame(
        {
            "session_id": [1, 1, 2],
            "prefix_items": [[10], [10, 20], [40]],
            "target_item": [20, 30, 50],
        }
    )

    assert reconstruct_sessions(examples) == [[10, 20, 30], [40, 50]]


def test_results_are_merged_into_json_and_clean_markdown(tmp_path: Path) -> None:
    evaluation = {
        "number_of_examples": 2,
        "catalog_size": 20,
        "average_latency_ms": 0.01,
        "metrics": {
            "5": {"recall": 0.5, "hitrate": 0.5, "mrr": 0.25, "ndcg": 0.3, "coverage": 0.2},
            "10": {"recall": 0.6, "hitrate": 0.6, "mrr": 0.3, "ndcg": 0.4, "coverage": 0.3},
            "20": {"recall": 0.7, "hitrate": 0.7, "mrr": 0.4, "ndcg": 0.5, "coverage": 0.4},
        },
    }

    json_path, markdown_path = save_evaluation_result(
        "Popularity", "validation", evaluation, tmp_path
    )
    save_evaluation_result("Markov", "validation", evaluation, tmp_path)

    assert json_path.name == "results_validation.json"
    assert markdown_path.name == "results_validation.md"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(saved["splits"]["validation"]) == {"Popularity", "Markov"}
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| Model | K | Recall | HitRate | MRR | NDCG | Coverage" in markdown
    assert "| Popularity | 5 |" in markdown
    assert "| Markov | 20 |" in markdown


def test_validation_and_test_reports_are_isolated(tmp_path: Path) -> None:
    evaluation = {
        "number_of_examples": 1,
        "catalog_size": 3,
        "average_latency_ms": 0.0,
        "metrics": {
            "20": {
                "recall": 1.0,
                "hitrate": 1.0,
                "mrr": 1.0,
                "ndcg": 1.0,
                "coverage": 1.0,
            }
        },
    }

    validation_path, _ = save_evaluation_result(
        "Popularity", "validation", evaluation, tmp_path
    )
    test_path, _ = save_evaluation_result("Markov", "test", evaluation, tmp_path)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    assert set(validation["splits"]) == {"validation"}
    assert set(test["splits"]) == {"test"}
    assert set(validation["splits"]["validation"]) == {"Popularity"}
    assert set(test["splits"]["test"]) == {"Markov"}
