"""Tests for train-only candidate-pool recall analysis."""

import json

import pandas as pd
import pytest

from scripts.candidate_recall import (
    evaluate_candidate_recall,
    fit_candidate_generator,
    save_candidate_recall_report,
    save_quota_candidate_recall_report,
)


def _train_examples() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"session_id": 1, "prefix_items": [1], "target_item": 2},
            {"session_id": 1, "prefix_items": [1, 2], "target_item": 3},
            {"session_id": 2, "prefix_items": [1], "target_item": 2},
            {"session_id": 2, "prefix_items": [1, 2], "target_item": 4},
        ]
    )


def test_candidate_sources_are_fitted_only_from_supplied_train_examples() -> None:
    generator = fit_candidate_generator(_train_examples())

    assert set(generator.popularity.ranked_items) == {1, 2, 3, 4}
    assert 999 not in generator.markov.transition_counts
    assert 999 not in generator.item_knn.neighbor_scores


def test_native_sources_do_not_receive_popularity_fallback() -> None:
    generator = fit_candidate_generator(_train_examples())
    examples = pd.DataFrame({"prefix_items": [[999]], "target_item": [1]})

    recall = evaluate_candidate_recall(generator, examples, cutoffs=(1,))

    assert recall["popularity"]["1"] == 1.0
    assert recall["markov"]["1"] == 0.0
    assert recall["item_knn"]["1"] == 0.0
    assert recall["combined"]["1"] == 1.0


def test_unknown_targets_are_retained_as_candidate_misses() -> None:
    generator = fit_candidate_generator(_train_examples())
    examples = pd.DataFrame(
        {
            "prefix_items": [[1, 2], [1, 2]],
            "target_item": [3, 0],
        }
    )

    recall = evaluate_candidate_recall(generator, examples, cutoffs=(1, 2))

    assert recall["markov"]["1"] == pytest.approx(0.5)
    assert recall["combined"]["2"] == pytest.approx(0.5)


def test_candidate_recall_report_uses_split_specific_paths(tmp_path) -> None:
    recall = {
        source: {"100": 0.1, "500": 0.2, "1000": 0.3}
        for source in ("popularity", "markov", "item_knn", "combined")
    }

    json_path, markdown_path = save_candidate_recall_report(
        "test",
        recall,
        number_of_examples=10,
        unknown_targets=2,
        reports_dir=tmp_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_path.name == "candidate_recall_test.json"
    assert markdown_path.name == "candidate_recall_test.md"
    assert payload["fit_split"] == "train"
    assert payload["policy"]["source_fallbacks"] is False
    assert "Recall@1000" in markdown_path.read_text(encoding="utf-8")


def test_quota_candidate_report_records_observed_test_metadata(tmp_path) -> None:
    generator = fit_candidate_generator(_train_examples())
    recall = {
        source: {"100": 0.1, "500": 0.2, "1000": 0.3}
        for source in ("item_knn", "combined", "quota_combined")
    }

    json_path, _ = save_quota_candidate_recall_report(
        "test", recall, 10, 2, tmp_path, generator
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["report_status"] == "observed_test"
    assert payload["candidate_mode"] == "quota_combined"
    assert payload["resolved_quotas"]["1000"] == {
        "item_knn": 700,
        "markov": 200,
        "popularity": 100,
    }
