"""Hand-computed tests for offline next-item ranking metrics."""

import math

import pytest

from evaluation.metrics import (
    coverage_at_k,
    hitrate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_metrics_match_provided_hand_computed_example() -> None:
    predictions = [[1, 2, 3], [4, 5, 6]]
    targets = [2, 7]

    assert recall_at_k(predictions, targets, k=3) == pytest.approx(0.5)
    assert hitrate_at_k(predictions, targets, k=3) == pytest.approx(0.5)
    assert mrr_at_k(predictions, targets, k=3) == pytest.approx(0.25)
    assert ndcg_at_k(predictions, targets, k=3) == pytest.approx(
        (1.0 / math.log2(3)) / 2
    )


def test_items_beyond_cutoff_are_misses() -> None:
    predictions = [[1, 2, 3], [4, 5, 6]]
    targets = [3, 5]

    assert recall_at_k(predictions, targets, k=2) == pytest.approx(0.5)
    assert hitrate_at_k(predictions, targets, k=2) == pytest.approx(0.5)
    assert mrr_at_k(predictions, targets, k=2) == pytest.approx(0.25)
    assert ndcg_at_k(predictions, targets, k=2) == pytest.approx(
        (1.0 / math.log2(3)) / 2
    )


def test_rank_one_and_rank_three_have_expected_values() -> None:
    predictions = [[9, 2, 3], [4, 5, 6]]
    targets = [9, 6]

    assert recall_at_k(predictions, targets, k=3) == pytest.approx(1.0)
    assert mrr_at_k(predictions, targets, k=3) == pytest.approx((1.0 + 1.0 / 3) / 2)
    assert ndcg_at_k(predictions, targets, k=3) == pytest.approx((1.0 + 0.5) / 2)


def test_duplicate_predictions_use_the_first_target_rank() -> None:
    predictions = [[5, 5, 7]]
    targets = [5]

    assert mrr_at_k(predictions, targets, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(predictions, targets, k=3) == pytest.approx(1.0)


def test_coverage_counts_distinct_recommended_items() -> None:
    predictions = [[1, 2, 2], [2, 3], []]

    assert coverage_at_k(predictions, total_items=10) == pytest.approx(0.3)


def test_empty_ranking_inputs_return_zero() -> None:
    assert recall_at_k([], [], k=20) == 0.0
    assert hitrate_at_k([], [], k=10) == 0.0
    assert mrr_at_k([], [], k=20) == 0.0
    assert ndcg_at_k([], [], k=20) == 0.0
    assert coverage_at_k([], total_items=10) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, hitrate_at_k, mrr_at_k, ndcg_at_k])
def test_ranking_metrics_reject_mismatched_example_counts(metric: object) -> None:
    with pytest.raises(ValueError, match="same number of examples"):
        metric([[1, 2, 3]], [1, 2], 3)


@pytest.mark.parametrize("invalid_k", [0, -1, 1.5, True])
def test_ranking_metrics_reject_invalid_cutoffs(invalid_k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k([[1]], [1], invalid_k)


@pytest.mark.parametrize("total_items", [0, -1, 1.5, True])
def test_coverage_rejects_invalid_catalog_sizes(total_items: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        coverage_at_k([[1]], total_items)
