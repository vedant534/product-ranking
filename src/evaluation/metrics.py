"""Offline ranking metrics for one relevant next-item target per example."""

import math
from collections.abc import Iterable
from numbers import Integral


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")
    return int(k)


def _prepare_ranking_inputs(
    predictions: Iterable[Iterable[object]],
    targets: Iterable[object],
    k: int,
) -> tuple[list[list[object]], list[object]]:
    cutoff = _validate_k(k)
    prediction_rows = [list(row)[:cutoff] for row in predictions]
    target_items = list(targets)
    if len(prediction_rows) != len(target_items):
        raise ValueError(
            "predictions and targets must contain the same number of examples; "
            f"found {len(prediction_rows)} predictions and {len(target_items)} targets"
        )
    return prediction_rows, target_items


def recall_at_k(
    predictions: Iterable[Iterable[object]],
    targets: Iterable[object],
    k: int,
) -> float:
    """Return mean Recall@k for one relevant target per example."""
    prediction_rows, target_items = _prepare_ranking_inputs(predictions, targets, k)
    if not target_items:
        return 0.0
    hits = sum(target in row for row, target in zip(prediction_rows, target_items))
    return float(hits / len(target_items))


def hitrate_at_k(
    predictions: Iterable[Iterable[object]],
    targets: Iterable[object],
    k: int,
) -> float:
    """Return mean HitRate@k, equivalent to Recall@k for one target."""
    return recall_at_k(predictions, targets, k)


def mrr_at_k(
    predictions: Iterable[Iterable[object]],
    targets: Iterable[object],
    k: int,
) -> float:
    """Return mean reciprocal rank with misses beyond k contributing zero."""
    prediction_rows, target_items = _prepare_ranking_inputs(predictions, targets, k)
    if not target_items:
        return 0.0

    reciprocal_rank_sum = 0.0
    for row, target in zip(prediction_rows, target_items):
        if target in row:
            reciprocal_rank_sum += 1.0 / (row.index(target) + 1)
    return reciprocal_rank_sum / len(target_items)


def ndcg_at_k(
    predictions: Iterable[Iterable[object]],
    targets: Iterable[object],
    k: int,
) -> float:
    """Return mean NDCG@k for a single binary-relevance target."""
    prediction_rows, target_items = _prepare_ranking_inputs(predictions, targets, k)
    if not target_items:
        return 0.0

    discounted_gain_sum = 0.0
    for row, target in zip(prediction_rows, target_items):
        if target in row:
            one_based_rank = row.index(target) + 1
            discounted_gain_sum += 1.0 / math.log2(one_based_rank + 1)
    return discounted_gain_sum / len(target_items)


def coverage_at_k(
    predictions: Iterable[Iterable[object]],
    total_items: int,
) -> float:
    """Return the fraction of the candidate catalog recommended at least once."""
    if (
        isinstance(total_items, bool)
        or not isinstance(total_items, Integral)
        or total_items <= 0
    ):
        raise ValueError("total_items must be a positive integer")

    recommended_items = {item for row in predictions for item in row}
    return len(recommended_items) / int(total_items)
