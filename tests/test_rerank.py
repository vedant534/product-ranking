"""Tests for MMR reranking and diversity metrics."""

import pytest

from ranking.recommend import RankedRecommendation
from ranking.rerank import (
    MMRReranker,
    category_count_at_k,
    intra_list_similarity_at_k,
    unique_recommended_items_at_k,
)


def _recommendations() -> list[RankedRecommendation]:
    return [
        RankedRecommendation(1, 1.0, ("neural",)),
        RankedRecommendation(2, 0.95, ("neural",)),
        RankedRecommendation(3, 0.80, ("neural",)),
    ]


def test_highly_similar_item_is_pushed_down() -> None:
    similarities = {1: {2: 1.0, 3: 0.0}, 2: {3: 0.0}}
    reranker = MMRReranker(similarities, alpha=0.5)

    reranked = reranker.rerank(_recommendations(), k=3)

    assert [item.item_id for item in reranked] == [1, 3, 2]


def test_relevance_still_matters_when_alpha_is_high() -> None:
    similarities = {1: {2: 1.0, 3: 0.0}, 2: {3: 0.0}}
    reranker = MMRReranker(similarities, alpha=0.95)

    reranked = reranker.rerank(_recommendations(), k=3)

    assert [item.item_id for item in reranked] == [1, 2, 3]


def test_alpha_changes_relevance_diversity_order() -> None:
    similarities = {1: {2: 1.0, 3: 0.0}, 2: {3: 0.0}}

    diversity_order = MMRReranker(similarities, alpha=0.5).rerank(
        _recommendations(),
        k=3,
    )
    relevance_order = MMRReranker(similarities, alpha=0.95).rerank(
        _recommendations(),
        k=3,
    )

    assert [item.item_id for item in diversity_order] == [1, 3, 2]
    assert [item.item_id for item in relevance_order] == [1, 2, 3]


def test_seen_items_are_excluded_by_reranker() -> None:
    reranker = MMRReranker({}, alpha=0.8)

    reranked = reranker.rerank(_recommendations(), k=2, seen_items={1})

    assert all(item.item_id != 1 for item in reranked)


def test_diversity_metrics_are_hand_computed() -> None:
    similarities = {1: {2: 0.8, 3: 0.2}, 2: {3: 0.4}}
    recommendations = _recommendations()

    assert intra_list_similarity_at_k(recommendations, similarities, k=3) == pytest.approx(
        (0.8 + 0.2 + 0.4) / 3
    )
    assert unique_recommended_items_at_k([[1, 2], [2, 3]], k=2) == 3
    assert category_count_at_k(recommendations, {1: "A", 2: "A", 3: "B"}, k=3) == 2
