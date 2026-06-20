"""MMR diversity reranking and recommendation-list diversity metrics."""

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from numbers import Integral
from typing import Optional, Union

import numpy as np

from ranking.recommend import RankedRecommendation

SimilaritySource = Union[
    Mapping[object, Mapping[object, float]],
    Callable[[object, object], float],
]


def _item_id(item: object) -> object:
    return item.item_id if isinstance(item, RankedRecommendation) else item


def item_similarity(
    first_item: object,
    second_item: object,
    similarities: SimilaritySource,
) -> float:
    if first_item == second_item:
        return 1.0
    if callable(similarities):
        return float(similarities(first_item, second_item))
    forward = similarities.get(first_item, {}).get(second_item)
    if forward is not None:
        return float(forward)
    return float(similarities.get(second_item, {}).get(first_item, 0.0))


class MMRReranker:
    """Greedily balance normalized relevance and similarity to selected items."""

    def __init__(self, similarities: SimilaritySource, alpha: float = 0.8) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in the interval [0, 1]")
        self.similarities = similarities
        self.alpha = float(alpha)

    def rerank(
        self,
        recommendations: list[RankedRecommendation],
        k: int,
        seen_items: Optional[set[object]] = None,
    ) -> list[RankedRecommendation]:
        if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
            raise ValueError("k must be a positive integer")
        seen = seen_items or set()
        remaining = [item for item in recommendations if item.item_id not in seen]
        if not remaining:
            return []

        minimum = min(item.score for item in remaining)
        maximum = max(item.score for item in remaining)
        if maximum == minimum:
            relevance = {item.item_id: 1.0 for item in remaining}
        else:
            scale = maximum - minimum
            relevance = {
                item.item_id: (item.score - minimum) / scale for item in remaining
            }

        item_ids = [item.item_id for item in remaining]
        item_by_id = {item.item_id: item for item in remaining}
        index_by_id = {item_id: index for index, item_id in enumerate(item_ids)}
        relevance_values = np.asarray(
            [relevance[item_id] for item_id in item_ids],
            dtype=np.float64,
        )
        maximum_similarity = np.zeros(len(item_ids), dtype=np.float64)
        active = np.ones(len(item_ids), dtype=bool)
        selected = []
        while active.any() and len(selected) < k:
            scores = self.alpha * relevance_values - (1 - self.alpha) * maximum_similarity
            scores[~active] = -np.inf
            best_index = int(np.argmax(scores))
            best_item = item_by_id[item_ids[best_index]]
            best_score = float(scores[best_index])
            selected.append(
                RankedRecommendation(
                    item_id=best_item.item_id,
                    score=float(best_score),
                    sources=best_item.sources,
                )
            )
            active[best_index] = False
            if callable(self.similarities):
                for item_index, item_id in enumerate(item_ids):
                    if active[item_index]:
                        maximum_similarity[item_index] = max(
                            maximum_similarity[item_index],
                            item_similarity(item_id, best_item.item_id, self.similarities),
                        )
            else:
                # Item-KNN co-visitation similarities are symmetric and sparse, so
                # only actual neighbors can change the MMR penalty.
                for neighbor_id, similarity in self.similarities.get(
                    best_item.item_id,
                    {},
                ).items():
                    neighbor_index = index_by_id.get(neighbor_id)
                    if neighbor_index is not None and active[neighbor_index]:
                        maximum_similarity[neighbor_index] = max(
                            maximum_similarity[neighbor_index],
                            float(similarity),
                        )
        return selected


def intra_list_similarity_at_k(
    recommendations: Sequence[object],
    similarities: SimilaritySource,
    k: int,
) -> float:
    items = [_item_id(item) for item in recommendations[:k]]
    pairs = list(combinations(items, 2))
    if not pairs:
        return 0.0
    return sum(item_similarity(first, second, similarities) for first, second in pairs) / len(
        pairs
    )


def unique_recommended_items_at_k(
    predictions: Sequence[Sequence[object]],
    k: int,
) -> int:
    return len(
        {
            _item_id(item)
            for recommendation_list in predictions
            for item in recommendation_list[:k]
        }
    )


def category_count_at_k(
    recommendations: Sequence[object],
    category_mapping: Mapping[object, object],
    k: int,
) -> int:
    return len(
        {
            category_mapping[item_id]
            for item_id in (_item_id(item) for item in recommendations[:k])
            if item_id in category_mapping
        }
    )
