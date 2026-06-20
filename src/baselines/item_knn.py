"""Train-session item co-visitation recommendation baseline."""

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import combinations

from baselines.popularity import (
    PopularityRecommender,
    TrainData,
    _item_sort_key,
    _validate_k,
    prepare_train_sessions,
)


class ItemKNNRecommender:
    """Aggregate recency-weighted item-neighbor scores from a session prefix."""

    def __init__(self, similarity: str = "cosine", recency_decay: float = 0.8) -> None:
        if similarity not in {"raw", "cosine"}:
            raise ValueError("similarity must be 'raw' or 'cosine'")
        if not 0 < recency_decay <= 1:
            raise ValueError("recency_decay must be in the interval (0, 1]")
        self.similarity = similarity
        self.recency_decay = float(recency_decay)
        self.neighbor_scores: dict[object, dict[object, float]] = {}
        self.popularity = PopularityRecommender()
        self._popularity_rank: dict[object, int] = {}
        self._is_fitted = False

    def fit(self, train_data: TrainData) -> "ItemKNNRecommender":
        sessions = prepare_train_sessions(train_data)
        session_frequency: Counter[object] = Counter()
        cooccurrence: defaultdict[object, Counter[object]] = defaultdict(Counter)

        for session in sessions:
            unique_items = list(dict.fromkeys(session))
            session_frequency.update(unique_items)
            for first_item, second_item in combinations(unique_items, 2):
                cooccurrence[first_item][second_item] += 1
                cooccurrence[second_item][first_item] += 1

        neighbor_scores = {}
        for item, neighbors in cooccurrence.items():
            item_scores = {}
            for neighbor, count in neighbors.items():
                if self.similarity == "raw":
                    score = float(count)
                else:
                    denominator = math.sqrt(
                        session_frequency[item] * session_frequency[neighbor]
                    )
                    score = count / denominator
                item_scores[neighbor] = score
            neighbor_scores[item] = item_scores

        self.neighbor_scores = neighbor_scores
        self.popularity.fit(sessions)
        self._popularity_rank = {
            item: rank for rank, item in enumerate(self.popularity.ranked_items)
        }
        self._is_fitted = True
        return self

    def recommend(self, prefix_items: Sequence[object], k: int) -> list[object]:
        if not self._is_fitted:
            raise RuntimeError("ItemKNNRecommender must be fitted before recommendation")
        cutoff = _validate_k(k)
        prefix = list(prefix_items)
        candidate_scores = self.score_candidates(prefix)

        recommendations = sorted(
            candidate_scores,
            key=lambda item: (
                -candidate_scores[item],
                self._popularity_rank.get(item, len(self._popularity_rank)),
                _item_sort_key(item),
            ),
        )[:cutoff]

        if len(recommendations) < cutoff:
            excluded = prefix + recommendations
            for item in self.popularity.recommend(excluded, cutoff):
                if item not in recommendations:
                    recommendations.append(item)
                if len(recommendations) == cutoff:
                    break
        return recommendations

    def score_candidates(self, prefix_items: Sequence[object]) -> dict[object, float]:
        """Return recency-weighted aggregate neighbor scores for unseen candidates."""
        if not self._is_fitted:
            raise RuntimeError("ItemKNNRecommender must be fitted before scoring")
        prefix = list(prefix_items)
        seen = set(prefix)
        candidate_scores: defaultdict[object, float] = defaultdict(float)
        for distance, item in enumerate(reversed(prefix)):
            weight = self.recency_decay**distance
            for neighbor, similarity in self.neighbor_scores.get(item, {}).items():
                if neighbor not in seen:
                    candidate_scores[neighbor] += weight * similarity
        return dict(candidate_scores)
