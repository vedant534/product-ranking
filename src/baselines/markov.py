"""First-order train-session transition recommendation baseline."""

from collections import Counter, defaultdict
from collections.abc import Sequence

from baselines.popularity import (
    PopularityRecommender,
    TrainData,
    _item_sort_key,
    _validate_k,
    prepare_train_sessions,
)


class MarkovRecommender:
    """Rank candidates by transitions from the final prefix item."""

    def __init__(self) -> None:
        self.transition_counts: dict[object, Counter[object]] = {}
        self.popularity = PopularityRecommender()
        self._popularity_rank: dict[object, int] = {}
        self._is_fitted = False

    def fit(self, train_data: TrainData) -> "MarkovRecommender":
        sessions = prepare_train_sessions(train_data)
        transitions: defaultdict[object, Counter[object]] = defaultdict(Counter)
        for session in sessions:
            for current_item, next_item in zip(session, session[1:]):
                transitions[current_item][next_item] += 1

        self.transition_counts = dict(transitions)
        self.popularity.fit(sessions)
        self._popularity_rank = {
            item: rank for rank, item in enumerate(self.popularity.ranked_items)
        }
        self._is_fitted = True
        return self

    def recommend(self, prefix_items: Sequence[object], k: int) -> list[object]:
        if not self._is_fitted:
            raise RuntimeError("MarkovRecommender must be fitted before recommendation")
        cutoff = _validate_k(k)
        prefix = list(prefix_items)
        recommendations = []

        candidate_scores = self.score_candidates(prefix)
        if candidate_scores:
            candidates = sorted(
                candidate_scores,
                key=lambda item: (
                    -candidate_scores[item],
                    self._popularity_rank.get(item, len(self._popularity_rank)),
                    _item_sort_key(item),
                ),
            )
            recommendations.extend(candidates[:cutoff])

        if len(recommendations) < cutoff:
            excluded = prefix + recommendations
            for item in self.popularity.recommend(excluded, cutoff):
                if item not in recommendations:
                    recommendations.append(item)
                if len(recommendations) == cutoff:
                    break
        return recommendations

    def score_candidates(self, prefix_items: Sequence[object]) -> dict[object, float]:
        """Return raw first-order transition counts from the final prefix item."""
        if not self._is_fitted:
            raise RuntimeError("MarkovRecommender must be fitted before scoring")
        prefix = list(prefix_items)
        if not prefix or prefix[-1] not in self.transition_counts:
            return {}
        seen = set(prefix)
        return {
            item: float(count)
            for item, count in self.transition_counts[prefix[-1]].items()
            if item not in seen
        }
