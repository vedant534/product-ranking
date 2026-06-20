"""Hybrid candidate generation from fitted classical recommenders."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional

from baselines.item_knn import ItemKNNRecommender
from baselines.markov import MarkovRecommender
from baselines.popularity import PopularityRecommender, _item_sort_key

SOURCE_ORDER = ("markov", "item_knn", "popularity", "neural")


@dataclass
class Candidate:
    item_id: object
    source_scores: dict[str, float] = field(default_factory=dict)
    pool_score: float = 0.0

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(source for source in SOURCE_ORDER if source in self.source_scores)


class CandidateGenerator:
    """Merge Markov, Item-KNN, and popularity candidates into a bounded pool."""

    def __init__(
        self,
        popularity: PopularityRecommender,
        markov: MarkovRecommender,
        item_knn: ItemKNNRecommender,
        candidate_pool_size: int = 500,
    ) -> None:
        if (
            isinstance(candidate_pool_size, bool)
            or not isinstance(candidate_pool_size, Integral)
            or candidate_pool_size <= 0
        ):
            raise ValueError("candidate_pool_size must be a positive integer")
        self.popularity = popularity
        self.markov = markov
        self.item_knn = item_knn
        self.candidate_pool_size = int(candidate_pool_size)
        self._popularity_rank = {
            item: rank for rank, item in enumerate(self.popularity.ranked_items)
        }

    @staticmethod
    def _top_scores(scores: dict[object, float], limit: int) -> dict[object, float]:
        ordered = sorted(
            scores,
            key=lambda item: (-scores[item], _item_sort_key(item)),
        )
        return {item: scores[item] for item in ordered[:limit]}

    def _resolve_pool_size(self, pool_size: Optional[int]) -> int:
        resolved = self.candidate_pool_size if pool_size is None else pool_size
        if isinstance(resolved, bool) or not isinstance(resolved, Integral) or resolved <= 0:
            raise ValueError("pool_size must be a positive integer")
        return int(resolved)

    def score_sources(
        self,
        prefix_items: list[object],
        pool_size: Optional[int] = None,
    ) -> dict[str, dict[object, float]]:
        """Return ranked native-source scores without cross-source fallback."""
        limit = self._resolve_pool_size(pool_size)
        popularity_items = self.popularity.recommend(
            prefix_items,
            limit,
        )
        return {
            "popularity": self.popularity.score_items(popularity_items),
            "markov": self._top_scores(
                self.markov.score_candidates(prefix_items),
                limit,
            ),
            "item_knn": self._top_scores(
                self.item_knn.score_candidates(prefix_items),
                limit,
            ),
        }

    def combine_source_scores(
        self,
        prefix_items: list[object],
        source_scores: dict[str, dict[object, float]],
        pool_size: Optional[int] = None,
    ) -> list[Candidate]:
        """Apply the serving merge to native source scores and bound the final pool."""
        limit = self._resolve_pool_size(pool_size)
        seen = set(prefix_items)
        candidates: dict[object, Candidate] = {}
        for source in ("popularity", "markov", "item_knn"):
            scores = self._top_scores(source_scores.get(source, {}), limit)
            maximum = max(scores.values(), default=0.0)
            for item_id, score in scores.items():
                if item_id in seen:
                    continue
                candidate = candidates.setdefault(item_id, Candidate(item_id=item_id))
                candidate.source_scores[source] = float(score)
                if maximum > 0:
                    candidate.pool_score += float(score) / maximum

        ordered_candidates = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.pool_score,
                self._popularity_rank.get(
                    candidate.item_id,
                    len(self._popularity_rank),
                ),
                _item_sort_key(candidate.item_id),
            ),
        )
        return ordered_candidates[:limit]

    def generate(self, prefix_items: list[object]) -> list[Candidate]:
        source_scores = self.score_sources(prefix_items)
        return self.combine_source_scores(prefix_items, source_scores)
