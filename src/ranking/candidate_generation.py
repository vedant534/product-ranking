"""Hybrid candidate generation from fitted classical recommenders."""

from dataclasses import dataclass, field
from heapq import nsmallest
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
        def rank_key(item: object) -> tuple[float, tuple[str, str]]:
            return -scores[item], _item_sort_key(item)

        if len(scores) <= limit:
            ordered = sorted(scores, key=rank_key)
        else:
            ordered = nsmallest(limit, scores, key=rank_key)
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
            scores = dict(list(source_scores.get(source, {}).items())[:limit])
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

    def combined_contains_target(
        self,
        source_scores: dict[str, dict[object, float]],
        target_item: object,
        pool_size: int,
    ) -> bool:
        """Check target membership in an exact bounded merge without sorting the full union.

        ``source_scores`` must be the ranked output of :meth:`score_sources`.
        """
        limit = self._resolve_pool_size(pool_size)
        ranked_sources = {
            source: list(source_scores.get(source, {}).items())[:limit]
            for source in ("popularity", "markov", "item_knn")
        }
        if not any(
            any(item_id == target_item for item_id, _ in scores)
            for scores in ranked_sources.values()
        ):
            return False

        pool_scores: dict[object, float] = {}
        for scores in ranked_sources.values():
            maximum = max((score for _, score in scores), default=0.0)
            for item_id, score in scores:
                if maximum > 0:
                    pool_scores[item_id] = pool_scores.get(item_id, 0.0) + score / maximum
                else:
                    pool_scores.setdefault(item_id, 0.0)

        target_score = pool_scores[target_item]
        target_key = (
            -target_score,
            self._popularity_rank.get(target_item, len(self._popularity_rank)),
            _item_sort_key(target_item),
        )
        better_items = 0
        for item_id, pool_score in pool_scores.items():
            item_key = (
                -pool_score,
                self._popularity_rank.get(item_id, len(self._popularity_rank)),
                _item_sort_key(item_id),
            )
            if item_key < target_key:
                better_items += 1
                if better_items >= limit:
                    return False
        return True

    def generate(self, prefix_items: list[object]) -> list[Candidate]:
        source_scores = self.score_sources(prefix_items)
        return self.combine_source_scores(prefix_items, source_scores)
