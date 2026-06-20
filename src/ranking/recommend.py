"""Common candidate-scoring and top-k recommendation interface."""

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Protocol

from baselines.item_knn import ItemKNNRecommender
from baselines.markov import MarkovRecommender
from baselines.popularity import PopularityRecommender, _item_sort_key
from ranking.candidate_generation import SOURCE_ORDER, Candidate, CandidateGenerator


@dataclass(frozen=True)
class RankedRecommendation:
    item_id: object
    score: float
    sources: tuple[str, ...]

    @property
    def source(self) -> str:
        return ",".join(self.sources)


class NeuralScorer(Protocol):
    def score_items(
        self,
        prefix_items: list[int],
        candidate_items: list[int],
    ) -> dict[int, float]:
        ...


class Reranker(Protocol):
    def rerank(
        self,
        recommendations: list[RankedRecommendation],
        k: int,
        seen_items: Optional[set[object]] = None,
    ) -> list[RankedRecommendation]:
        ...


class RecommendationEngine:
    """Generate one shared pool and score it with a selected fitted model."""

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        popularity: PopularityRecommender,
        markov: MarkovRecommender,
        item_knn: ItemKNNRecommender,
        neural_model: Optional[NeuralScorer] = None,
    ) -> None:
        self.candidate_generator = candidate_generator
        self.popularity = popularity
        self.markov = markov
        self.item_knn = item_knn
        self.neural_model = neural_model

    @staticmethod
    def _validate_k(k: int) -> int:
        if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
            raise ValueError("k must be a positive integer")
        return int(k)

    @staticmethod
    def _sources(candidate: Candidate, selected_source: str) -> tuple[str, ...]:
        sources = set(candidate.sources)
        sources.add(selected_source)
        return tuple(source for source in SOURCE_ORDER if source in sources)

    def _classical_scores(
        self,
        prefix_items: list[object],
        candidates: list[Candidate],
        model_name: str,
    ) -> tuple[dict[object, float], dict[object, float]]:
        candidate_ids = [candidate.item_id for candidate in candidates]
        popularity_scores = self.popularity.score_items(candidate_ids)
        if model_name == "popularity":
            return popularity_scores, popularity_scores
        if model_name == "markov":
            return self.markov.score_candidates(prefix_items), popularity_scores
        return self.item_knn.score_candidates(prefix_items), popularity_scores

    def recommend(
        self,
        prefix_items: list[object],
        model_name: str,
        k: int,
    ) -> list[RankedRecommendation]:
        cutoff = self._validate_k(k)
        normalized_name = "gru" if model_name in {"gru", "gru4rec", "neural"} else model_name
        if normalized_name not in {"popularity", "markov", "item_knn", "gru"}:
            raise ValueError(f"Unknown model_name: {model_name}")

        seen = set(prefix_items)
        candidates = [
            candidate
            for candidate in self.candidate_generator.generate(prefix_items)
            if candidate.item_id not in seen
        ]
        pool_rank = {candidate.item_id: rank for rank, candidate in enumerate(candidates)}

        if normalized_name == "gru":
            if self.neural_model is None:
                raise RuntimeError("A neural model is required for GRU scoring")
            candidate_ids = [int(candidate.item_id) for candidate in candidates]
            scores = self.neural_model.score_items(
                [int(item) for item in prefix_items],
                candidate_ids,
            )
            ranked = sorted(
                (candidate for candidate in candidates if candidate.item_id in scores),
                key=lambda candidate: (
                    -scores[candidate.item_id],
                    pool_rank[candidate.item_id],
                    _item_sort_key(candidate.item_id),
                ),
            )
            return [
                RankedRecommendation(
                    item_id=candidate.item_id,
                    score=float(scores[candidate.item_id]),
                    sources=self._sources(candidate, "neural"),
                )
                for candidate in ranked[:cutoff]
            ]

        primary_scores, popularity_scores = self._classical_scores(
            prefix_items,
            candidates,
            normalized_name,
        )
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                0 if candidate.item_id in primary_scores else 1,
                -primary_scores.get(
                    candidate.item_id,
                    popularity_scores.get(candidate.item_id, 0.0),
                ),
                pool_rank[candidate.item_id],
                _item_sort_key(candidate.item_id),
            ),
        )
        output = []
        for candidate in ranked[:cutoff]:
            selected_source = (
                normalized_name if candidate.item_id in primary_scores else "popularity"
            )
            score = primary_scores.get(
                candidate.item_id,
                popularity_scores.get(candidate.item_id, 0.0),
            )
            output.append(
                RankedRecommendation(
                    item_id=candidate.item_id,
                    score=float(score),
                    sources=self._sources(candidate, selected_source),
                )
            )
        return output


class RecommendationAdapter:
    """Expose engine recommendations as item IDs for the unified evaluator."""

    def __init__(
        self,
        engine: RecommendationEngine,
        model_name: str,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.engine = engine
        self.model_name = model_name
        self.reranker = reranker

    def recommend_with_scores(
        self,
        prefix_items: list[int],
        k: int,
    ) -> list[RankedRecommendation]:
        requested = (
            self.engine.candidate_generator.candidate_pool_size
            if self.reranker is not None
            else k
        )
        recommendations = self.engine.recommend(prefix_items, self.model_name, requested)
        if self.reranker is not None:
            return self.reranker.rerank(recommendations, k, seen_items=set(prefix_items))
        return recommendations[:k]

    def recommend(self, prefix_items: list[int], k: int) -> list[int]:
        return [
            int(recommendation.item_id)
            for recommendation in self.recommend_with_scores(prefix_items, k)
        ]
