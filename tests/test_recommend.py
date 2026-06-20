"""Tests for hybrid candidate generation and the common recommendation interface."""

import pytest

from baselines.item_knn import ItemKNNRecommender
from baselines.markov import MarkovRecommender
from baselines.popularity import PopularityRecommender
from ranking.candidate_generation import CandidateGenerator
from ranking.recommend import RecommendationEngine


class IncreasingItemNeuralScorer:
    def score_items(
        self,
        prefix_items: list[int],
        candidate_items: list[int],
    ) -> dict[int, float]:
        return {item: float(item) for item in candidate_items}


@pytest.fixture
def fitted_models() -> tuple[
    PopularityRecommender,
    MarkovRecommender,
    ItemKNNRecommender,
]:
    sessions = [[1, 2, 3], [1, 4, 5], [2, 6, 7], [8, 9, 10]]
    return (
        PopularityRecommender().fit(sessions),
        MarkovRecommender().fit(sessions),
        ItemKNNRecommender().fit(sessions),
    )


def _engine(
    fitted_models: tuple[
        PopularityRecommender,
        MarkovRecommender,
        ItemKNNRecommender,
    ],
    pool_size: int = 8,
) -> RecommendationEngine:
    popularity, markov, item_knn = fitted_models
    generator = CandidateGenerator(
        popularity,
        markov,
        item_knn,
        candidate_pool_size=pool_size,
    )
    return RecommendationEngine(
        generator,
        popularity,
        markov,
        item_knn,
        neural_model=IncreasingItemNeuralScorer(),
    )


def test_seen_items_are_excluded(fitted_models: tuple[object, object, object]) -> None:
    engine = _engine(fitted_models)

    recommendations = engine.recommend([1, 2], model_name="item_knn", k=5)

    assert all(item.item_id not in {1, 2} for item in recommendations)


def test_candidate_pool_size_is_respected(
    fitted_models: tuple[object, object, object],
) -> None:
    engine = _engine(fitted_models, pool_size=3)

    candidates = engine.candidate_generator.generate([1])

    assert len(candidates) == 3


def test_output_length_is_k_when_enough_candidates_exist(
    fitted_models: tuple[object, object, object],
) -> None:
    engine = _engine(fitted_models, pool_size=8)

    recommendations = engine.recommend([1], model_name="markov", k=5)

    assert len(recommendations) == 5
    assert all(recommendation.sources for recommendation in recommendations)


def test_gru_scores_candidates_and_adds_neural_source_tag(
    fitted_models: tuple[object, object, object],
) -> None:
    engine = _engine(fitted_models, pool_size=8)

    recommendations = engine.recommend([1], model_name="gru", k=3)

    assert [item.item_id for item in recommendations] == sorted(
        (item.item_id for item in recommendations), reverse=True
    )
    assert all("neural" in item.sources for item in recommendations)


def test_native_source_scores_do_not_include_serving_fallback(
    fitted_models: tuple[object, object, object],
) -> None:
    engine = _engine(fitted_models)

    source_scores = engine.candidate_generator.score_sources([999], pool_size=5)

    assert source_scores["popularity"]
    assert source_scores["markov"] == {}
    assert source_scores["item_knn"] == {}


def test_precomputed_sources_can_be_merged_to_an_exact_smaller_pool(
    fitted_models: tuple[object, object, object],
) -> None:
    engine = _engine(fitted_models, pool_size=8)
    source_scores = engine.candidate_generator.score_sources([1], pool_size=8)

    candidates = engine.candidate_generator.combine_source_scores(
        [1], source_scores, pool_size=3
    )

    assert len(candidates) == 3
    assert all(candidate.item_id != 1 for candidate in candidates)
    for target_item in range(1, 11):
        expected = any(candidate.item_id == target_item for candidate in candidates)
        assert engine.candidate_generator.combined_contains_target(
            source_scores,
            target_item,
            pool_size=3,
        ) is expected
