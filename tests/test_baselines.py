"""Unit tests for train-only classical recommendation baselines."""

import pandas as pd

from baselines.item_knn import ItemKNNRecommender
from baselines.markov import MarkovRecommender
from baselines.popularity import PopularityRecommender
from scripts.run_baselines import reconstruct_train_sessions


def test_popularity_ranks_most_frequent_item_first() -> None:
    model = PopularityRecommender().fit([[1, 2], [1, 3], [1, 4]])

    assert model.recommend([], k=3)[0] == 1


def test_popularity_excludes_seen_items() -> None:
    model = PopularityRecommender().fit([[1, 2], [1, 3], [1, 4]])

    recommendations = model.recommend([1, 3], k=10)

    assert 1 not in recommendations
    assert 3 not in recommendations
    assert recommendations == [2, 4]


def test_popularity_returns_available_items_when_k_is_too_large() -> None:
    model = PopularityRecommender().fit([[1, 2]])

    assert model.recommend([1], k=20) == [2]


def test_dataframe_fit_uses_train_interactions_only() -> None:
    interactions = pd.DataFrame(
        {
            "session_id": [1, 1, 2, 2, 3, 3],
            "session_position": [0, 1, 0, 1, 0, 1],
            "itemid": [1, 2, 99, 99, 99, 99],
            "split": ["train", "train", "validation", "validation", "test", "test"],
        }
    )

    model = PopularityRecommender().fit(interactions)

    assert set(model.item_counts) == {1, 2}
    assert 99 not in model.recommend([], k=10)


def test_markov_recommends_observed_next_items() -> None:
    model = MarkovRecommender().fit([["A", "B", "C"], ["A", "B", "D"]])

    recommendations = model.recommend(["A", "B"], k=2)

    assert recommendations == ["C", "D"]


def test_markov_excludes_seen_items_and_backfills_popularity() -> None:
    model = MarkovRecommender().fit([[1, 2, 1], [1, 2, 3]])

    recommendations = model.recommend([1, 2], k=3)

    assert 1 not in recommendations
    assert 2 not in recommendations
    assert recommendations == [3]


def test_markov_falls_back_for_unknown_last_item() -> None:
    model = MarkovRecommender().fit([[1, 2], [1, 3]])

    assert model.recommend([999], k=2) == model.popularity.recommend([999], k=2)


def test_item_knn_recommends_cooccurring_item() -> None:
    model = ItemKNNRecommender(similarity="raw").fit([[1, 2, 3], [1, 2]])

    assert model.recommend([1], k=1) == [2]


def test_item_knn_supports_cosine_similarity_and_excludes_seen_items() -> None:
    model = ItemKNNRecommender(similarity="cosine").fit([[1, 2, 3], [1, 2]])

    recommendations = model.recommend([1, 2], k=3)

    assert 1 not in recommendations
    assert 2 not in recommendations
    assert recommendations == [3]


def test_item_knn_gives_more_weight_to_recent_prefix_items() -> None:
    model = ItemKNNRecommender(similarity="raw", recency_decay=0.5).fit(
        [[1, 10], [2, 20]]
    )

    recommendations = model.recommend([1, 2], k=2)

    assert recommendations == [20, 10]


def test_item_knn_falls_back_for_unknown_items() -> None:
    model = ItemKNNRecommender().fit([[1, 2], [1, 3]])

    assert model.recommend([999], k=2) == model.popularity.recommend([999], k=2)


def test_runner_reconstructs_train_sessions_without_counting_repeated_prefixes() -> None:
    train_examples = pd.DataFrame(
        {
            "session_id": [1, 1, 2],
            "prefix_items": [[10], [10, 20], [40]],
            "target_item": [20, 30, 50],
        }
    )

    assert reconstruct_train_sessions(train_examples) == [[10, 20, 30], [40, 50]]
