from pathlib import Path

import pytest

from baselines.item_knn import ItemKNNRecommender
from baselines.markov import MarkovRecommender
from baselines.popularity import PopularityRecommender
from utils.model_artifacts import load_baseline_bundle, save_baseline_bundle


def _fitted_models() -> tuple[
    PopularityRecommender,
    MarkovRecommender,
    ItemKNNRecommender,
]:
    sessions = [[1, 2, 3], [1, 2, 4]]
    return (
        PopularityRecommender().fit(sessions),
        MarkovRecommender().fit(sessions),
        ItemKNNRecommender().fit(sessions),
    )


def test_baseline_bundle_round_trip_preserves_recommendations(tmp_path: Path) -> None:
    popularity, markov, item_knn = _fitted_models()
    artifact_path = save_baseline_bundle(
        tmp_path / "baselines.pkl",
        popularity,
        markov,
        item_knn,
        catalog_size=4,
    )

    loaded = load_baseline_bundle(artifact_path, expected_catalog_size=4)

    assert loaded[0].recommend([1], 2) == popularity.recommend([1], 2)
    assert loaded[1].recommend([1, 2], 2) == markov.recommend([1, 2], 2)
    assert loaded[2].recommend([1], 2) == item_knn.recommend([1], 2)


def test_baseline_bundle_rejects_catalog_mismatch(tmp_path: Path) -> None:
    artifact_path = save_baseline_bundle(
        tmp_path / "baselines.pkl",
        *_fitted_models(),
        catalog_size=4,
    )

    with pytest.raises(ValueError, match="catalog size"):
        load_baseline_bundle(artifact_path, expected_catalog_size=5)
