"""Persistence helpers for trusted, locally fitted baseline recommenders."""

import pickle
from pathlib import Path
from typing import Optional

from baselines.item_knn import ItemKNNRecommender
from baselines.markov import MarkovRecommender
from baselines.popularity import PopularityRecommender

BASELINE_ARTIFACT_VERSION = 1


def save_baseline_bundle(
    path: Path,
    popularity: PopularityRecommender,
    markov: MarkovRecommender,
    item_knn: ItemKNNRecommender,
    catalog_size: int,
) -> Path:
    """Persist already fitted baseline models for read-only serving."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": BASELINE_ARTIFACT_VERSION,
        "catalog_size": int(catalog_size),
        "models": {
            "popularity": popularity,
            "markov": markov,
            "item_knn": item_knn,
        },
    }
    with path.open("wb") as artifact_file:
        pickle.dump(payload, artifact_file, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_baseline_bundle(
    path: Path,
    expected_catalog_size: Optional[int] = None,
) -> tuple[PopularityRecommender, MarkovRecommender, ItemKNNRecommender]:
    """Load and validate a trusted baseline bundle created by this project."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing baseline artifact '{path}'. Run scripts/build_demo_artifacts.py first."
        )
    with path.open("rb") as artifact_file:
        payload = pickle.load(artifact_file)

    if (
        not isinstance(payload, dict)
        or payload.get("artifact_version") != BASELINE_ARTIFACT_VERSION
    ):
        raise ValueError(f"Unsupported or invalid baseline artifact: '{path}'")
    if (
        expected_catalog_size is not None
        and payload.get("catalog_size") != expected_catalog_size
    ):
        raise ValueError(
            "Baseline artifact catalog size does not match the processed item mapping. "
            "Rebuild it with scripts/build_demo_artifacts.py."
        )

    models = payload.get("models", {})
    popularity = models.get("popularity")
    markov = models.get("markov")
    item_knn = models.get("item_knn")
    if not isinstance(popularity, PopularityRecommender):
        raise ValueError("Baseline artifact does not contain a valid popularity model")
    if not isinstance(markov, MarkovRecommender):
        raise ValueError("Baseline artifact does not contain a valid Markov model")
    if not isinstance(item_knn, ItemKNNRecommender):
        raise ValueError("Baseline artifact does not contain a valid Item-KNN model")
    return popularity, markov, item_knn
