import json
from pathlib import Path

import pandas as pd
import torch

from baselines.item_knn import ItemKNNRecommender
from baselines.markov import MarkovRecommender
from baselines.popularity import PopularityRecommender
from evaluation.candidate_gru_evaluate import (
    CONTROL_NAME,
    MAIN_NAME,
    MMR_NAME,
    evaluate_on_shared_candidate_pools,
    validation_gate_passes,
)
from models.candidate_gru import CandidateGRU
from ranking.candidate_generation import CandidateGenerator


def _validation_payload(
    control_mrr,
    candidate_mrr,
    checkpoint_hash="checkpoint",
    config_hash="config",
):
    return {
        "metadata": {
            "candidate_gru_checkpoint_sha256": checkpoint_hash,
            "candidate_gru_config_sha256": config_hash,
        },
        "models": {
            CONTROL_NAME: {"metrics": {"20": {"mrr": control_mrr}}},
            MAIN_NAME: {"metrics": {"20": {"mrr": candidate_mrr}}},
        },
    }


def test_validation_gate_requires_strict_mrr_improvement_and_matching_hashes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate_gru_validation.json"
    path.write_text(json.dumps(_validation_payload(0.1, 0.11)), encoding="utf-8")
    assert validation_gate_passes(tmp_path, "checkpoint", "config")[0] is True

    path.write_text(json.dumps(_validation_payload(0.1, 0.1)), encoding="utf-8")
    assert validation_gate_passes(tmp_path, "checkpoint", "config")[0] is False

    path.write_text(json.dumps(_validation_payload(0.1, 0.11)), encoding="utf-8")
    assert validation_gate_passes(tmp_path, "wrong", "config")[0] is False


def test_controlled_evaluation_generates_one_shared_pool_per_example() -> None:
    sessions = [[1, 2, 3], [1, 4, 5], [2, 6, 7], [8, 9, 10]]
    popularity = PopularityRecommender().fit(sessions)
    markov = MarkovRecommender().fit(sessions)
    item_knn = ItemKNNRecommender().fit(sessions)
    generator = CandidateGenerator(
        popularity,
        markov,
        item_knn,
        candidate_pool_size=8,
        candidate_mode="quota_combined",
    )
    original_generate = generator.generate
    calls = []

    def counted_generate(prefix_items):
        calls.append(tuple(prefix_items))
        return original_generate(prefix_items)

    generator.generate = counted_generate
    current_model = CandidateGRU(num_items=11, embedding_dim=4, hidden_dim=4)
    candidate_model = CandidateGRU(num_items=11, embedding_dim=4, hidden_dim=4)
    examples = pd.DataFrame(
        {"prefix_items": [[1], [2]], "target_item": [3, 6]}
    )

    results = evaluate_on_shared_candidate_pools(
        current_model,
        candidate_model,
        generator,
        examples,
        catalog_size=10,
        device=torch.device("cpu"),
        mmr_alpha=0.8,
        batch_size=2,
    )

    assert calls == [(1,), (2,)]
    assert list(results) == [CONTROL_NAME, MAIN_NAME, MMR_NAME]
    assert all(result["latency_mode"] == "batched_throughput" for result in results.values())
