"""Correctness, cache, and overfit tests for the isolated CandidateGRU."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.candidate_gru import CandidateGRU
from models.train_candidate_gru import (
    CandidateGRUTrainingConfig,
    build_hard_negative_cache,
    cache_signature,
    make_candidate_collate_fn,
    sample_hard_negatives,
    train_candidate_gru_model,
)
from ranking.candidate_generation import Candidate


class FixedGenerator:
    def generate(self, prefix_items):
        return [Candidate(item_id=item) for item in (2, 3, 4, 5) if item not in prefix_items]


def test_candidate_logits_mask_and_positive_column() -> None:
    model = CandidateGRU(num_items=6, embedding_dim=4, hidden_dim=4, dropout=0.0)
    input_ids = torch.tensor([[1], [2]], dtype=torch.long)
    lengths = torch.tensor([1, 1], dtype=torch.long)
    candidates = torch.tensor([[2, 3, 0], [3, 4, 5]], dtype=torch.long)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    logits = model.candidate_logits(input_ids, lengths, candidates, mask)

    assert logits.shape == (2, 3)
    assert torch.isneginf(logits[0, 2])
    batch = [([1], 2, [3, 4]), ([2], 3, [4, 5])]
    _, _, candidate_ids, candidate_mask = make_candidate_collate_fn(
        model.padding_idx, 10
    )(batch)
    assert candidate_ids[:, 0].tolist() == [2, 3]
    assert candidate_mask[:, 0].all()


def test_hard_negatives_are_deterministic_and_exclude_positive_and_prefix() -> None:
    first = sample_hard_negatives([1, 2, 3, 4, 5], [1, 2], 3, 2, seed=9)
    second = sample_hard_negatives([1, 2, 3, 4, 5], [1, 2], 3, 2, seed=9)

    assert first == second
    assert set(first).isdisjoint({0, 1, 2, 3})


def test_hard_negative_cache_reuses_matching_and_rebuilds_stale_signature(
    tmp_path: Path,
) -> None:
    examples = pd.DataFrame({"prefix_items": [[1], [1]], "target_item": [2, 2]})
    config = CandidateGRUTrainingConfig(hard_candidate_negatives=2, candidate_pool_size=4)

    _, _, first = build_hard_negative_cache(
        FixedGenerator(),
        examples,
        config,
        tmp_path,
        "signature-a",
        progress_every=0,
        identity_metadata={"source_examples_sha256": "source-a"},
    )
    _, _, reused = build_hard_negative_cache(
        FixedGenerator(),
        examples,
        config,
        tmp_path,
        "signature-a",
        progress_every=0,
        identity_metadata={"source_examples_sha256": "source-a"},
    )
    _, _, rebuilt = build_hard_negative_cache(
        FixedGenerator(),
        examples,
        config,
        tmp_path,
        "signature-b",
        progress_every=0,
        identity_metadata={"source_examples_sha256": "source-b"},
    )

    assert first["signature"] == reused["signature"] == "signature-a"
    assert reused["source_examples_sha256"] == "source-a"
    assert rebuilt["signature"] == "signature-b"
    assert rebuilt["source_examples_sha256"] == "source-b"
    assert cache_signature("a", "b", config, "train", 2) != cache_signature(
        "a", "c", config, "train", 2
    )
    assert cache_signature(
        "a", "b", config, "validation", 2, source_examples_hash="train-a"
    ) != cache_signature(
        "a", "b", config, "validation", 2, source_examples_hash="train-b"
    )


def test_candidate_gru_overfits_tiny_hard_negative_task(tmp_path: Path) -> None:
    examples = pd.DataFrame(
        {
            "prefix_items": [[1]] * 32 + [[2]] * 32,
            "target_item": [2] * 32 + [3] * 32,
        }
    )
    negatives = np.asarray([[3, 4]] * 32 + [[1, 4]] * 32, dtype=np.int32)
    negative_lengths = np.full(len(examples), 2, dtype=np.uint16)
    validation = examples.iloc[[0, 32]].reset_index(drop=True)
    validation_candidates = np.asarray([[2, 3, 4], [3, 1, 4]], dtype=np.int32)
    validation_lengths = np.asarray([3, 3], dtype=np.uint16)
    config = CandidateGRUTrainingConfig(
        embedding_dim=8,
        hidden_dim=8,
        dropout=0.0,
        batch_size=16,
        learning_rate=0.05,
        epochs=12,
        early_stopping_patience=12,
        device="cpu",
    )
    checkpoint = tmp_path / "candidate.pt"

    model, history = train_candidate_gru_model(
        examples,
        validation,
        negatives,
        negative_lengths,
        validation_candidates,
        validation_lengths,
        num_items=5,
        config=config,
        checkpoint_path=checkpoint,
        log_path=tmp_path / "log.json",
        cache_metadata={},
    )
    loaded = CandidateGRU.load_checkpoint(checkpoint)

    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert loaded.checkpoint_metadata["best_validation_metrics"]["mrr"] == 1.0
    assert model.recommend_from_candidates([1], [2, 3, 4], 1) == [2]
