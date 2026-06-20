"""Correctness and tiny-overfit tests for the GRU4Rec-style model."""

from pathlib import Path

import pandas as pd
import torch

from models.gru4rec import GRU4Rec
from models.predict import load_gru_model
from models.train_gru import (
    GRUTrainingConfig,
    sampled_training_logits,
    train_gru_model,
)


def test_forward_shape_and_recommendations_exclude_padding() -> None:
    model = GRU4Rec(
        num_items=5,
        embedding_dim=8,
        hidden_dim=12,
        padding_idx=5,
        dropout=0.0,
    )
    input_ids = torch.tensor([[1, 2], [3, 5]], dtype=torch.long)
    lengths = torch.tensor([2, 1], dtype=torch.long)

    logits = model(input_ids, lengths)
    recommendations = model.recommend([1, 2], k=10)

    assert logits.shape == (2, 5)
    assert model.padding_idx not in recommendations
    assert model.unknown_idx not in recommendations
    assert 1 not in recommendations
    assert 2 not in recommendations
    assert len(recommendations) == 2

    batch_recommendations = model.recommend_batch([[1, 2], [3]], k=2, batch_size=2)
    assert len(batch_recommendations) == 2
    assert all(item not in {1, 2} for item in batch_recommendations[0])
    assert 3 not in batch_recommendations[1]
    scores = model.score_items([1], [2, 3, model.unknown_idx, model.padding_idx])
    assert set(scores) == {2, 3}


def test_sampled_cross_entropy_contains_every_batch_target() -> None:
    model = GRU4Rec(num_items=20, embedding_dim=8, hidden_dim=8, dropout=0.0)
    input_ids = torch.tensor([[1, 2], [3, model.padding_idx]], dtype=torch.long)
    lengths = torch.tensor([2, 1], dtype=torch.long)
    targets = torch.tensor([7, 11], dtype=torch.long)

    logits, sampled_targets = sampled_training_logits(
        model,
        input_ids,
        lengths,
        targets,
        negative_count=5,
        generator=torch.Generator().manual_seed(3),
    )

    assert logits.shape[0] == 2
    assert logits.shape[1] <= 7
    assert sampled_targets.shape == targets.shape
    assert torch.isfinite(torch.nn.functional.cross_entropy(logits, sampled_targets))


def _overfit_examples(repeats: int) -> pd.DataFrame:
    rows = []
    for index in range(repeats):
        rows.append(
            {
                "prefix_items": [1],
                "target_item": 2,
                "session_id": index,
                "split": "train",
                "prefix_length": 1,
            }
        )
        rows.append(
            {
                "prefix_items": [1, 2],
                "target_item": 3,
                "session_id": index,
                "split": "train",
                "prefix_length": 2,
            }
        )
    return pd.DataFrame(rows)


def test_gru_can_overfit_save_reload_and_predict(tmp_path: Path) -> None:
    train_examples = _overfit_examples(40)
    validation_examples = _overfit_examples(8)
    checkpoint_path = tmp_path / "tiny_gru.pt"
    log_path = tmp_path / "tiny_gru_log.json"
    config = GRUTrainingConfig(
        embedding_dim=8,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
        batch_size=16,
        learning_rate=0.03,
        epochs=15,
        max_sequence_length=10,
        early_stopping_patience=15,
        device="cpu",
        seed=7,
    )

    model, history = train_gru_model(
        train_examples,
        validation_examples,
        num_items=4,
        config=config,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )
    reloaded = GRU4Rec.load_checkpoint(checkpoint_path)

    assert checkpoint_path.is_file()
    assert log_path.is_file()
    assert min(epoch["train_loss"] for epoch in history) < history[0]["train_loss"] * 0.2
    assert reloaded.checkpoint_metadata["best_validation_metrics"]["mrr"] > 0.9
    assert model.recommend([1], k=1) == [2]
    assert model.recommend([1, 2], k=1) == [3]
    assert reloaded.recommend([1], k=1) == [2]
    assert reloaded.padding_idx not in reloaded.recommend([1], k=10)


def test_artifact_inference_helper_loads_without_training(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    model = GRU4Rec(num_items=5, embedding_dim=4, hidden_dim=4, dropout=0.0)
    model.save_checkpoint(checkpoint_path)

    loaded = load_gru_model(checkpoint_path, device="cpu")

    assert loaded.training is False
    assert next(loaded.parameters()).device.type == "cpu"
    assert 0 not in loaded.recommend([1], k=3)
