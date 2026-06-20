"""Training and validation workflow for the GRU4Rec-style model."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import hitrate_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from models.gru4rec import GRU4Rec


@dataclass
class GRUTrainingConfig:
    embedding_dim: int = 100
    hidden_dim: int = 100
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 10
    max_sequence_length: int = 50
    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 3
    device: str = "auto"
    seed: int = 42
    sampled_softmax_negatives: int = 0
    validation_max_examples: Optional[int] = None


class PrefixTargetDataset(Dataset[tuple[list[int], int]]):
    def __init__(self, examples: pd.DataFrame) -> None:
        self.prefixes = [
            [int(item) for item in prefix] for prefix in examples["prefix_items"].tolist()
        ]
        self.targets = examples["target_item"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.prefixes[index], self.targets[index]


def make_collate_fn(
    padding_idx: int,
    max_sequence_length: int,
) -> Callable[[list[tuple[list[int], int]]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    def collate(
        batch: list[tuple[list[int], int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefixes = [prefix[-max_sequence_length:] or [0] for prefix, _ in batch]
        lengths = torch.tensor([len(prefix) for prefix in prefixes], dtype=torch.long)
        max_length = int(lengths.max())
        input_ids = torch.full(
            (len(prefixes), max_length),
            fill_value=padding_idx,
            dtype=torch.long,
        )
        for row_index, prefix in enumerate(prefixes):
            input_ids[row_index, : len(prefix)] = torch.tensor(prefix, dtype=torch.long)
        targets = torch.tensor([target for _, target in batch], dtype=torch.long)
        return input_ids, lengths, targets

    return collate


def resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_gru(
    model: GRU4Rec,
    examples: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    k: int = 20,
) -> dict[str, float]:
    dataset = PrefixTargetDataset(examples)
    if not dataset:
        return {"recall": 0.0, "hitrate": 0.0, "mrr": 0.0, "ndcg": 0.0}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(model.padding_idx, model.max_sequence_length),
    )
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for input_ids, lengths, batch_targets in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            top_items = model.predict_topk(
                input_ids,
                lengths,
                k,
                exclude_input_items=True,
            )
            predictions.extend(top_items.cpu().tolist())
            targets.extend(batch_targets.tolist())
    return {
        "recall": recall_at_k(predictions, targets, k),
        "hitrate": hitrate_at_k(predictions, targets, k),
        "mrr": mrr_at_k(predictions, targets, k),
        "ndcg": ndcg_at_k(predictions, targets, k),
    }


def sampled_training_logits(
    model: GRU4Rec,
    input_ids: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    negative_count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score batch targets plus shared sampled train-vocabulary negatives."""
    if negative_count <= 0 or negative_count >= model.num_items - 1:
        return model(input_ids, lengths), targets
    sampled = torch.randint(
        low=1,
        high=model.num_items,
        size=(negative_count,),
        generator=generator,
        device="cpu",
    )
    candidate_ids = torch.unique(
        torch.cat([targets.detach().cpu(), sampled]),
        sorted=True,
    ).to(targets.device)
    hidden = model.encode(input_ids, lengths)
    logits = functional.linear(
        hidden,
        model.output.weight[candidate_ids],
        model.output.bias[candidate_ids],
    )
    candidate_targets = torch.searchsorted(candidate_ids, targets)
    return logits, candidate_targets


def train_gru_model(
    train_examples: pd.DataFrame,
    validation_examples: pd.DataFrame,
    num_items: int,
    config: GRUTrainingConfig,
    checkpoint_path: Path,
    log_path: Path,
) -> tuple[GRU4Rec, list[dict[str, object]]]:
    """Train, checkpoint by validation MRR@20, reload, and return the best model."""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Resolved training device: {device}")
    model = GRU4Rec(
        num_items=num_items,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_sequence_length=config.max_sequence_length,
    ).to(device)

    train_dataset = PrefixTargetDataset(train_examples)
    if not train_dataset:
        raise ValueError("Training examples must not be empty")
    generator = torch.Generator().manual_seed(config.seed)
    negative_generator = torch.Generator().manual_seed(config.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=make_collate_fn(model.padding_idx, config.max_sequence_length),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_mrr = -1.0
    epochs_without_improvement = 0
    checkpoint_validation = validation_examples
    if (
        config.validation_max_examples is not None
        and len(validation_examples) > config.validation_max_examples
    ):
        checkpoint_validation = validation_examples.sample(
            n=config.validation_max_examples,
            random_state=config.seed,
        ).sort_index()
    print(
        f"Checkpoint validation examples: {len(checkpoint_validation):,}/"
        f"{len(validation_examples):,}"
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for input_ids, lengths, targets in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss_targets = sampled_training_logits(
                model,
                input_ids,
                lengths,
                targets,
                config.sampled_softmax_negatives,
                negative_generator,
            )
            loss = criterion(logits, loss_targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_size = len(targets)
            loss_sum += float(loss.detach()) * batch_size
            example_count += batch_size

        train_loss = loss_sum / example_count
        validation = evaluate_gru(
            model,
            checkpoint_validation,
            batch_size=config.batch_size,
            device=device,
            k=20,
        )
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_recall_at_20": validation["recall"],
            "validation_mrr_at_20": validation["mrr"],
            "validation_ndcg_at_20": validation["ndcg"],
        }
        history.append(epoch_log)
        print(
            f"Epoch {epoch:02d}: loss={train_loss:.6f}, "
            f"Recall@20={validation['recall']:.6f}, "
            f"MRR@20={validation['mrr']:.6f}, "
            f"NDCG@20={validation['ndcg']:.6f}"
        )

        if validation["mrr"] > best_mrr:
            best_mrr = validation["mrr"]
            epochs_without_improvement = 0
            model.save_checkpoint(
                checkpoint_path,
                metadata={
                    "best_epoch": epoch,
                    "best_validation_metrics": validation,
                    "training_config": asdict(config),
                    "checkpoint_validation_examples": len(checkpoint_validation),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    best_model = GRU4Rec.load_checkpoint(checkpoint_path, map_location=str(device)).to(device)
    return best_model, history
