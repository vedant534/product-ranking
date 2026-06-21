"""Cache-backed hard-negative training for the experimental CandidateGRU."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import hitrate_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from models.candidate_gru import CandidateGRU
from models.train_gru import make_collate_fn, resolve_device


@dataclass
class CandidateGRUTrainingConfig:
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 20
    max_sequence_length: int = 50
    gradient_clip_norm: float = 5.0
    early_stopping_patience: int = 5
    hard_candidate_negatives: int = 256
    candidate_pool_size: int = 1000
    candidate_mode: str = "quota_combined"
    quota_weights: dict[str, int] = field(
        default_factory=lambda: {"item_knn": 700, "markov": 200, "popularity": 100}
    )
    validation_max_examples: int = 20000
    mmr_alpha: float = 0.8
    device: str = "auto"
    seed: int = 42
    cache_dir: str = "data/interim/candidate_gru"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_signature(
    mapping_hash: str,
    examples_hash: str,
    config: CandidateGRUTrainingConfig,
    cache_kind: str,
    number_of_examples: int,
    source_examples_hash: Optional[str] = None,
) -> str:
    payload = {
        "mapping_hash": mapping_hash,
        "examples_hash": examples_hash,
        "source_examples_hash": source_examples_hash or examples_hash,
        "cache_kind": cache_kind,
        "number_of_examples": number_of_examples,
        "candidate_mode": config.candidate_mode,
        "candidate_pool_size": config.candidate_pool_size,
        "quota_weights": config.quota_weights,
        "hard_candidate_negatives": config.hard_candidate_negatives,
        "seed": config.seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def sample_hard_negatives(
    candidate_items: list[int],
    prefix_items: list[int],
    target_item: int,
    negative_count: int,
    seed: int,
) -> list[int]:
    """Sample deterministic candidate negatives while excluding invalid items."""
    excluded = set(prefix_items)
    excluded.update({target_item, 0})
    eligible = list(dict.fromkeys(item for item in candidate_items if item not in excluded))
    if len(eligible) <= negative_count:
        return eligible
    return random.Random(seed).sample(eligible, negative_count)


def _load_matching_cache(
    metadata_path: Path,
    expected_signature: str,
    array_paths: list[Path],
    identity_metadata: Optional[dict[str, object]] = None,
) -> Optional[tuple[dict[str, object], list[np.ndarray]]]:
    if not metadata_path.is_file() or not all(path.is_file() for path in array_paths):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != expected_signature:
        return None
    if identity_metadata is not None and any(
        metadata.get(key) != value for key, value in identity_metadata.items()
    ):
        return None
    arrays = [np.load(path, mmap_mode="r") for path in array_paths]
    return metadata, arrays


def build_hard_negative_cache(
    generator,
    examples: pd.DataFrame,
    config: CandidateGRUTrainingConfig,
    cache_dir: Path,
    signature: str,
    cache_prefix: str = "train",
    progress_every: int = 10000,
    identity_metadata: Optional[dict[str, object]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Create or load fixed-width hard negatives for each training example."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    negatives_path = cache_dir / f"{cache_prefix}_hard_negatives.npy"
    lengths_path = cache_dir / f"{cache_prefix}_hard_negative_lengths.npy"
    metadata_path = cache_dir / f"{cache_prefix}_hard_negatives.json"
    cached = _load_matching_cache(
        metadata_path,
        signature,
        [negatives_path, lengths_path],
        identity_metadata,
    )
    if cached is not None:
        metadata, arrays = cached
        return arrays[0], arrays[1], metadata

    negatives = np.lib.format.open_memmap(
        negatives_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(examples), config.hard_candidate_negatives),
    )
    negatives[:] = 0
    lengths = np.lib.format.open_memmap(
        lengths_path,
        mode="w+",
        dtype=np.uint16,
        shape=(len(examples),),
    )
    natural_target_hits = 0
    for index, (prefix, target) in enumerate(
        zip(examples["prefix_items"], examples["target_item"])
    ):
        prefix_items = [int(item) for item in prefix]
        target_item = int(target)
        pool = generator.generate(prefix_items)
        candidate_items = [int(candidate.item_id) for candidate in pool]
        natural_target_hits += int(target_item in candidate_items)
        sampled = sample_hard_negatives(
            candidate_items,
            prefix_items,
            target_item,
            config.hard_candidate_negatives,
            config.seed + index * 1000003,
        )
        lengths[index] = len(sampled)
        if sampled:
            negatives[index, : len(sampled)] = sampled
        if progress_every and (index + 1) % progress_every == 0:
            print(f"Cached hard negatives for {index + 1:,}/{len(examples):,} examples.")
    negatives.flush()
    lengths.flush()
    metadata = {
        "signature": signature,
        "number_of_examples": len(examples),
        "width": config.hard_candidate_negatives,
        "natural_target_hits": natural_target_hits,
        "natural_target_recall": natural_target_hits / len(examples) if len(examples) else 0.0,
    }
    if identity_metadata is not None:
        metadata.update(identity_metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return np.load(negatives_path, mmap_mode="r"), np.load(lengths_path, mmap_mode="r"), metadata


def build_candidate_pool_cache(
    generator,
    examples: pd.DataFrame,
    config: CandidateGRUTrainingConfig,
    cache_dir: Path,
    signature: str,
    cache_prefix: str = "validation",
    progress_every: int = 5000,
    identity_metadata: Optional[dict[str, object]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Create or load full quota pools used for exact checkpoint metrics."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = cache_dir / f"{cache_prefix}_candidate_pools.npy"
    lengths_path = cache_dir / f"{cache_prefix}_candidate_lengths.npy"
    metadata_path = cache_dir / f"{cache_prefix}_candidate_pools.json"
    cached = _load_matching_cache(
        metadata_path,
        signature,
        [candidates_path, lengths_path],
        identity_metadata,
    )
    if cached is not None:
        metadata, arrays = cached
        return arrays[0], arrays[1], metadata

    candidates = np.lib.format.open_memmap(
        candidates_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(examples), config.candidate_pool_size),
    )
    candidates[:] = 0
    lengths = np.lib.format.open_memmap(
        lengths_path,
        mode="w+",
        dtype=np.uint16,
        shape=(len(examples),),
    )
    natural_target_hits = 0
    for index, (prefix, target) in enumerate(
        zip(examples["prefix_items"], examples["target_item"])
    ):
        prefix_items = [int(item) for item in prefix]
        pool = [int(candidate.item_id) for candidate in generator.generate(prefix_items)]
        length = min(len(pool), config.candidate_pool_size)
        lengths[index] = length
        if length:
            candidates[index, :length] = pool[:length]
        natural_target_hits += int(int(target) in pool[:length])
        if progress_every and (index + 1) % progress_every == 0:
            print(f"Cached candidate pools for {index + 1:,}/{len(examples):,} examples.")
    candidates.flush()
    lengths.flush()
    metadata = {
        "signature": signature,
        "number_of_examples": len(examples),
        "width": config.candidate_pool_size,
        "natural_target_hits": natural_target_hits,
        "natural_target_recall": natural_target_hits / len(examples) if len(examples) else 0.0,
    }
    if identity_metadata is not None:
        metadata.update(identity_metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return np.load(candidates_path, mmap_mode="r"), np.load(lengths_path, mmap_mode="r"), metadata


class CandidateTrainingDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        negatives: np.ndarray,
        negative_lengths: np.ndarray,
    ) -> None:
        self.prefixes = examples["prefix_items"].tolist()
        self.targets = examples["target_item"].astype(int).tolist()
        self.negatives = negatives
        self.negative_lengths = negative_lengths

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        length = int(self.negative_lengths[index])
        return (
            [int(item) for item in self.prefixes[index]],
            self.targets[index],
            self.negatives[index, :length].astype(np.int64).tolist(),
        )


def make_candidate_collate_fn(padding_idx: int, max_sequence_length: int):
    def collate(batch):
        prefixes = [prefix[-max_sequence_length:] or [0] for prefix, _, _ in batch]
        lengths = torch.tensor([len(prefix) for prefix in prefixes], dtype=torch.long)
        input_ids = torch.full(
            (len(batch), int(lengths.max())),
            fill_value=padding_idx,
            dtype=torch.long,
        )
        for row, prefix in enumerate(prefixes):
            input_ids[row, : len(prefix)] = torch.tensor(prefix, dtype=torch.long)
        maximum_negatives = max((len(negative) for _, _, negative in batch), default=0)
        candidate_ids = torch.zeros(
            (len(batch), 1 + maximum_negatives),
            dtype=torch.long,
        )
        candidate_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
        for row, (_, target, negatives) in enumerate(batch):
            candidate_ids[row, 0] = int(target)
            candidate_mask[row, 0] = True
            if negatives:
                candidate_ids[row, 1 : 1 + len(negatives)] = torch.tensor(
                    negatives, dtype=torch.long
                )
                candidate_mask[row, 1 : 1 + len(negatives)] = True
        return input_ids, lengths, candidate_ids, candidate_mask

    return collate


def evaluate_candidate_gru(
    model: CandidateGRU,
    examples: pd.DataFrame,
    candidate_pools: np.ndarray,
    candidate_lengths: np.ndarray,
    batch_size: int,
    device: torch.device,
    k: int = 20,
) -> dict[str, float]:
    predictions = []
    targets = examples["target_item"].astype(int).tolist()
    collate_prefixes = make_collate_fn(model.padding_idx, model.max_sequence_length)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            stop = min(start + batch_size, len(examples))
            batch_examples = [
                (list(examples.iloc[index]["prefix_items"]), int(targets[index]))
                for index in range(start, stop)
            ]
            input_ids, lengths, _ = collate_prefixes(batch_examples)
            pool_ids = torch.tensor(
                np.asarray(candidate_pools[start:stop]), dtype=torch.long, device=device
            )
            pool_lengths = torch.tensor(
                np.asarray(candidate_lengths[start:stop]), dtype=torch.long, device=device
            )
            positions = torch.arange(pool_ids.shape[1], device=device).unsqueeze(0)
            pool_mask = positions < pool_lengths.unsqueeze(1)
            logits = model.candidate_logits(
                input_ids.to(device),
                lengths.to(device),
                pool_ids,
                pool_mask,
            )
            cutoff = min(k, pool_ids.shape[1])
            top_positions = torch.topk(logits, k=cutoff, dim=1).indices
            top_items = pool_ids.gather(1, top_positions).cpu().tolist()
            top_masks = pool_mask.gather(1, top_positions).cpu().tolist()
            predictions.extend(
                [item for item, valid in zip(items, mask) if valid and item != 0]
                for items, mask in zip(top_items, top_masks)
            )
    return {
        "recall": recall_at_k(predictions, targets, k),
        "hitrate": hitrate_at_k(predictions, targets, k),
        "mrr": mrr_at_k(predictions, targets, k),
        "ndcg": ndcg_at_k(predictions, targets, k),
    }


def train_candidate_gru_model(
    train_examples: pd.DataFrame,
    validation_examples: pd.DataFrame,
    train_negatives: np.ndarray,
    train_negative_lengths: np.ndarray,
    validation_candidates: np.ndarray,
    validation_candidate_lengths: np.ndarray,
    num_items: int,
    config: CandidateGRUTrainingConfig,
    checkpoint_path: Path,
    log_path: Path,
    cache_metadata: dict[str, object],
) -> tuple[CandidateGRU, list[dict[str, object]]]:
    """Train with candidate negatives and checkpoint by candidate-pool MRR@20."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    print(f"Resolved candidate-GRU training device: {device}")
    model = CandidateGRU(
        num_items=num_items,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_sequence_length=config.max_sequence_length,
    ).to(device)
    dataset = CandidateTrainingDataset(
        train_examples,
        train_negatives,
        train_negative_lengths,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=make_candidate_collate_fn(model.padding_idx, config.max_sequence_length),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    best_mrr = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for input_ids, lengths, candidate_ids, candidate_mask in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            candidate_ids = candidate_ids.to(device)
            candidate_mask = candidate_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.candidate_loss(
                input_ids,
                lengths,
                candidate_ids,
                candidate_mask,
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_count = len(input_ids)
            loss_sum += float(loss.detach()) * batch_count
            example_count += batch_count
        train_loss = loss_sum / example_count
        validation = evaluate_candidate_gru(
            model,
            validation_examples,
            validation_candidates,
            validation_candidate_lengths,
            config.batch_size,
            device,
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
            f"MRR@20={validation['mrr']:.6f}, NDCG@20={validation['ndcg']:.6f}"
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
                    "checkpoint_validation_examples": len(validation_examples),
                    "cache_metadata": cache_metadata,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}.")
                break
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    best_model = CandidateGRU.load_checkpoint(checkpoint_path, map_location=str(device)).to(device)
    return best_model, history
