"""Train the isolated candidate-aware GRU experiment."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model_candidate_gru.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args(argv)


def load_candidate_training_config(config_path: Path):
    from models.train_candidate_gru import CandidateGRUTrainingConfig

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config '{config_path}' must contain a YAML mapping")
    model = raw.get("model", {})
    training = raw.get("training", {})
    candidate = raw.get("candidate", {})
    if not all(isinstance(section, dict) for section in (model, training, candidate)):
        raise ValueError("model, training, and candidate sections must be mappings")
    return CandidateGRUTrainingConfig(
        embedding_dim=int(model.get("embedding_dim", 128)),
        hidden_dim=int(model.get("hidden_dim", 128)),
        num_layers=int(model.get("num_layers", 1)),
        dropout=float(model.get("dropout", 0.1)),
        batch_size=int(training.get("batch_size", 256)),
        learning_rate=float(training.get("learning_rate", 0.001)),
        epochs=int(training.get("epochs", 20)),
        max_sequence_length=int(training.get("max_sequence_length", 50)),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 5.0)),
        early_stopping_patience=int(training.get("early_stopping_patience", 5)),
        validation_max_examples=int(training.get("validation_max_examples", 20000)),
        device=str(training.get("device", "auto")),
        seed=int(training.get("seed", 42)),
        hard_candidate_negatives=int(candidate.get("hard_negatives", 256)),
        candidate_pool_size=int(candidate.get("pool_size", 1000)),
        candidate_mode=str(candidate.get("mode", "quota_combined")),
        quota_weights={
            str(source): int(weight)
            for source, weight in candidate.get(
                "quotas",
                {"item_knn": 700, "markov": 200, "popularity": 100},
            ).items()
        },
        mmr_alpha=float(candidate.get("mmr_alpha", 0.8)),
        cache_dir=str(candidate.get("cache_dir", "data/interim/candidate_gru")),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    from evaluation.evaluate import filter_seen_target_examples, load_processed_examples
    from models.candidate_gru import CandidateGRU
    from models.train_candidate_gru import (
        build_candidate_pool_cache,
        build_hard_negative_cache,
        cache_signature,
        sha256_file,
        train_candidate_gru_model,
    )
    from scripts.candidate_recall import fit_candidate_generator

    args = parse_args(argv)
    config = load_candidate_training_config(args.config)
    raw_train_examples = load_processed_examples(args.data_dir, "train")
    train_examples = filter_seen_target_examples(raw_train_examples)
    validation_examples = filter_seen_target_examples(
        load_processed_examples(args.data_dir, "validation")
    )
    if config.validation_max_examples and len(validation_examples) > config.validation_max_examples:
        validation_examples = validation_examples.sample(
            n=config.validation_max_examples,
            random_state=config.seed,
        ).sort_index().reset_index(drop=True)
    cache_prefix = "full"
    source_fit_examples = raw_train_examples
    if args.sample:
        train_examples = train_examples.head(2048).reset_index(drop=True)
        validation_examples = validation_examples.head(1024).reset_index(drop=True)
        source_fit_examples = raw_train_examples.head(20000).reset_index(drop=True)
        config.embedding_dim = min(config.embedding_dim, 32)
        config.hidden_dim = min(config.hidden_dim, 32)
        config.epochs = min(config.epochs, 3)
        config.early_stopping_patience = config.epochs
        config.hard_candidate_negatives = min(config.hard_candidate_negatives, 64)
        config.candidate_pool_size = min(config.candidate_pool_size, 200)
        cache_prefix = "sample"

    mapping_path = args.data_dir / "item_id_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    print(
        f"Fitting candidate sources on {len(source_fit_examples):,} train examples "
        f"({'sample smoke policy' if args.sample else 'full train policy'})."
    )
    generator = fit_candidate_generator(
        source_fit_examples,
        candidate_pool_size=config.candidate_pool_size,
        candidate_mode=config.candidate_mode,
        quota_weights=config.quota_weights,
    )
    mapping_hash = sha256_file(mapping_path)
    train_hash = sha256_file(args.data_dir / "train_examples.parquet")
    validation_hash = sha256_file(args.data_dir / "validation_examples.parquet")
    config_hash = sha256_file(args.config)
    cache_dir = Path(config.cache_dir)
    common_cache_identity = {
        "item_mapping_sha256": mapping_hash,
        "source_examples_sha256": train_hash,
        "candidate_config_sha256": config_hash,
        "candidate_mode": config.candidate_mode,
        "candidate_pool_size": config.candidate_pool_size,
        "base_quotas": dict(config.quota_weights),
        "resolved_quotas": generator.resolved_quotas(),
        "seed": config.seed,
    }
    train_cache_identity = {
        **common_cache_identity,
        "examples_sha256": train_hash,
    }
    validation_cache_identity = {
        **common_cache_identity,
        "examples_sha256": validation_hash,
    }
    train_signature = cache_signature(
        mapping_hash,
        train_hash,
        config,
        f"{cache_prefix}_hard_negatives",
        len(train_examples),
        source_examples_hash=train_hash,
    )
    validation_signature = cache_signature(
        mapping_hash,
        validation_hash,
        config,
        f"{cache_prefix}_validation_pools",
        len(validation_examples),
        source_examples_hash=train_hash,
    )
    train_negatives, train_lengths, train_cache = build_hard_negative_cache(
        generator,
        train_examples,
        config,
        cache_dir,
        train_signature,
        cache_prefix=f"{cache_prefix}_train",
        identity_metadata=train_cache_identity,
    )
    validation_pools, validation_lengths, validation_cache = build_candidate_pool_cache(
        generator,
        validation_examples,
        config,
        cache_dir,
        validation_signature,
        cache_prefix=f"{cache_prefix}_validation",
        identity_metadata=validation_cache_identity,
    )
    artifact_path = args.artifact_path or Path("artifacts") / (
        "candidate_gru_sample.pt" if args.sample else "candidate_gru_best.pt"
    )
    log_path = Path("reports") / (
        "candidate_gru_training_log_sample.json"
        if args.sample
        else "candidate_gru_training_log.json"
    )
    model, history = train_candidate_gru_model(
        train_examples,
        validation_examples,
        train_negatives,
        train_lengths,
        validation_pools,
        validation_lengths,
        num_items=len(mapping),
        config=config,
        checkpoint_path=artifact_path,
        log_path=log_path,
        cache_metadata={
            "mapping_sha256": mapping_hash,
            "train_examples_sha256": train_hash,
            "validation_examples_sha256": validation_hash,
            "train_cache": train_cache,
            "validation_cache": validation_cache,
            "config_sha256": config_hash,
        },
    )
    reloaded = CandidateGRU.load_checkpoint(artifact_path)
    print(f"Best epoch: {reloaded.checkpoint_metadata.get('best_epoch')}")
    print(f"Best metrics: {reloaded.checkpoint_metadata.get('best_validation_metrics')}")
    print(f"Saved candidate-GRU checkpoint: {artifact_path}")
    print(f"Saved {len(history)} training epochs: {log_path}")
    del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
