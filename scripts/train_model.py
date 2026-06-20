"""Train and verify the GRU4Rec-style next-item model."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model_gru.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args(argv)


def _load_training_config(config_path: Path) -> object:
    from models.train_gru import GRUTrainingConfig

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config '{config_path}' must contain a YAML mapping")
    model = raw_config.get("model", {})
    training = raw_config.get("training", {})
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError("model and training config sections must be mappings")
    return GRUTrainingConfig(
        embedding_dim=int(model.get("embedding_dim", 100)),
        hidden_dim=int(model.get("hidden_dim", model.get("hidden_size", 100))),
        num_layers=int(model.get("num_layers", 1)),
        dropout=float(model.get("dropout", 0.1)),
        batch_size=int(training.get("batch_size", 256)),
        learning_rate=float(training.get("learning_rate", 0.001)),
        epochs=int(training.get("epochs", training.get("max_epochs", 10))),
        max_sequence_length=int(training.get("max_sequence_length", 50)),
        gradient_clip_norm=float(training.get("gradient_clip_norm", 5.0)),
        early_stopping_patience=int(training.get("early_stopping_patience", 3)),
        device=str(training.get("device", "auto")),
        seed=int(training.get("seed", 42)),
        sampled_softmax_negatives=int(training.get("sampled_softmax_negatives", 0)),
        validation_max_examples=(
            int(training["validation_max_examples"])
            if training.get("validation_max_examples") is not None
            else None
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    from evaluation.evaluate import filter_seen_target_examples, load_processed_examples
    from models.gru4rec import GRU4Rec
    from models.train_gru import train_gru_model

    args = parse_args(argv)
    config = _load_training_config(args.config)
    train_examples = load_processed_examples(args.data_dir, "train")
    validation_examples = load_processed_examples(args.data_dir, "validation")
    train_examples = filter_seen_target_examples(train_examples)
    validation_examples = filter_seen_target_examples(validation_examples)
    mapping_path = args.data_dir / "item_id_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    num_items = len(mapping)

    if args.sample:
        train_examples = train_examples.head(2048).copy()
        validation_examples = validation_examples.head(1024).copy()
        config.epochs = min(config.epochs, 3)
        config.early_stopping_patience = config.epochs

    artifact_path = args.artifact_path
    if artifact_path is None:
        artifact_name = "gru4rec_sample.pt" if args.sample else "gru4rec_best.pt"
        artifact_path = Path("artifacts") / artifact_name
    log_name = "gru_training_log_sample.json" if args.sample else "gru_training_log.json"
    log_path = Path("reports") / log_name

    print(
        f"Training on {len(train_examples):,} examples; validating on "
        f"{len(validation_examples):,}; device={config.device}."
    )
    model, history = train_gru_model(
        train_examples,
        validation_examples,
        num_items=num_items,
        config=config,
        checkpoint_path=artifact_path,
        log_path=log_path,
    )

    reloaded = GRU4Rec.load_checkpoint(artifact_path)
    sample_prefix = validation_examples.iloc[0]["prefix_items"]
    recommendations = reloaded.recommend([int(item) for item in sample_prefix], k=20)
    if reloaded.padding_idx in recommendations:
        raise RuntimeError("Reloaded model recommended the padding index")
    best_epoch = reloaded.checkpoint_metadata.get("best_epoch")
    best_metrics = reloaded.checkpoint_metadata.get("best_validation_metrics")
    print(f"Best epoch: {best_epoch}; validation metrics: {best_metrics}")
    print(f"Saved and reloaded model artifact: {artifact_path}")
    print(f"Saved {len(history)} epoch logs: {log_path}")
    print(f"Reloaded model recommendations: {recommendations[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
