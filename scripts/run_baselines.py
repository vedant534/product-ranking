"""Train and evaluate the required non-neural recommendation baselines."""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_NAMES = ("popularity", "markov", "item_knn")
DISPLAY_NAMES = {
    "popularity": "Popularity",
    "markov": "Markov",
    "item_knn": "Item-KNN",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--k", type=_positive_int, default=20)
    parser.add_argument("--similarity", choices=("raw", "cosine"), default="cosine")
    parser.add_argument("--recency-decay", type=float, default=0.8)
    return parser.parse_args(argv)


def _resolve_data_dir(config_path: Path, explicit_data_dir: Optional[Path]) -> Path:
    if explicit_data_dir is not None:
        return explicit_data_dir
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as error:
        raise FileNotFoundError(f"Could not read config '{config_path}': {error}") from error
    if isinstance(config, dict):
        paths = config.get("paths")
        if isinstance(paths, dict) and "processed_data" in paths:
            return Path(str(paths["processed_data"]))
    return Path("data/processed")


def _load_examples(data_dir: Path, split_name: str) -> pd.DataFrame:
    path = data_dir / f"{split_name}_examples.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset artifact '{path}'. Run the make-dataset preparation step first."
        )
    examples = pd.read_parquet(path)
    observed_splits = set(examples["split"].astype(str).unique())
    if observed_splits and observed_splits != {split_name}:
        raise ValueError(f"{path} contains unexpected split values: {sorted(observed_splits)}")
    examples["prefix_items"] = examples["prefix_items"].map(
        lambda prefix: [int(item) for item in prefix]
    )
    examples["target_item"] = examples["target_item"].astype(int)
    return examples


def reconstruct_train_sessions(train_examples: pd.DataFrame) -> list[list[int]]:
    """Recover each mapped train sequence without counting duplicated prefixes."""
    sessions = []
    for session_id, examples in train_examples.groupby("session_id", sort=False):
        first_prefix = list(examples.iloc[0]["prefix_items"])
        if len(first_prefix) != 1:
            raise ValueError(
                f"Train examples for session {session_id} do not begin with a length-one prefix"
            )
        sequence = first_prefix + examples["target_item"].astype(int).tolist()
        sessions.append(sequence)
    return sessions


def _load_catalog_size(data_dir: Path) -> int:
    mapping_path = data_dir / "item_id_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing item mapping artifact '{mapping_path}'")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping.get("<UNK_ITEM>") != 0:
        raise ValueError("item_id_mapping.json must reserve index 0 for <UNK_ITEM>")
    return len(mapping) - 1


def _build_model(model_name: str, similarity: str, recency_decay: float) -> object:
    from baselines.item_knn import ItemKNNRecommender
    from baselines.markov import MarkovRecommender
    from baselines.popularity import PopularityRecommender

    if model_name == "popularity":
        return PopularityRecommender()
    if model_name == "markov":
        return MarkovRecommender()
    return ItemKNNRecommender(similarity=similarity, recency_decay=recency_decay)


def evaluate_model(
    model: object,
    evaluation_examples: pd.DataFrame,
    total_items: int,
    k: int,
) -> dict[str, float]:
    from evaluation.metrics import (
        coverage_at_k,
        hitrate_at_k,
        mrr_at_k,
        ndcg_at_k,
        recall_at_k,
    )

    prefixes = evaluation_examples["prefix_items"].tolist()
    targets = evaluation_examples["target_item"].astype(int).tolist()
    for prefix in prefixes[: min(100, len(prefixes))]:
        model.recommend(prefix, k)

    predictions = []
    query_latencies_ms = []
    for prefix in prefixes:
        started = time.perf_counter()
        predictions.append(model.recommend(prefix, k))
        query_latencies_ms.append((time.perf_counter() - started) * 1000)

    hitrate_cutoff = min(10, k)
    sorted_latencies = sorted(query_latencies_ms)
    p95_index = max(0, round(0.95 * len(sorted_latencies)) - 1)
    return {
        "recall": recall_at_k(predictions, targets, k),
        "mrr": mrr_at_k(predictions, targets, k),
        "ndcg": ndcg_at_k(predictions, targets, k),
        "hitrate": hitrate_at_k(predictions, targets, hitrate_cutoff),
        "coverage": coverage_at_k(predictions, total_items),
        "latency_mean_ms": (
            sum(query_latencies_ms) / len(query_latencies_ms) if query_latencies_ms else 0.0
        ),
        "latency_p95_ms": sorted_latencies[p95_index] if sorted_latencies else 0.0,
    }


def _print_results(results: list[tuple[str, dict[str, float]]], k: int) -> None:
    hitrate_cutoff = min(10, k)
    print(
        f"\n| Model | Recall@{k} | MRR@{k} | NDCG@{k} | "
        f"HitRate@{hitrate_cutoff} | Coverage@{k} | Latency/query | p95 latency |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model_name, metrics in results:
        print(
            f"| {DISPLAY_NAMES[model_name]} | {metrics['recall']:.6f} | "
            f"{metrics['mrr']:.6f} | {metrics['ndcg']:.6f} | "
            f"{metrics['hitrate']:.6f} | {metrics['coverage']:.6f} | "
            f"{metrics['latency_mean_ms']:.4f} ms | {metrics['latency_p95_ms']:.4f} ms |"
        )

    if len(results) == 1:
        model_name, metrics = results[0]
        display_name = DISPLAY_NAMES[model_name]
        print(f"\n{display_name} Recall@{k}: {metrics['recall']:.6f}")
        print(f"{display_name} MRR@{k}: {metrics['mrr']:.6f}")
        print(f"{display_name} NDCG@{k}: {metrics['ndcg']:.6f}")
        print(f"Coverage@{k}: {metrics['coverage']:.6f}")
        print(f"Latency/query: {metrics['latency_mean_ms']:.4f} ms")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from evaluation.evaluate import filter_seen_target_examples

    args = parse_args(argv)
    data_dir = _resolve_data_dir(args.config, args.data_dir)
    train_examples = _load_examples(data_dir, "train")
    evaluation_examples = _load_examples(data_dir, args.split)
    evaluation_examples = filter_seen_target_examples(evaluation_examples)
    train_sessions = reconstruct_train_sessions(train_examples)
    total_items = _load_catalog_size(data_dir)
    selected_models = MODEL_NAMES if args.model == "all" else (args.model,)

    print(
        f"Fitting on {len(train_sessions):,} train sessions only; "
        f"evaluating {len(evaluation_examples):,} {args.split} examples."
    )
    results = []
    for model_name in selected_models:
        model = _build_model(model_name, args.similarity, args.recency_decay)
        model.fit(train_sessions)
        metrics = evaluate_model(model, evaluation_examples, total_items, args.k)
        results.append((model_name, metrics))
    _print_results(results, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
