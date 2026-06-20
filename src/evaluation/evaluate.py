"""Unified loading, ranking evaluation, and result reporting."""

import json
import time
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd

from evaluation.metrics import (
    coverage_at_k,
    hitrate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from ranking.rerank import (
    SimilaritySource,
    category_count_at_k,
    intra_list_similarity_at_k,
    unique_recommended_items_at_k,
)

DEFAULT_CUTOFFS = (5, 10, 20)
MODEL_DISPLAY_ORDER = {
    "Popularity": 0,
    "Markov": 1,
    "Item-KNN": 2,
    "GRU4Rec": 3,
    "GRU4Rec + MMR": 4,
}


class Recommender(Protocol):
    def recommend(self, prefix_items: Sequence[int], k: int) -> list[int]:
        ...


def load_processed_examples(data_dir: Path, split_name: str) -> pd.DataFrame:
    """Load and validate one processed prefix-target Parquet artifact."""
    path = data_dir / f"{split_name}_examples.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed examples: '{path}'")
    examples = pd.read_parquet(path)
    required = {"prefix_items", "target_item", "session_id", "split", "prefix_length"}
    missing = sorted(required - set(examples.columns))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
    observed_splits = set(examples["split"].astype(str).unique())
    if observed_splits and observed_splits != {split_name}:
        raise ValueError(f"{path} contains unexpected split values: {sorted(observed_splits)}")
    examples["prefix_items"] = examples["prefix_items"].map(
        lambda prefix: [int(item) for item in prefix]
    )
    examples["target_item"] = examples["target_item"].astype(int)
    return examples


def load_catalog_size(data_dir: Path) -> int:
    mapping_path = data_dir / "item_id_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing item mapping: '{mapping_path}'")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping.get("<UNK_ITEM>") != 0:
        raise ValueError("item_id_mapping.json must map <UNK_ITEM> to index 0")
    return len(mapping) - 1


def filter_seen_target_examples(examples: pd.DataFrame) -> pd.DataFrame:
    """Remove known targets that serving policy must exclude as already seen.

    UNK index 0 is retained because multiple distinct raw unseen items collapse to
    that index; equality at index 0 does not prove the raw target was seen.
    """
    keep = [
        int(target) == 0 or int(target) not in set(prefix)
        for prefix, target in zip(examples["prefix_items"], examples["target_item"])
    ]
    return examples.loc[keep].reset_index(drop=True)


def reconstruct_sessions(train_examples: pd.DataFrame) -> list[list[int]]:
    """Recover each train sequence without counting repeated prefix rows."""
    sessions = []
    for session_id, examples in train_examples.groupby("session_id", sort=False):
        first_prefix = list(examples.iloc[0]["prefix_items"])
        if len(first_prefix) != 1:
            raise ValueError(
                f"Train examples for session {session_id} do not begin with a length-one prefix"
            )
        sessions.append(first_prefix + examples["target_item"].astype(int).tolist())
    return sessions


def _validate_cutoffs(cutoffs: Sequence[int]) -> tuple[int, ...]:
    if not cutoffs:
        raise ValueError("At least one evaluation cutoff is required")
    if any(
        isinstance(cutoff, bool) or not isinstance(cutoff, Integral) or cutoff <= 0
        for cutoff in cutoffs
    ):
        raise ValueError("Evaluation cutoffs must be positive integers")
    return tuple(sorted(set(int(cutoff) for cutoff in cutoffs)))


def evaluate_recommender(
    model: Recommender,
    examples: pd.DataFrame,
    total_items: int,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
    warmup_queries: int = 100,
    similarities: Optional[SimilaritySource] = None,
    category_mapping: Optional[dict[object, object]] = None,
) -> dict[str, object]:
    """Evaluate one fitted recommender at all requested cutoffs."""
    evaluated_cutoffs = _validate_cutoffs(cutoffs)
    max_cutoff = max(evaluated_cutoffs)
    prefixes = examples["prefix_items"].tolist()
    targets = examples["target_item"].astype(int).tolist()

    for prefix in prefixes[: min(warmup_queries, len(prefixes))]:
        model.recommend(prefix, max_cutoff)

    predictions = []
    latencies_ms = []
    recommend_batch = getattr(model, "recommend_batch", None)
    if callable(recommend_batch):
        started = time.perf_counter()
        predictions = recommend_batch(prefixes, max_cutoff)
        total_latency_ms = (time.perf_counter() - started) * 1000
        latencies_ms = [total_latency_ms / len(prefixes)] * len(prefixes) if prefixes else []
        latency_mode = "batched_throughput"
    else:
        for prefix in prefixes:
            started = time.perf_counter()
            predictions.append(model.recommend(prefix, max_cutoff))
            latencies_ms.append((time.perf_counter() - started) * 1000)
        latency_mode = "single_query"

    metrics = {}
    for cutoff in evaluated_cutoffs:
        top_k_predictions = [prediction[:cutoff] for prediction in predictions]
        cutoff_metrics = {
            "recall": recall_at_k(top_k_predictions, targets, cutoff),
            "hitrate": hitrate_at_k(top_k_predictions, targets, cutoff),
            "mrr": mrr_at_k(top_k_predictions, targets, cutoff),
            "ndcg": ndcg_at_k(top_k_predictions, targets, cutoff),
            "coverage": coverage_at_k(top_k_predictions, total_items),
            "unique_recommended_items": unique_recommended_items_at_k(
                top_k_predictions,
                cutoff,
            ),
        }
        if similarities is not None:
            average_similarity = sum(
                intra_list_similarity_at_k(prediction, similarities, cutoff)
                for prediction in top_k_predictions
            ) / len(top_k_predictions) if top_k_predictions else 0.0
            cutoff_metrics["intra_list_similarity"] = average_similarity
            cutoff_metrics["diversity"] = 1.0 - average_similarity
        if category_mapping is not None:
            cutoff_metrics["average_category_count"] = (
                sum(
                    category_count_at_k(prediction, category_mapping, cutoff)
                    for prediction in top_k_predictions
                )
                / len(top_k_predictions)
                if top_k_predictions
                else 0.0
            )
        metrics[str(cutoff)] = cutoff_metrics

    return {
        "number_of_examples": len(examples),
        "catalog_size": total_items,
        "average_latency_ms": (
            sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        ),
        "latency_mode": latency_mode,
        "metrics": metrics,
    }


def _render_markdown(results: dict[str, object]) -> str:
    lines = [
        "# Evaluation Results",
        "",
        "Results use the processed chronological artifacts. Model fitting and artifact loading are",
        "excluded from latency. GRU latency is amortized batched throughput; classical latency is",
        "measured per query.",
    ]
    splits = results.get("splits", {})
    for split_name, model_results in splits.items():
        lines.extend(
            [
                "",
                f"## {str(split_name).title()}",
                "",
                "| Model | K | Recall | HitRate | MRR | NDCG | Coverage | "
                "Diversity | ILS | Unique items | Avg latency (ms) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        ordered_models = sorted(
            model_results,
            key=lambda name: (MODEL_DISPLAY_ORDER.get(name, 99), name),
        )
        for model_name in ordered_models:
            result = model_results[model_name]
            ordered_metrics = sorted(
                result["metrics"].items(), key=lambda item: int(item[0])
            )
            for cutoff, metrics in ordered_metrics:
                diversity = metrics.get("diversity")
                intra_list_similarity = metrics.get("intra_list_similarity")
                diversity_text = "N/A" if diversity is None else f"{diversity:.6f}"
                similarity_text = (
                    "N/A"
                    if intra_list_similarity is None
                    else f"{intra_list_similarity:.6f}"
                )
                unique_items = metrics.get("unique_recommended_items")
                if unique_items is None:
                    unique_items = round(metrics["coverage"] * result["catalog_size"])
                lines.append(
                    f"| {model_name} | {cutoff} | {metrics['recall']:.6f} | "
                    f"{metrics['hitrate']:.6f} | {metrics['mrr']:.6f} | "
                    f"{metrics['ndcg']:.6f} | {metrics['coverage']:.6f} | "
                    f"{diversity_text} | {similarity_text} | {unique_items} | "
                    f"{result['average_latency_ms']:.4f} |"
                )
    return "\n".join(lines) + "\n"


def save_evaluation_result(
    model_name: str,
    split_name: str,
    evaluation: dict[str, object],
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Merge one model into an isolated split report and regenerate Markdown."""
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be 'validation' or 'test'")
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"results_{split_name}.json"
    markdown_path = reports_dir / f"results_{split_name}.md"
    if json_path.is_file():
        results = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        results = {"splits": {}}
    splits = results.setdefault("splits", {})
    unexpected_splits = sorted(set(splits) - {split_name})
    if unexpected_splits:
        raise ValueError(
            f"Split report '{json_path}' contains unexpected split(s): "
            f"{', '.join(unexpected_splits)}"
        )
    split_results = splits.setdefault(split_name, {})
    split_results[model_name] = evaluation

    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(results), encoding="utf-8")
    return json_path, markdown_path
