"""Segmented next-item ranking evaluation and Markdown reporting."""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from math import ceil
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

PREFIX_SEGMENTS = ("1", "2", "3-5", ">5")
POPULARITY_SEGMENTS = ("head", "mid", "tail")
MODEL_ORDER = ("Popularity", "Markov", "Item-KNN", "GRU4Rec", "GRU4Rec + MMR")


class Recommender(Protocol):
    def recommend(self, prefix_items: Sequence[int], k: int) -> list[int]:
        ...


def prefix_length_segment(prefix_length: int) -> str:
    """Map a prefix length to the required reporting bucket."""
    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")
    if prefix_length == 1:
        return "1"
    if prefix_length == 2:
        return "2"
    if prefix_length <= 5:
        return "3-5"
    return ">5"


def build_popularity_segments(
    train_sessions: Sequence[Sequence[int]],
    head_fraction: float = 0.2,
    mid_fraction: float = 0.3,
) -> dict[int, str]:
    """Rank train items by interaction count and assign head/mid/tail labels."""
    if not 0 < head_fraction < 1:
        raise ValueError("head_fraction must be in the interval (0, 1)")
    if not 0 < mid_fraction < 1 or head_fraction + mid_fraction >= 1:
        raise ValueError("head_fraction + mid_fraction must be less than 1")

    counts = Counter(int(item) for session in train_sessions for item in session)
    ranked_items = sorted(counts, key=lambda item: (-counts[item], item))
    if not ranked_items:
        return {}

    head_end = min(len(ranked_items), max(1, ceil(len(ranked_items) * head_fraction)))
    mid_end = min(
        len(ranked_items),
        max(head_end, ceil(len(ranked_items) * (head_fraction + mid_fraction))),
    )
    return {
        item: "head" if rank < head_end else "mid" if rank < mid_end else "tail"
        for rank, item in enumerate(ranked_items)
    }


def add_segment_columns(
    examples: pd.DataFrame,
    popularity_segments: Mapping[int, str],
) -> tuple[pd.DataFrame, Optional[str]]:
    """Add required segment labels without mutating the input examples."""
    required = {"prefix_items", "target_item", "prefix_length"}
    missing = sorted(required - set(examples.columns))
    if missing:
        raise ValueError(f"Examples are missing required column(s): {', '.join(missing)}")

    segmented = examples.copy()
    segmented["prefix_length_segment"] = segmented["prefix_length"].map(
        lambda length: prefix_length_segment(int(length))
    )
    segmented["target_popularity_segment"] = segmented["target_item"].map(
        lambda item: popularity_segments.get(int(item), "tail")
    )

    event_column = None
    for candidate in ("target_event", "event"):
        if candidate in segmented.columns:
            event_column = candidate
            segmented["event_type_segment"] = segmented[candidate].fillna("unknown").astype(str)
            break
    return segmented, event_column


def _segment_metrics(
    predictions: list[list[int]],
    targets: list[int],
    total_items: int,
    k: int,
) -> dict[str, object]:
    return {
        "examples": len(targets),
        "recall": recall_at_k(predictions, targets, k),
        "hitrate": hitrate_at_k(predictions, targets, k),
        "mrr": mrr_at_k(predictions, targets, k),
        "ndcg": ndcg_at_k(predictions, targets, k),
        "coverage": coverage_at_k(predictions, total_items),
    }


def analyze_segments(
    models: Mapping[str, Recommender],
    examples: pd.DataFrame,
    train_sessions: Sequence[Sequence[int]],
    total_items: int,
    k: int = 20,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    """Generate each model's predictions once and aggregate metrics by segment."""
    if k <= 0:
        raise ValueError("k must be positive")
    popularity_segments = build_popularity_segments(train_sessions)
    segmented, event_column = add_segment_columns(examples, popularity_segments)
    prefixes = segmented["prefix_items"].tolist()
    targets = segmented["target_item"].astype(int).tolist()

    predictions = {}
    for model_name, model in models.items():
        if progress is not None:
            progress(model_name)
        recommend_batch = getattr(model, "recommend_batch", None)
        if callable(recommend_batch):
            predictions[model_name] = recommend_batch(prefixes, k)
        else:
            predictions[model_name] = [
                [int(item) for item in model.recommend(prefix, k)] for prefix in prefixes
            ]

    dimensions = {
        "Prefix length": ("prefix_length_segment", PREFIX_SEGMENTS),
        "Target popularity": ("target_popularity_segment", POPULARITY_SEGMENTS),
    }
    if event_column is not None:
        event_values = tuple(sorted(segmented["event_type_segment"].unique()))
        dimensions["Event type"] = ("event_type_segment", event_values)

    results = {}
    for dimension_name, (column, segment_names) in dimensions.items():
        segment_results = {}
        for segment_name in segment_names:
            positions = [
                position
                for position, value in enumerate(segmented[column].tolist())
                if value == segment_name
            ]
            segment_targets = [targets[position] for position in positions]
            model_results = {}
            for model_name in models:
                segment_predictions = [predictions[model_name][position] for position in positions]
                model_results[model_name] = _segment_metrics(
                    segment_predictions,
                    segment_targets,
                    total_items,
                    k,
                )
            segment_results[str(segment_name)] = model_results
        results[dimension_name] = segment_results

    return {
        "k": k,
        "number_of_examples": len(segmented),
        "event_column": event_column,
        "popularity_definition": {
            "head": "top 20% of training items by interaction count",
            "mid": "next 30% of training items by interaction count",
            "tail": "remaining 50%, including items unseen in training",
        },
        "dimensions": results,
    }


def _ordered_models(model_results: Mapping[str, object]) -> list[str]:
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    return sorted(model_results, key=lambda name: (order.get(name, len(order)), name))


def _best_model(segment_results: Mapping[str, Mapping[str, object]]) -> tuple[str, float]:
    model_name = max(segment_results, key=lambda name: segment_results[name]["recall"])
    return model_name, float(segment_results[model_name]["recall"])


def _build_insights(analysis: Mapping[str, object]) -> list[str]:
    dimensions = analysis["dimensions"]
    insights = []
    prefix_results = dimensions["Prefix length"]
    if prefix_results["1"]:
        model_name, recall = _best_model(prefix_results["1"])
        insights.append(
            f"For prefix length 1, {model_name} has the highest Recall@{analysis['k']} "
            f"({recall:.4f})."
        )

    improved_segments = []
    sequential_models = {"Markov", "Item-KNN", "GRU4Rec"}
    for segment_name in ("2", "3-5", ">5"):
        model_results = prefix_results[segment_name]
        if "Popularity" not in model_results:
            continue
        best_sequential = max(
            (
                metrics["recall"]
                for name, metrics in model_results.items()
                if name in sequential_models
            ),
            default=0.0,
        )
        if best_sequential > model_results["Popularity"]["recall"]:
            improved_segments.append(segment_name)
    if improved_segments:
        insights.append(
            "At least one sequential model outperforms Popularity for prefix segment(s): "
            + ", ".join(improved_segments)
            + "."
        )

    popularity_results = dimensions["Target popularity"]
    head_model, head_recall = _best_model(popularity_results["head"])
    tail_model, tail_recall = _best_model(popularity_results["tail"])
    if tail_recall < head_recall:
        insights.append(
            f"Tail-item prediction remains harder: best tail Recall@{analysis['k']} is "
            f"{tail_recall:.4f} ({tail_model}) versus {head_recall:.4f} for head items "
            f"({head_model})."
        )
    else:
        insights.append(
            f"Best tail Recall@{analysis['k']} is {tail_recall:.4f} ({tail_model}); "
            f"best head recall is {head_recall:.4f} ({head_model})."
        )
    return insights


def render_segment_report(analysis: Mapping[str, object], split_name: str) -> str:
    """Render segmented comparisons and measured observations as Markdown."""
    k = analysis["k"]
    lines = [
        "# Segment Analysis",
        "",
        f"Split: **{split_name}**  ",
        f"Examples: **{analysis['number_of_examples']:,}**  ",
        f"Ranking cutoff: **K={k}**",
        "",
        "Target popularity is fitted on training interactions only: head is the top 20% of",
        "training items, mid is the next 30%, and tail is the remaining 50%. Targets unseen",
        "during training are assigned to tail.",
    ]

    dimensions = analysis["dimensions"]
    for dimension_name, segment_results in dimensions.items():
        lines.extend(
            [
                "",
                f"## {dimension_name}",
                "",
                f"| Segment | Model | Examples | Recall@{k} | HitRate@{k} | MRR@{k} | "
                f"NDCG@{k} | Coverage@{k} |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for segment_name, model_results in segment_results.items():
            for model_name in _ordered_models(model_results):
                metrics = model_results[model_name]
                lines.append(
                    f"| {segment_name} | {model_name} | {metrics['examples']} | "
                    f"{metrics['recall']:.6f} | {metrics['hitrate']:.6f} | "
                    f"{metrics['mrr']:.6f} | {metrics['ndcg']:.6f} | "
                    f"{metrics['coverage']:.6f} |"
                )

    if analysis["event_column"] is None:
        lines.extend(
            [
                "",
                "## Event type",
                "",
                "Event-type analysis is unavailable because the processed prefix-target",
                "artifacts do not retain `event` or `target_event`.",
            ]
        )

    lines.extend(["", "## Observations", ""])
    lines.extend(f"- {insight}" for insight in _build_insights(analysis))
    return "\n".join(lines) + "\n"


def save_segment_report(
    analysis: Mapping[str, object],
    split_name: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_segment_report(analysis, split_name), encoding="utf-8")
    return output_path
