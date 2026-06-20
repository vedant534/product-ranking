"""Validation helpers for saved evaluation and candidate-recall reports."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

MODEL_DISPLAY_ORDER = {
    "Popularity": 0,
    "Markov": 1,
    "Item-KNN": 2,
    "GRU4Rec": 3,
    "GRU4Rec + MMR": 4,
}
DISPLAY_NAMES = {"GRU4Rec + MMR": "GRU4Rec + diversity (MMR)"}
CANDIDATE_SOURCE_ORDER = ("popularity", "markov", "item_knn", "combined")
CANDIDATE_DISPLAY_NAMES = {
    "popularity": "Popularity",
    "markov": "Markov",
    "item_knn": "Item-KNN",
    "combined": "Combined",
}


def _read_payload(path: Path) -> Optional[dict[str, object]]:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Report '{path}' must contain a JSON object")
    return payload


def load_evaluation_summary(
    path: Path,
    expected_split: str,
) -> Optional[pd.DataFrame]:
    """Load validated K=20 metrics for one recorded split."""
    payload = _read_payload(path)
    if payload is None:
        return None
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != {expected_split}:
        raise ValueError(
            f"Evaluation report '{path}' must contain only the '{expected_split}' split"
        )
    split_results = splits[expected_split]
    if not isinstance(split_results, dict) or not split_results:
        raise ValueError(f"Evaluation report '{path}' has no model results")

    rows = []
    for model_name in sorted(
        split_results,
        key=lambda name: (MODEL_DISPLAY_ORDER.get(name, len(MODEL_DISPLAY_ORDER)), name),
    ):
        result = split_results[model_name]
        if not isinstance(result, dict):
            raise ValueError(f"Evaluation report '{path}' has an invalid '{model_name}' result")
        metrics = result.get("metrics", {}).get("20")
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "model": DISPLAY_NAMES.get(model_name, model_name),
                "Recall@20": metrics.get("recall"),
                "MRR@20": metrics.get("mrr"),
                "NDCG@20": metrics.get("ndcg"),
                "Coverage@20": metrics.get("coverage"),
                "Latency/query (ms)": result.get("average_latency_ms"),
            }
        )
    if not rows:
        raise ValueError(f"Evaluation report '{path}' does not contain K=20 metrics")
    return pd.DataFrame(rows)


def summarize_metric_winners(summary: pd.DataFrame) -> dict[str, tuple[str, float]]:
    """Select deterministic K=20 leaders, including Recall-selected neural variant."""
    winners = {}
    for metric in ("Recall@20", "MRR@20", "NDCG@20", "Coverage@20"):
        valid = summary.dropna(subset=[metric])
        if valid.empty:
            continue
        winner = valid.loc[valid[metric].astype(float).idxmax()]
        winners[metric] = (str(winner["model"]), float(winner[metric]))

    neural = summary[summary["model"].astype(str).str.startswith("GRU4Rec")]
    neural = neural.dropna(subset=["Recall@20"])
    if not neural.empty:
        winner = neural.loc[neural["Recall@20"].astype(float).idxmax()]
        winners["Best neural variant"] = (
            str(winner["model"]),
            float(winner["Recall@20"]),
        )
    return winners


def load_candidate_recall_summary(
    path: Path,
    expected_split: str,
) -> Optional[pd.DataFrame]:
    """Load a validated source-by-cutoff candidate-recall table."""
    payload = _read_payload(path)
    if payload is None:
        return None
    if payload.get("split") != expected_split:
        raise ValueError(
            f"Candidate recall report '{path}' is not for split '{expected_split}'"
        )
    cutoffs = payload.get("cutoffs")
    recall = payload.get("recall")
    if cutoffs != [100, 500, 1000] or not isinstance(recall, dict):
        raise ValueError(f"Candidate recall report '{path}' has an invalid schema")

    rows = []
    for source in CANDIDATE_SOURCE_ORDER:
        source_metrics = recall.get(source)
        if not isinstance(source_metrics, dict):
            raise ValueError(f"Candidate recall report '{path}' is missing source '{source}'")
        rows.append(
            {
                "candidate_source": CANDIDATE_DISPLAY_NAMES[source],
                **{
                    f"Recall@{cutoff}": source_metrics.get(str(cutoff))
                    for cutoff in cutoffs
                },
            }
        )
    return pd.DataFrame(rows)
