"""Measure train-fitted candidate-pool recall on a chronological split."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

CUTOFFS = (100, 500, 1000)
SOURCES = ("popularity", "markov", "item_knn", "combined")
DISPLAY_NAMES = {
    "popularity": "Popularity",
    "markov": "Markov",
    "item_knn": "Item-KNN",
    "combined": "Combined",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--similarity", choices=("raw", "cosine"), default="cosine")
    parser.add_argument("--recency-decay", type=float, default=0.8)
    return parser.parse_args(argv)


def _resolve_data_dir(config_path: Path, data_dir: Optional[Path]) -> Path:
    if data_dir is not None:
        return data_dir
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    return Path(str(paths.get("processed_data", "data/processed")))


def fit_candidate_generator(
    train_examples: pd.DataFrame,
    similarity: str = "cosine",
    recency_decay: float = 0.8,
):
    """Fit all candidate sources exclusively from the supplied train examples."""
    from baselines.item_knn import ItemKNNRecommender
    from baselines.markov import MarkovRecommender
    from baselines.popularity import PopularityRecommender
    from evaluation.evaluate import reconstruct_sessions
    from ranking.candidate_generation import CandidateGenerator

    train_sessions = reconstruct_sessions(train_examples)
    popularity = PopularityRecommender().fit(train_sessions)
    markov = MarkovRecommender().fit(train_sessions)
    item_knn = ItemKNNRecommender(
        similarity=similarity,
        recency_decay=recency_decay,
    ).fit(train_sessions)
    return CandidateGenerator(
        popularity,
        markov,
        item_knn,
        candidate_pool_size=max(CUTOFFS),
    )


def evaluate_candidate_recall(
    generator,
    examples: pd.DataFrame,
    cutoffs: Sequence[int] = CUTOFFS,
    progress_every: int = 0,
) -> dict[str, object]:
    """Compute one-target recall for native sources and exact bounded merges."""
    evaluated_cutoffs = tuple(sorted(set(int(cutoff) for cutoff in cutoffs)))
    if not evaluated_cutoffs or any(cutoff <= 0 for cutoff in evaluated_cutoffs):
        raise ValueError("cutoffs must contain positive integers")
    maximum_cutoff = max(evaluated_cutoffs)
    hits = {
        source: {str(cutoff): 0 for cutoff in evaluated_cutoffs}
        for source in SOURCES
    }

    for index, (prefix, target) in enumerate(
        zip(examples["prefix_items"], examples["target_item"]),
        start=1,
    ):
        prefix_items = [int(item) for item in prefix]
        target_item = int(target)
        source_scores = generator.score_sources(prefix_items, maximum_cutoff)
        source_rankings = {
            source: list(source_scores[source])
            for source in ("popularity", "markov", "item_knn")
        }

        for cutoff in evaluated_cutoffs:
            cutoff_key = str(cutoff)
            for source, ranking in source_rankings.items():
                if target_item in ranking[:cutoff]:
                    hits[source][cutoff_key] += 1
            if generator.combined_contains_target(
                source_scores,
                target_item,
                pool_size=cutoff,
            ):
                hits["combined"][cutoff_key] += 1

        if progress_every and index % progress_every == 0:
            print(f"Processed {index:,}/{len(examples):,} examples.")

    denominator = len(examples)
    return {
        source: {
            cutoff: count / denominator if denominator else 0.0
            for cutoff, count in source_hits.items()
        }
        for source, source_hits in hits.items()
    }


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        f"# Candidate Recall: {str(payload['split']).title()}",
        "",
        f"Examples: **{int(payload['number_of_examples']):,}**  ",
        f"Unknown targets counted as misses: **{int(payload['unknown_targets']):,}**  ",
        "Fit split: **train**",
        "",
        "Markov and Item-KNN rows use native candidates without popularity fallback. The",
        "combined row uses the serving score normalization and is truncated to exactly K items.",
        "",
        "| Candidate source | Recall@100 | Recall@500 | Recall@1000 |",
        "|---|---:|---:|---:|",
    ]
    recall = payload["recall"]
    for source in SOURCES:
        metrics = recall[source]
        lines.append(
            f"| {DISPLAY_NAMES[source]} | {metrics['100']:.6f} | "
            f"{metrics['500']:.6f} | {metrics['1000']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def save_candidate_recall_report(
    split_name: str,
    recall: dict[str, object],
    number_of_examples: int,
    unknown_targets: int,
    reports_dir: Path,
) -> tuple[Path, Path]:
    """Save one split's candidate recall with explicit leakage policy metadata."""
    if split_name not in {"validation", "test"}:
        raise ValueError("split_name must be 'validation' or 'test'")
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split_name,
        "fit_split": "train",
        "number_of_examples": int(number_of_examples),
        "unknown_targets": int(unknown_targets),
        "cutoffs": list(CUTOFFS),
        "policy": {
            "combined_pool": "normalized source-score merge truncated to exactly K",
            "source_fallbacks": False,
            "seen_items_excluded": True,
            "unknown_targets_count_as_misses": True,
        },
        "recall": recall,
    }
    json_path = reports_dir / f"candidate_recall_{split_name}.json"
    markdown_path = reports_dir / f"candidate_recall_{split_name}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    from evaluation.evaluate import (
        filter_seen_target_examples,
        load_processed_examples,
    )

    args = parse_args(argv)
    data_dir = _resolve_data_dir(args.config, args.data_dir)
    train_examples = load_processed_examples(data_dir, "train")
    evaluation_examples = filter_seen_target_examples(
        load_processed_examples(data_dir, args.split)
    )
    print(f"Fitting candidate generators on {len(train_examples):,} train examples.")
    generator = fit_candidate_generator(
        train_examples,
        similarity=args.similarity,
        recency_decay=args.recency_decay,
    )
    print(
        f"Evaluating candidate recall on {len(evaluation_examples):,} "
        f"{args.split} examples."
    )
    recall = evaluate_candidate_recall(
        generator,
        evaluation_examples,
        progress_every=10_000,
    )
    json_path, markdown_path = save_candidate_recall_report(
        args.split,
        recall,
        number_of_examples=len(evaluation_examples),
        unknown_targets=int(evaluation_examples["target_item"].eq(0).sum()),
        reports_dir=args.reports_dir,
    )
    for source in SOURCES:
        metrics = recall[source]
        print(
            f"{DISPLAY_NAMES[source]}: Recall@100={metrics['100']:.6f}, "
            f"Recall@500={metrics['500']:.6f}, Recall@1000={metrics['1000']:.6f}"
        )
    print(f"Saved {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
