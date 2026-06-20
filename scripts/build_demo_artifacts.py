"""Fit and persist train-only classical models used by the Streamlit demo."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baseline_models.pkl"),
    )
    parser.add_argument("--similarity", choices=("raw", "cosine"), default="cosine")
    parser.add_argument("--recency-decay", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from baselines.item_knn import ItemKNNRecommender
    from baselines.markov import MarkovRecommender
    from baselines.popularity import PopularityRecommender
    from evaluation.evaluate import (
        load_catalog_size,
        load_processed_examples,
        reconstruct_sessions,
    )
    from utils.model_artifacts import save_baseline_bundle

    args = parse_args(argv)
    train_examples = load_processed_examples(args.data_dir, "train")
    train_sessions = reconstruct_sessions(train_examples)
    catalog_size = load_catalog_size(args.data_dir)

    print(f"Fitting baselines on {len(train_sessions):,} training sessions...")
    popularity = PopularityRecommender().fit(train_sessions)
    markov = MarkovRecommender().fit(train_sessions)
    item_knn = ItemKNNRecommender(
        similarity=args.similarity,
        recency_decay=args.recency_decay,
    ).fit(train_sessions)
    output_path = save_baseline_bundle(
        args.output,
        popularity,
        markov,
        item_knn,
        catalog_size,
    )
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
