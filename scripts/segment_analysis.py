"""Compare recommendation models across behavior and item-popularity segments."""

import argparse
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
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/gru4rec_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/segment_analysis.md"),
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--similarity", choices=("raw", "cosine"), default="cosine")
    parser.add_argument("--recency-decay", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--candidate-pool-size", type=int, default=500)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def _resolve_data_dir(config_path: Path, data_dir: Optional[Path]) -> Path:
    if data_dir is not None:
        return data_dir
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    return Path(str(paths.get("processed_data", "data/processed")))


def main(argv: Optional[Sequence[str]] = None) -> int:
    from baselines.item_knn import ItemKNNRecommender
    from baselines.markov import MarkovRecommender
    from baselines.popularity import PopularityRecommender
    from evaluation.evaluate import (
        filter_seen_target_examples,
        load_catalog_size,
        load_processed_examples,
        reconstruct_sessions,
    )
    from evaluation.segment_analysis import analyze_segments, save_segment_report
    from models.gru4rec import GRU4Rec
    from models.train_gru import resolve_device
    from ranking.candidate_generation import CandidateGenerator
    from ranking.recommend import RecommendationAdapter, RecommendationEngine
    from ranking.rerank import MMRReranker

    args = parse_args(argv)
    data_dir = _resolve_data_dir(args.config, args.data_dir)
    train_examples = load_processed_examples(data_dir, "train")
    evaluation_examples = load_processed_examples(data_dir, args.split)
    evaluation_examples = filter_seen_target_examples(evaluation_examples)
    catalog_size = load_catalog_size(data_dir)
    train_sessions = reconstruct_sessions(train_examples)

    print(f"Fitting classical models on {len(train_sessions):,} training sessions...")
    popularity = PopularityRecommender().fit(train_sessions)
    markov = MarkovRecommender().fit(train_sessions)
    item_knn = ItemKNNRecommender(
        similarity=args.similarity,
        recency_decay=args.recency_decay,
    ).fit(train_sessions)
    device = resolve_device(args.device)
    gru = GRU4Rec.load_checkpoint(args.artifact_path).to(device)
    print(f"GRU inference device: {device}")
    expected_num_items = catalog_size + 1
    if gru.num_items != expected_num_items:
        raise ValueError(
            f"GRU checkpoint has {gru.num_items} item classes, but processed data requires "
            f"{expected_num_items}. Use matching dataset and checkpoint artifacts."
        )

    generator = CandidateGenerator(
        popularity,
        markov,
        item_knn,
        candidate_pool_size=args.candidate_pool_size,
    )
    engine = RecommendationEngine(
        generator,
        popularity,
        markov,
        item_knn,
        neural_model=gru,
    )
    reranked_gru = RecommendationAdapter(
        engine,
        "gru",
        reranker=MMRReranker(item_knn.neighbor_scores, alpha=args.alpha),
    )
    models = {
        "Popularity": popularity,
        "Markov": markov,
        "Item-KNN": item_knn,
        "GRU4Rec": gru,
        "GRU4Rec + MMR": reranked_gru,
    }

    analysis = analyze_segments(
        models,
        evaluation_examples,
        train_sessions,
        total_items=catalog_size,
        k=args.k,
        progress=lambda model_name: print(f"Generating {model_name} recommendations..."),
    )
    output_path = save_segment_report(analysis, args.split, args.output)
    event_status = analysis["event_column"] or "not available in processed examples"
    print(
        f"Analyzed {analysis['number_of_examples']:,} {args.split} examples at K={args.k}."
    )
    print(f"Event segmentation: {event_status}")
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
