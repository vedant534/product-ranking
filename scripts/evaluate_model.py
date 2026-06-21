"""Evaluate a fitted recommender on processed chronological examples."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_NAMES = ("popularity", "markov", "item_knn", "gru", "gru4rec", "candidate_gru")
DISPLAY_NAMES = {
    "popularity": "Popularity",
    "markov": "Markov",
    "item_knn": "Item-KNN",
    "gru": "GRU4Rec",
    "gru4rec": "GRU4Rec",
    "candidate_gru": "Candidate-aware GRU",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/gru4rec_best.pt"),
    )
    parser.add_argument("--similarity", choices=("raw", "cosine"), default="cosine")
    parser.add_argument("--recency-decay", type=float, default=0.8)
    parser.add_argument("--rerank", choices=("none", "mmr"), default="none")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--candidate-pool-size", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--candidate-artifact-path",
        type=Path,
        default=Path("artifacts/candidate_gru_best.pt"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("configs/model_candidate_gru.yaml"),
    )
    return parser.parse_args(argv)


def _resolve_data_dir(config_path: Path, data_dir: Optional[Path]) -> Path:
    if data_dir is not None:
        return data_dir
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    return Path(str(paths.get("processed_data", "data/processed")))


def _build_model(model_name: str, similarity: str, recency_decay: float) -> object:
    from baselines.item_knn import ItemKNNRecommender
    from baselines.markov import MarkovRecommender
    from baselines.popularity import PopularityRecommender

    if model_name == "popularity":
        return PopularityRecommender()
    if model_name == "markov":
        return MarkovRecommender()
    return ItemKNNRecommender(similarity=similarity, recency_decay=recency_decay)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from evaluation.evaluate import (
        evaluate_recommender,
        filter_seen_target_examples,
        load_catalog_size,
        load_processed_examples,
        reconstruct_sessions,
        save_evaluation_result,
    )

    args = parse_args(argv)
    data_dir = _resolve_data_dir(args.config, args.data_dir)
    evaluation_examples = load_processed_examples(data_dir, args.split)
    original_evaluation_count = len(evaluation_examples)
    evaluation_examples = filter_seen_target_examples(evaluation_examples)
    excluded_seen_targets = original_evaluation_count - len(evaluation_examples)
    if excluded_seen_targets:
        print(
            f"Excluded {excluded_seen_targets:,} examples whose known target already "
            "appeared in the prefix."
        )
    catalog_size = load_catalog_size(data_dir)

    if args.model == "candidate_gru":
        from evaluation.candidate_gru_evaluate import (
            evaluate_candidate_gru_suite,
            write_candidate_gru_diagnostics,
        )
        from models.train_gru import resolve_device
        from scripts.candidate_recall import fit_candidate_generator
        from scripts.train_candidate_gru import load_candidate_training_config

        candidate_config = load_candidate_training_config(args.candidate_config)
        raw_train_examples = load_processed_examples(data_dir, "train")
        generator = fit_candidate_generator(
            raw_train_examples,
            candidate_pool_size=candidate_config.candidate_pool_size,
            candidate_mode=candidate_config.candidate_mode,
            quota_weights=candidate_config.quota_weights,
        )
        device = resolve_device(args.device)
        json_path, markdown_path, payload = evaluate_candidate_gru_suite(
            args.split,
            evaluation_examples,
            catalog_size,
            generator,
            current_checkpoint=args.artifact_path,
            candidate_checkpoint=args.candidate_artifact_path,
            config_path=args.candidate_config,
            reports_dir=args.reports_dir,
            device=device,
            mmr_alpha=candidate_config.mmr_alpha,
        )
        diagnostics_path = write_candidate_gru_diagnostics(args.reports_dir)
        print(f"Candidate-GRU inference device: {device}")
        for model_name, result in payload["models"].items():
            metrics = result["metrics"]["20"]
            print(
                f"{model_name}: Recall@20={metrics['recall']:.6f}, "
                f"MRR@20={metrics['mrr']:.6f}, NDCG@20={metrics['ndcg']:.6f}, "
                f"Coverage@20={metrics['coverage']:.6f}"
            )
        print(f"Saved {json_path}, {markdown_path}, and {diagnostics_path}")
        return 0

    normalized_model_name = "gru4rec" if args.model == "gru" else args.model

    if normalized_model_name == "gru4rec":
        from models.gru4rec import GRU4Rec
        from models.train_gru import resolve_device

        device = resolve_device(args.device)
        model = GRU4Rec.load_checkpoint(args.artifact_path).to(device)
        print(f"GRU inference device: {device}")
    else:
        train_examples = load_processed_examples(data_dir, "train")
        train_sessions = reconstruct_sessions(train_examples)
        model = _build_model(normalized_model_name, args.similarity, args.recency_decay)
        model.fit(train_sessions)

    result_name = DISPLAY_NAMES[args.model]
    diversity_similarities = None
    adapter = None
    if args.rerank == "mmr":
        from baselines.item_knn import ItemKNNRecommender
        from baselines.markov import MarkovRecommender
        from baselines.popularity import PopularityRecommender
        from ranking.candidate_generation import CandidateGenerator
        from ranking.recommend import RecommendationAdapter, RecommendationEngine
        from ranking.rerank import MMRReranker

        if "train_sessions" not in locals():
            train_examples = load_processed_examples(data_dir, "train")
            train_sessions = reconstruct_sessions(train_examples)
        popularity = PopularityRecommender().fit(train_sessions)
        markov = MarkovRecommender().fit(train_sessions)
        item_knn = ItemKNNRecommender(
            similarity=args.similarity,
            recency_decay=args.recency_decay,
        ).fit(train_sessions)
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
            neural_model=model if normalized_model_name == "gru4rec" else None,
        )
        reranker = MMRReranker(item_knn.neighbor_scores, alpha=args.alpha)
        engine_model_name = "gru" if normalized_model_name == "gru4rec" else normalized_model_name
        adapter = RecommendationAdapter(engine, engine_model_name, reranker=reranker)
        diversity_similarities = item_knn.neighbor_scores

        if normalized_model_name == "gru4rec":
            direct_result = evaluate_recommender(
                model,
                evaluation_examples,
                catalog_size,
                similarities=diversity_similarities,
            )
            save_evaluation_result(
                "GRU4Rec",
                args.split,
                direct_result,
                args.reports_dir,
            )
        model = adapter
        result_name = f"{DISPLAY_NAMES[args.model]} + MMR"

    result = evaluate_recommender(
        model,
        evaluation_examples,
        catalog_size,
        similarities=diversity_similarities,
    )
    json_path, markdown_path = save_evaluation_result(
        result_name,
        args.split,
        result,
        args.reports_dir,
    )

    print(
        f"Evaluated {result_name} on {result['number_of_examples']:,} "
        f"{args.split} examples."
    )
    for cutoff, metrics in result["metrics"].items():
        print(
            f"K={cutoff}: Recall={metrics['recall']:.6f}, "
            f"HitRate={metrics['hitrate']:.6f}, MRR={metrics['mrr']:.6f}, "
            f"NDCG={metrics['ndcg']:.6f}, Coverage={metrics['coverage']:.6f}"
        )
        if "diversity" in metrics:
            print(
                f"  Diversity={metrics['diversity']:.6f}, "
                f"ILS={metrics['intra_list_similarity']:.6f}, "
                f"UniqueItems={metrics['unique_recommended_items']}"
            )
    print(f"Average latency/query: {result['average_latency_ms']:.4f} ms")
    if adapter is not None and not evaluation_examples.empty:
        prefix = evaluation_examples.iloc[0]["prefix_items"]
        recommendations = adapter.recommend_with_scores(prefix, 20)
        print(f"\nPrefix: {prefix}")
        print("Top 20 recommendations:")
        print("item_id, score, source")
        for recommendation in recommendations:
            print(
                f"{recommendation.item_id}, {recommendation.score:.6f}, "
                f"{recommendation.source}"
            )
    print(f"Saved {json_path} and {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
