"""Controlled evaluation and diagnostics for the candidate-aware GRU experiment."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import (
    coverage_at_k,
    hitrate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from models.candidate_gru import CandidateGRU
from models.train_candidate_gru import sha256_file
from models.train_gru import make_collate_fn
from ranking.recommend import RankedRecommendation
from ranking.rerank import (
    MMRReranker,
    intra_list_similarity_at_k,
    unique_recommended_items_at_k,
)

CONTROL_NAME = "Current GRU - quota_combined"
MAIN_NAME = "Candidate-aware GRU - quota_combined"
MMR_NAME = "Candidate-aware GRU - quota_combined + MMR"
MODEL_ORDER = (CONTROL_NAME, MAIN_NAME, MMR_NAME)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _result_payload(
    predictions: list[list[int]],
    targets: list[int],
    catalog_size: int,
    elapsed_seconds: float,
    similarities=None,
) -> dict[str, object]:
    metrics = {
        "recall": recall_at_k(predictions, targets, 20),
        "hitrate": hitrate_at_k(predictions, targets, 20),
        "mrr": mrr_at_k(predictions, targets, 20),
        "ndcg": ndcg_at_k(predictions, targets, 20),
        "coverage": coverage_at_k(predictions, catalog_size),
        "unique_recommended_items": unique_recommended_items_at_k(predictions, 20),
    }
    if similarities is not None:
        average_similarity = (
            sum(
                intra_list_similarity_at_k(prediction, similarities, 20)
                for prediction in predictions
            )
            / len(predictions)
            if predictions
            else 0.0
        )
        metrics["intra_list_similarity"] = average_similarity
        metrics["diversity"] = 1.0 - average_similarity
    return {
        "number_of_examples": len(predictions),
        "catalog_size": catalog_size,
        "average_latency_ms": (
            elapsed_seconds * 1000 / len(predictions) if predictions else 0.0
        ),
        "latency_mode": "batched_throughput",
        "metrics": {"20": metrics},
    }


def evaluate_on_shared_candidate_pools(
    current_model: CandidateGRU,
    candidate_model: CandidateGRU,
    generator,
    examples: pd.DataFrame,
    catalog_size: int,
    device: torch.device,
    mmr_alpha: float,
    batch_size: int = 256,
) -> dict[str, dict[str, object]]:
    """Generate each quota pool once and score it with both GRUs."""
    current_model.eval()
    candidate_model.eval()
    current_collate = make_collate_fn(
        current_model.padding_idx,
        current_model.max_sequence_length,
    )
    candidate_collate = make_collate_fn(
        candidate_model.padding_idx,
        candidate_model.max_sequence_length,
    )
    reranker = MMRReranker(generator.item_knn.neighbor_scores, alpha=mmr_alpha)
    targets = examples["target_item"].astype(int).tolist()
    prefixes = examples["prefix_items"].tolist()
    current_predictions: list[list[int]] = []
    candidate_predictions: list[list[int]] = []
    mmr_predictions: list[list[int]] = []
    generation_elapsed = 0.0
    current_elapsed = 0.0
    candidate_elapsed = 0.0
    current_ranking_elapsed = 0.0
    candidate_ranking_elapsed = 0.0
    mmr_elapsed = 0.0

    with torch.no_grad():
        for start in range(0, len(prefixes), batch_size):
            batch_prefixes = [
                [int(item) for item in prefix]
                for prefix in prefixes[start : start + batch_size]
            ]
            started = time.perf_counter()
            pools = [generator.generate(prefix) for prefix in batch_prefixes]
            generation_elapsed += time.perf_counter() - started
            maximum_width = max((len(pool) for pool in pools), default=0)
            if maximum_width == 0:
                current_predictions.extend([] for _ in pools)
                candidate_predictions.extend([] for _ in pools)
                mmr_predictions.extend([] for _ in pools)
                continue
            pool_array = np.zeros((len(pools), maximum_width), dtype=np.int64)
            pool_mask_array = np.zeros_like(pool_array, dtype=bool)
            for row, pool in enumerate(pools):
                item_ids = [int(candidate.item_id) for candidate in pool]
                pool_array[row, : len(item_ids)] = item_ids
                pool_mask_array[row, : len(item_ids)] = True
            pool_ids = torch.as_tensor(pool_array, dtype=torch.long, device=device)
            pool_mask = torch.as_tensor(pool_mask_array, dtype=torch.bool, device=device)

            batch_targets = targets[start : start + len(batch_prefixes)]
            current_inputs, current_lengths, _ = current_collate(
                list(zip(batch_prefixes, batch_targets))
            )
            candidate_inputs, candidate_lengths, _ = candidate_collate(
                list(zip(batch_prefixes, batch_targets))
            )
            _synchronize(device)
            started = time.perf_counter()
            current_logits = current_model.candidate_logits(
                current_inputs.to(device),
                current_lengths.to(device),
                pool_ids,
                pool_mask,
            )
            _synchronize(device)
            current_elapsed += time.perf_counter() - started
            started = time.perf_counter()
            candidate_logits = candidate_model.candidate_logits(
                candidate_inputs.to(device),
                candidate_lengths.to(device),
                pool_ids,
                pool_mask,
            )
            _synchronize(device)
            candidate_elapsed += time.perf_counter() - started
            current_scores = current_logits.cpu().numpy()
            candidate_scores = candidate_logits.cpu().numpy()

            started = time.perf_counter()
            current_orders = [
                np.argsort(-current_scores[row, : len(pool)], kind="stable")
                for row, pool in enumerate(pools)
            ]
            current_predictions.extend(
                [int(pools[row][position].item_id) for position in order[:20]]
                for row, order in enumerate(current_orders)
            )
            current_ranking_elapsed += time.perf_counter() - started

            started = time.perf_counter()
            candidate_orders = [
                np.argsort(-candidate_scores[row, : len(pool)], kind="stable")
                for row, pool in enumerate(pools)
            ]
            candidate_predictions.extend(
                [int(pools[row][position].item_id) for position in order[:20]]
                for row, order in enumerate(candidate_orders)
            )
            candidate_ranking_elapsed += time.perf_counter() - started

            started = time.perf_counter()
            for row, order in enumerate(candidate_orders):
                ranked = [
                    RankedRecommendation(
                        item_id=int(pools[row][position].item_id),
                        score=float(candidate_scores[row, position]),
                        sources=pools[row][position].sources + ("neural",),
                    )
                    for position in order
                ]
                reranked = reranker.rerank(
                    ranked,
                    20,
                    seen_items=set(batch_prefixes[row]),
                )
                mmr_predictions.append([int(item.item_id) for item in reranked])
            mmr_elapsed += time.perf_counter() - started

    shared_generation = generation_elapsed
    current_total = shared_generation + current_elapsed + current_ranking_elapsed
    candidate_total = shared_generation + candidate_elapsed + candidate_ranking_elapsed
    mmr_total = candidate_total + mmr_elapsed
    return {
        CONTROL_NAME: _result_payload(
            current_predictions,
            targets,
            catalog_size,
            current_total,
        ),
        MAIN_NAME: _result_payload(
            candidate_predictions,
            targets,
            catalog_size,
            candidate_total,
        ),
        MMR_NAME: _result_payload(
            mmr_predictions,
            targets,
            catalog_size,
            mmr_total,
            similarities=generator.item_knn.neighbor_scores,
        ),
    }


def _render_candidate_report(payload: dict[str, object]) -> str:
    metadata = payload["metadata"]
    lines = [
        f"# Candidate-GRU Results: {str(metadata['split']).title()}",
        "",
        f"Report status: **{metadata['report_status']}**  ",
        f"Candidate mode: **{metadata['candidate_mode']}**  ",
        f"Candidate pool size: **{metadata['candidate_pool_size']}**  ",
        f"Metric cutoff: **K={metadata['metric_k']}**  ",
        f"MMR alpha: **{metadata['mmr_alpha']}**",
        "",
        "The primary control and candidate-aware model use the identical quota pool. MMR is a",
        "secondary result. The full-vocabulary GRU reference is not an equal control.",
        "",
        "| Model | Recall@20 | MRR@20 | NDCG@20 | Coverage@20 | Latency/query (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_ORDER:
        result = payload["models"][model_name]
        metrics = result["metrics"]["20"]
        lines.append(
            f"| {model_name} | {metrics['recall']:.6f} | {metrics['mrr']:.6f} | "
            f"{metrics['ndcg']:.6f} | {metrics['coverage']:.6f} | "
            f"{result['average_latency_ms']:.4f} |"
        )
    reference = payload.get("secondary_full_vocabulary_reference")
    if isinstance(reference, dict):
        metrics = reference["metrics"]["20"]
        lines.extend(
            [
                "",
                "## Secondary diagnostic: full-vocabulary GRU",
                "",
                "This result has different candidate availability and is not directly comparable",
                "to the controlled quota-pool rows.",
                "",
                f"Recall@20={metrics['recall']:.6f}, MRR@20={metrics['mrr']:.6f}, "
                f"NDCG@20={metrics['ndcg']:.6f}, Coverage@20={metrics['coverage']:.6f}.",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_full_vocabulary_reference(reports_dir: Path, split_name: str):
    path = reports_dir / f"results_{split_name}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("splits", {}).get(split_name, {}).get("GRU4Rec")


def validation_gate_passes(
    reports_dir: Path,
    candidate_checkpoint_hash: str,
    config_hash: str,
) -> tuple[bool, str]:
    path = reports_dir / "candidate_gru_validation.json"
    if not path.is_file():
        return False, "candidate-GRU validation report is missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("candidate_gru_checkpoint_sha256") != candidate_checkpoint_hash:
        return False, "candidate-GRU checkpoint does not match the finalized validation report"
    if metadata.get("candidate_gru_config_sha256") != config_hash:
        return False, "candidate-GRU config does not match the finalized validation report"
    models = payload.get("models", {})
    control_mrr = models.get(CONTROL_NAME, {}).get("metrics", {}).get("20", {}).get("mrr")
    candidate_mrr = models.get(MAIN_NAME, {}).get("metrics", {}).get("20", {}).get("mrr")
    if control_mrr is None or candidate_mrr is None:
        return False, "validation report does not contain the controlled MRR@20 values"
    if float(candidate_mrr) <= float(control_mrr):
        return False, "candidate-aware GRU did not improve validation MRR@20 over the control"
    return True, "validation MRR@20 gate passed"


def evaluate_candidate_gru_suite(
    split_name: str,
    examples: pd.DataFrame,
    catalog_size: int,
    generator,
    current_checkpoint: Path,
    candidate_checkpoint: Path,
    config_path: Path,
    reports_dir: Path,
    device,
    mmr_alpha: float = 0.8,
) -> tuple[Path, Path, dict[str, object]]:
    """Evaluate the controlled quota-pool experiment and save required metadata."""
    current_hash = sha256_file(current_checkpoint)
    candidate_hash = sha256_file(candidate_checkpoint)
    config_hash = sha256_file(config_path)
    if split_name == "test":
        passed, reason = validation_gate_passes(reports_dir, candidate_hash, config_hash)
        if not passed:
            raise RuntimeError(f"Observed-test evaluation refused: {reason}")

    current_model = CandidateGRU.load_checkpoint(current_checkpoint).to(device)
    candidate_model = CandidateGRU.load_checkpoint(candidate_checkpoint).to(device)
    models = evaluate_on_shared_candidate_pools(
        current_model,
        candidate_model,
        generator,
        examples,
        catalog_size,
        device,
        mmr_alpha,
    )
    payload = {
        "metadata": {
            "split": split_name,
            "report_status": "validation" if split_name == "validation" else "observed_test",
            "metric_k": 20,
            "candidate_mode": generator.candidate_mode,
            "candidate_pool_size": generator.candidate_pool_size,
            "base_quotas": dict(generator.quota_weights),
            "resolved_quotas": generator.resolved_quotas(),
            "mmr_alpha": float(mmr_alpha),
            "current_gru_checkpoint_sha256": current_hash,
            "candidate_gru_checkpoint_sha256": candidate_hash,
            "candidate_gru_config_sha256": config_hash,
            "number_of_examples": len(examples),
            "unknown_targets": int(examples["target_item"].eq(0).sum()),
            "latency_mode": "batched_throughput",
        },
        "models": models,
        "secondary_full_vocabulary_reference": _load_full_vocabulary_reference(
            reports_dir, split_name
        ),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"candidate_gru_{split_name}.json"
    markdown_path = reports_dir / f"candidate_gru_{split_name}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_candidate_report(payload), encoding="utf-8")
    return json_path, markdown_path, payload


def write_candidate_gru_diagnostics(reports_dir: Path) -> Path:
    """Summarize validation effects and optional observed-test status without overclaiming."""
    validation_path = reports_dir / "candidate_gru_validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError("candidate-GRU validation report is required for diagnostics")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    models = validation["models"]
    control = models[CONTROL_NAME]["metrics"]["20"]
    candidate = models[MAIN_NAME]["metrics"]["20"]
    mmr = models[MMR_NAME]["metrics"]["20"]
    quota_path = reports_dir / "candidate_recall_quota_validation.json"
    quota = json.loads(quota_path.read_text(encoding="utf-8")) if quota_path.is_file() else None
    baseline_path = reports_dir / "results_validation.json"
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))["splits"]["validation"]
    candidate_mrr_gain = float(candidate["mrr"]) - float(control["mrr"])
    gate_passed = candidate_mrr_gain > 0
    training_log_path = reports_dir / "candidate_gru_training_log.json"
    training_log = (
        json.loads(training_log_path.read_text(encoding="utf-8"))
        if training_log_path.is_file()
        else []
    )
    checkpoint_path = Path("artifacts/candidate_gru_best.pt")
    checkpoint_metadata = (
        CandidateGRU.load_checkpoint(checkpoint_path).checkpoint_metadata
        if checkpoint_path.is_file()
        else {}
    )
    cache_metadata = checkpoint_metadata.get("cache_metadata", {})
    train_cache = cache_metadata.get("train_cache", {})
    validation_cache = cache_metadata.get("validation_cache", {})
    training_config = checkpoint_metadata.get("training_config", {})
    best_checkpoint_mrr = checkpoint_metadata.get("best_validation_metrics", {}).get(
        "mrr", float("nan")
    )
    cache_dir = Path(str(training_config.get("cache_dir", "data/interim/candidate_gru")))
    full_cache_bytes = sum(
        path.stat().st_size for path in cache_dir.glob("full_*") if path.is_file()
    )
    unknown_targets = int(validation["metadata"]["unknown_targets"])
    eligible_examples = int(validation["metadata"]["number_of_examples"])
    unknown_rate = unknown_targets / eligible_examples if eligible_examples else 0.0
    lines = [
        "# Candidate-GRU Diagnostics",
        "",
        "## Validation decision",
        "",
        f"Candidate-aware MRR@20 change versus the same-pool current GRU: "
        f"**{candidate_mrr_gain:+.6f}**.",
        "",
        f"Observed-test gate: **{'passed' if gate_passed else 'failed'}**.",
        "",
        "| Controlled model | Recall@20 | MRR@20 | NDCG@20 | Coverage@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in ((CONTROL_NAME, control), (MAIN_NAME, candidate), (MMR_NAME, mmr)):
        lines.append(
            f"| {name} | {metrics['recall']:.6f} | {metrics['mrr']:.6f} | "
            f"{metrics['ndcg']:.6f} | {metrics['coverage']:.6f} |"
        )
    if quota is not None:
        lines.extend(["", "## Candidate-generation effect", ""])
        for cutoff in (100, 500, 1000):
            old = quota["recall"]["combined"][str(cutoff)]
            new = quota["recall"]["quota_combined"][str(cutoff)]
            lines.append(
                f"- Recall@{cutoff}: old combined {old:.6f}, quota combined {new:.6f}, "
                f"change {new - old:+.6f}."
            )
    lines.extend(
        [
            "",
            "## Training and cache evidence",
            "",
            f"- Architecture: embedding {training_config.get('embedding_dim', 'unknown')}, "
            f"hidden {training_config.get('hidden_dim', 'unknown')}, "
            f"GRU layers {training_config.get('num_layers', 'unknown')}, sampled candidate "
            "cross-entropy with the positive at class 0.",
            f"- Best checkpoint epoch: {checkpoint_metadata.get('best_epoch', 'unknown')}; "
            f"checkpoint-selection MRR@20: "
            f"{best_checkpoint_mrr:.6f} "
            f"on {checkpoint_metadata.get('checkpoint_validation_examples', 'unknown')} fixed "
            "validation examples.",
            f"- Final attempted epoch loss: "
            f"{training_log[-1]['train_loss'] if training_log else float('nan'):.6f} "
            f"after {len(training_log)} recorded epochs.",
            f"- Train hard-negative cache: {int(train_cache.get('number_of_examples', 0)):,} "
            f"examples; natural in-sample target-in-pool rate "
            f"{float(train_cache.get('natural_target_recall', 0.0)):.6f}.",
            f"- Fixed checkpoint-validation pool: "
            f"{int(validation_cache.get('number_of_examples', 0)):,} examples; natural "
            f"target-in-pool rate {float(validation_cache.get('natural_target_recall', 0.0)):.6f}.",
            f"- Full cache footprint: {full_cache_bytes / (1024 ** 2):.1f} MiB. Cache identity "
            "is tied to mapping, train source, examples, candidate config, quotas, and seed.",
            "- Neural training and controlled scoring used the configured automatic accelerator; "
            "this completed on Apple MPS for the recorded run.",
        ]
    )
    lines.extend(["", "## Baseline comparison", ""])
    for baseline_name in ("Markov", "Item-KNN"):
        metrics = baselines[baseline_name]["metrics"]["20"]
        won = [
            metric
            for metric in ("recall", "mrr", "ndcg", "coverage")
            if float(candidate[metric]) > float(metrics[metric])
        ]
        lines.append(
            f"- Versus {baseline_name}, candidate-aware GRU wins: "
            f"{', '.join(won) if won else 'no reported K=20 metric'}."
        )
    lines.extend(
        [
            "",
            "## Attribution and limitations",
            "",
            f"- Neural-training effect is measured by the same-pool MRR change "
            f"({candidate_mrr_gain:+.6f}).",
            f"- Fixed-alpha MMR effect on MRR@20 is "
            f"{float(mmr['mrr']) - float(candidate['mrr']):+.6f}; "
            "MMR is secondary and uses co-visitation similarity, not category diversity.",
            "- Candidate recall bounds every quota-pool neural model; unknown targets remain "
            "misses.",
            f"- Validation contains {unknown_targets:,}/{eligible_examples:,} unknown targets "
            f"({unknown_rate:.2%}), imposing an unknown-item ceiling before candidate retrieval "
            "and ranking errors.",
            f"- Batched validation latency/query: control "
            f"{models[CONTROL_NAME]['average_latency_ms']:.4f} ms, candidate-aware "
            f"{models[MAIN_NAME]['average_latency_ms']:.4f} ms, and candidate-aware + MMR "
            f"{models[MMR_NAME]['average_latency_ms']:.4f} ms. These are offline throughput "
            "measurements, not production service latency.",
            "- The near-perfect train target-in-pool rate is in-sample and must not be read as "
            "held-out candidate recall.",
            "- The existing test split is already observed and cannot provide unbiased final "
            "proof.",
            "- No production, business-uplift, or guaranteed neural-win claim is supported.",
        ]
    )
    test_path = reports_dir / "candidate_gru_test.json"
    lines.append(
        f"- Observed-test candidate-GRU evaluation was "
        f"{'run' if test_path.is_file() else 'not run'} under the validation gate."
    )
    if test_path.is_file():
        test_payload = json.loads(test_path.read_text(encoding="utf-8"))
        test_models = test_payload["models"]
        lines.append(
            f"- Observed-test MRR@20 was "
            f"{test_models[MAIN_NAME]['metrics']['20']['mrr']:.6f} for candidate-aware GRU "
            f"versus {test_models[CONTROL_NAME]['metrics']['20']['mrr']:.6f} for the same-pool "
            "control; this is an observed diagnostic, not unbiased final proof."
        )
    output_path = reports_dir / "candidate_gru_diagnostics.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
