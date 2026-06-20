# Project Specification: Real-Time Session-Based Product Ranking for Anonymous Users

## 1. Project Summary

Build an offline-trained, low-latency recommendation system that ranks the next product an anonymous shopper is likely to interact with during the current session. The system will use the RetailRocket e-commerce dataset, compare three non-neural baselines with a simple GRU4Rec-style PyTorch model, and expose recommendations through a Streamlit demo.

The primary offline result must use a chronological train/validation/test split. A random split may be used only as a clearly labeled diagnostic and must not be presented as the main result.

## 2. Objectives

### Primary objective

Given an ordered session prefix

```text
[item_1, item_2, ..., item_t]
```

return a ranked list of `K` item IDs that predicts `item_(t+1)`.

### Success criteria

The completed project must:

1. Reproduce preprocessing and chronological splits from configuration and a fixed seed.
2. Train and evaluate Most Popular, Item-KNN/co-visitation, first-order Markov, and GRU4Rec-style models through one evaluation interface.
3. Report Recall@20, MRR@20, NDCG@20, HitRate@10, catalog coverage, and recommendation latency.
4. Demonstrate that the GRU model runs end to end; it is not required to beat every baseline, but results and tradeoffs must be reported honestly.
5. Serve top-K recommendations in a Streamlit app for an existing or manually built session.
6. Prevent future interactions and future item metadata from leaking into training or validation.

## 3. Scope

### In scope

- Anonymous, within-session next-item ranking.
- Event sequence preprocessing and configurable sessionization.
- Offline training, tuning, evaluation, and model comparison.
- Popularity, co-visitation, Markov, and GRU-based recommenders.
- CPU-compatible inference and latency measurement.
- A local Streamlit demonstration.
- Basic item/category context from RetailRocket metadata where available.

### Non-goals

- Identifying users across visitors, devices, or sessions.
- Long-term personalization or user-profile modeling.
- Production deployment, distributed serving, autoscaling, or online experimentation.
- Optimizing revenue, margin, diversity, fairness, or business rules.
- Causal claims about recommendation impact.
- Rich semantic product understanding; RetailRocket item properties are anonymized.
- Rebuilding GRU4Rec exactly as described in the original paper or reproducing its published results.
- Using transformers, large language models, graph neural networks, or external product data in the required implementation.

## 4. Dataset

Use the RetailRocket e-commerce dataset files:

| File | Expected fields | Use |
|---|---|---|
| `events.csv` | `timestamp`, `visitorid`, `event`, `itemid`, `transactionid` | Required source of ordered interactions and next-item labels |
| `item_properties.csv` or sharded equivalents | `timestamp`, `itemid`, `property`, `value` | Optional item context, catalog filtering, analysis, and demo display |
| `category_tree.csv` | `categoryid`, `parentid` | Optional category hierarchy for analysis and demo display |

Raw data must remain immutable. The ingestion layer must validate required columns, parse timestamps consistently as UTC milliseconds, report malformed/null rows, and write processed artifacts separately.

### Event policy

- Include `view`, `addtocart`, and `transaction` events in the default sequence.
- Treat the prediction target as the next interacted item, independent of event type.
- Preserve event type for descriptive analysis, but the required GRU takes item IDs only.
- Sort by `visitorid`, `timestamp`, then original row order to make timestamp ties deterministic.
- Preserve all allowed interactions. The implemented unseen-item ranking task excludes train-known
  targets already present in their prefix from training and evaluation, and every recommender
  excludes prefix items from its output. The dataset statistics report how many examples this
  eligibility rule removes.

### Metadata policy

The required models must remain runnable from `events.csv` alone. If time-varying item properties are used, select only the most recent value available at or before the relevant cutoff. Never use a property's future value to construct training or evaluation features. Category/property values may be missing or anonymized; the UI must fall back to the item ID.

## 5. Session Construction

Although `visitorid` is anonymous, one visitor may have multiple sessions. Build sessions as follows:

1. Group events by `visitorid` and order them using the event policy above.
2. Start a new session when the gap from the previous event is greater than 30 minutes. Make the inactivity threshold configurable.
3. Assign a deterministic session ID after globally sorting visitors and timestamps.
4. Retain sessions with at least two item interactions after filtering so that each session has at least one context-target pair.

Persist a sessionized events table containing at least:

```text
session_id, visitorid, timestamp, event, itemid, transactionid, session_position
```

Also persist session count, session-length statistics, and event-type counts.

## 6. Chronological Data Split

### Main split

Use a global, session-level chronological split to avoid placing prefixes from one session in multiple partitions:

1. Sort sessions by `session_start`, with `session_id` as a deterministic tie-breaker.
2. Assign the earliest 70% of sessions to train, the next 15% to validation, and the final 15% to test.
3. Purge boundary-crossing sessions so every retained train event precedes every validation event
   and every retained validation event precedes every test event.
4. Store boundaries and aggregate split diagnostics in `data/processed/split_summary.json`; the
   split label on each processed example records its assignment.
5. Fit vocabularies, popularity counts, neighbors, transition counts, model parameters, and item
   similarities only on the training partition.
6. Select the stopping epoch on validation MRR@20.
7. Evaluate frozen artifacts on held-out data.

Purged boundary-spanning sessions appear in no partition. Report each retained partition's event
time range, session count, event count, item count, and visitor count.

### Training and evaluation examples

For a sequence `[A, B, C, D]`, generate prefix-target examples:

```text
[A]       -> B
[A, B]    -> C
[A, B, C] -> D
```

Training may batch all next-step targets efficiently by shifting complete sequences. Validation
and test metrics are macro-averaged over eligible prefix-target examples, not just the last event
of each session. A prefix is truncated from the left when it exceeds the configured sequence limit.

The candidate catalog is the set of item IDs observed in training. Validation/test items outside
that catalog map explicitly to `<UNK_ITEM>`; this class is never recommended, so unknown targets
remain misses. Train-known targets already present in a prefix are excluded consistently from the
unseen-item ranking objective. Both unknown and eligibility counts are recorded in
`dataset_stats.json`.

## 7. Recommender Interface

All implementations must expose equivalent behavior:

```python
fit(train_sessions) -> None
recommend(session_item_ids, k) -> list[tuple[item_id, score]]
save(path) -> None
load(path) -> Recommender
```

Recommendations must be deterministic for a fixed artifact and input. Resolve equal scores by a documented stable rule, preferably global popularity followed by numeric item ID. If a model has fewer than `K` candidates, backfill with unseen entries from the training popularity list while avoiding duplicate outputs.

## 8. Required Baselines

### 8.1 Most Popular

Rank items by training interaction count. Use recency or event weighting only as optional validation-tuned variants; the required reference result uses unweighted counts. This is also the fallback for empty sessions and unknown input items.

### 8.2 Item-KNN / Co-visitation

Build sparse item-item co-visitation counts from training sessions only.

- Count item pairs within a configurable positional window, default `5`.
- Prevent long sessions from dominating by capping session length during pair generation, default `50` most recent items.
- Normalize co-occurrence using cosine similarity by default.
- Retain a configurable top number of neighbors per item, default `200`.
- Score candidates by the similarity-weighted sum over the most recent session items, optionally applying a validation-tuned recency decay.
- Fall back to popularity when the session contains no known item or neighbors are insufficient.

### 8.3 First-Order Markov Chain

Estimate item-to-item transition counts from adjacent item pairs in training sessions. Rank next items by smoothed or normalized transition probability from the last known session item. Use popularity for missing states and backfill.

The required baseline is first order only. Higher-order variants may appear as optional experiments but cannot replace it.

## 9. Required GRU4Rec-Style Model

Implement a simple PyTorch sequential model with:

```text
item ID -> embedding -> single-layer GRU -> linear output over training items
```

### Minimum architecture

- Train-only item vocabulary plus `PAD` and `UNK` input tokens.
- Item embedding dimension: configurable; the full-data reference config uses `32`.
- One GRU layer with configurable hidden size; the full-data reference config uses `32`.
- Dropout: default `0.1` where applicable.
- Output logits over the train candidate catalog.
- Cross-entropy next-item loss with padding ignored.
- Adam optimizer, default learning rate `1e-3`.
- Gradient clipping, default maximum norm `5.0`.
- Padded mini-batches with masks and configurable maximum sequence length, default `50` most recent items.
- Early stopping on validation MRR@20 with configurable patience.
- Best-validation checkpoint restoration before test evaluation.

Use a fixed seed for Python, NumPy, and PyTorch. CPU execution must work; automatic device
selection prefers CUDA, then Apple MPS, then CPU. Keep the model intentionally simple: no
attention, user embeddings, category features, or external product data.

The full-data reference uses sampled cross-entropy for tractable training and exact full-catalog
ranking for checkpoint validation and final inference. Setting the negative-sample count to zero
retains full-softmax training mode.

## 10. Evaluation Protocol

For each eligible prefix, request at least the top 20 ranked items. Let `r` be the one-based rank of the true next item, or infinity when absent.

| Metric | Definition |
|---|---|
| Recall@20 | `1` if `r <= 20`, else `0`, averaged over examples; with one relevant next item this equals HitRate@20 |
| MRR@20 | `1/r` if `r <= 20`, else `0`, averaged over examples |
| NDCG@20 | `1/log2(r + 1)` if `r <= 20`, else `0`, averaged over examples |
| HitRate@10 | `1` if `r <= 10`, else `0`, averaged over examples |
| Coverage@20 | Number of distinct items appearing in any top-20 test recommendation divided by the train candidate-catalog size |
| Latency | Mean wall-clock recommendation time per evaluated query |

Because there is one relevant item per example, document the equivalence between Recall@K and HitRate@K rather than implying they measure different behavior at the same cutoff.

### Latency measurement

- Load the model before measurement and exclude training, artifact loading, and Streamlit rendering.
- Measure with `time.perf_counter`.
- Classical recommenders report single-query mean latency. Batched GRU evaluation reports
  amortized throughput per query and labels that mode explicitly, so these values are not presented
  as identical serving benchmarks.

### Required reporting

Produce unified K=5/10/20 result tables and a segment report by prefix length, train target
popularity, and target event type. Reports must distinguish MMR co-visitation diversity from
category diversity and must not claim business uplift.

Do not tune model hyperparameters on test results. Development test analysis is descriptive and
must not be represented as a pristine one-shot benchmark.

## 11. Streamlit Demo

Create a local Streamlit app that loads precomputed artifacts and supports two input paths:

1. Select a sample session from held-out data and choose a visible prefix length.
2. Build a session manually from comma- or whitespace-separated item IDs.

The app must provide:

- A side-by-side comparison covering the required baselines, GRU, and GRU plus MMR.
- A configurable `K`, with a practical range of `5-20`.
- The current ordered session.
- Ranked recommendations containing rank, item ID, score, and available category/property context.
- Clear fallback messaging for empty sessions, unknown items, or missing metadata.
- Candidate-source and concise explanation columns.
- A recorded offline-results panel when `reports/results.json` exists.

Sample held-out sessions are for demonstration only and must not be embedded into trained artifacts. The app must perform no training and should cache loaded models/data with Streamlit resource/data caching.

Recommended command:

```bash
python3 -m streamlit run app/streamlit_app.py
```

## 12. Repository Layout

```text
.
├── PROJECT_SPEC.md
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                  # ignored by version control
│   └── processed/            # generated sessions and split manifest
├── artifacts/                # generated trained models and indexes
├── reports/
│   ├── results.json
│   └── results.md
├── src/
│   ├── data/
│   ├── baselines/
│   ├── models/
│   ├── ranking/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── tests/
└── app/streamlit_app.py
```

## 13. Configuration and Reproducibility

The checked-in default configuration must include:

- Raw/processed/artifact paths.
- Random seed.
- Included event types and session gap.
- Session gap and maximum sequence length.
- Split fractions.
- GRU architecture, optimizer, batch size, epoch limit, and early stopping.
- Evaluation cutoffs.

Processed runs emit machine-readable session, split, and dataset summaries. Training checkpoints
store model configuration and validation metadata; evaluation results record split, metrics,
catalog size, example count, and latency mode.

## 14. Testing Requirements

Minimum automated tests:

- Session boundaries at, below, and above the inactivity threshold.
- Stable ordering for equal timestamps.
- No session ID overlap and monotonically ordered chronological partitions.
- Train-only fitting of vocabularies and recommender statistics.
- Hand-calculated Recall, MRR, NDCG, HitRate, and coverage examples.
- Popularity fallback for empty/all-unknown sessions.
- Co-visitation and Markov behavior on a small synthetic dataset.
- GRU tensor shapes, padding masks, one short training pass, checkpoint reload, and top-K output.
- End-to-end component tests using tiny RetailRocket-like CSV fixtures.

## 15. Deliverables

1. Source code and pinned dependency specification.
2. Reproducible preprocessing, training, and evaluation commands.
3. Saved split manifest and model artifacts.
4. Test suite.
5. Results report with all required metrics and latency methodology.
6. Streamlit demo and run instructions.
7. README describing setup, data placement, commands, assumptions, and known limitations.

## 16. Acceptance Criteria

The project is complete when:

- All three required baselines and the GRU4Rec-style model can be trained and loaded from artifacts.
- The same chronological test examples and candidate policy are used for every model.
- The main report contains Recall@20, MRR@20, NDCG@20, HitRate@10, Coverage@20, and latency for every model.
- Split and leakage tests pass, and there is no random split in the main reported result.
- Running the documented Streamlit command allows a user to build/select a session and receive deterministic top-K recommendations.
- Empty, short, unknown-item, and missing-metadata inputs fail gracefully or use the documented popularity fallback.
- A fresh environment can reproduce preprocessing and at least one full training/evaluation run using documented commands.

## 17. Limitations

- Offline next-event accuracy is a proxy and does not establish business value or causal lift.
- RetailRocket interactions are implicit feedback; a view is not necessarily positive preference, and unobserved items are not confirmed negatives.
- Anonymous sessions provide little context at the first interaction, where popularity fallback may dominate.
- Chronological drift and train-unseen items can reduce achievable accuracy, especially in later test periods.
- A global cutoff does not simulate continual retraining or online catalog updates.
- Item properties and categories are anonymized, sparse, and time-varying, limiting explanation quality in the demo.
- Full-catalog GRU scoring may become the latency bottleneck as the catalog grows.
- Coverage measures catalog spread but not novelty, diversity, fairness, inventory validity, or recommendation quality by itself.
- Results are specific to the chosen sessionization, event policy, candidate catalog, and dataset period; comparisons with other work are invalid unless those choices match.
- The Streamlit app is a demonstration, not a production service and not a security-, privacy-, or load-tested deployment.

## 18. Implementation Milestones

1. **Data foundation:** ingestion validation, deterministic sessionization, chronological split, manifests, and tests.
2. **Baselines:** common interface, popularity, co-visitation, Markov, fallbacks, and unit tests.
3. **Neural model:** GRU training loop, validation, early stopping, checkpointing, and inference.
4. **Evaluation:** shared ranking metrics, cold-target reporting, coverage, latency benchmark, and result tables.
5. **Demo:** artifact loading, session selection/builder, ranking display, metadata context, and error states.
6. **Finalization:** end-to-end reproduction, documentation, limitation review, and acceptance test run.
