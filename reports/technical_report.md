# Technical Report

## Data and task

RetailRocket timestamps are parsed as UTC Unix milliseconds. Events are ordered by visitor and
timestamp, then split into sessions after gaps strictly greater than 30 minutes. Sessions shorter
than two interactions are removed.

The main split is chronological and session-level: 70% train, 15% validation, and 15% test.
Boundary-crossing sessions are purged to enforce strict event-time ordering. The retained data has
267,933 train sessions, 57,397 validation sessions, and 57,417 test sessions.

Each session produces immediate next-item prefix-target examples. The train-only mapping contains
112,714 items plus `<UNK_ITEM>` at index 0. Unknown validation/test items map to UNK, which is never
recommended. Known targets already observed in the prefix are excluded consistently from the
unseen-item objective, leaving 421,798 train, 88,159 validation, and 92,892 test examples.

## Models

- Popularity counts train interactions.
- Markov counts adjacent train transitions and falls back to train popularity.
- Item-KNN uses cosine-normalized train-session co-visitation, recency-weighted prefix aggregation,
  and popularity fallback.
- GRU4Rec uses a 32-dimensional item embedding, one 32-unit GRU layer, and a full-catalog output
  layer. The full run used sampled cross-entropy with 1,024 negatives per training example.
- Candidate generation merges Popularity, Markov, and Item-KNN candidates. GRU ranks that shared
  pool in the recommendation engine; direct GRU evaluation also supports exact full-catalog top-K.
- MMR trades GRU relevance against Item-KNN cosine co-visitation similarity. Categories are not
  used in the diversity penalty.

The full GRU run used Apple MPS. Training loss decreased from 7.1680 to 5.6494 over five epochs;
fixed-subset validation MRR@20 increased from 0.00615 to 0.03072. The saved checkpoint is epoch 5.

## Evaluation

All models use the shared metric implementation for Recall, HitRate, MRR, NDCG, and coverage at
K=5, 10, and 20. Validation results are in `reports/results.json` and `reports/results.md`; test
breakdowns are in `reports/segment_analysis.md`.

At validation K=20, Item-KNN has the highest Recall (0.230867), Markov has the highest MRR
(0.101537), and direct GRU Recall is 0.072664. GRU plus MMR reaches Recall 0.111548 and co-visitation
diversity 0.990282, but remains below both sequential classical baselines and is slower.

Latency excludes fitting and artifact loading. Classical models report single-query mean latency;
GRU reports amortized batched throughput per query. These modes are explicitly recorded and should
not be treated as a controlled production latency comparison.

## Reproducibility

Processed summaries record full-data status, split boundaries, eligibility counts, and vocabulary
size. The saved baseline bundle and GRU checkpoint are checked against the processed catalog size
before inference. The Streamlit app reads only processed data and saved artifacts and never fits a
model.
