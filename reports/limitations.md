# Limitations

- RetailRocket item IDs and properties are anonymized, so the demo cannot show product names or
  semantic explanations.
- Offline next-item metrics do not establish causal impact, revenue uplift, or production value.
- The train-only vocabulary cannot recommend items first appearing in validation or test. The full
  artifacts contain 14,764 unknown validation targets and 23,515 unknown test targets.
- The task excludes train-known targets already present in the prefix. This makes the project an
  unseen-item ranker, not a repeated-consumption predictor.
- The category artifact covers 100,246 of 112,714 train-vocabulary items (88.94%). Categories are
  display metadata only; the MMR penalty uses train-only Item-KNN cosine co-visitation similarity
  and must not be described as category diversity.
- The full-data GRU uses sampled cross-entropy during training. Checkpoint selection and final
  ranking use exact full-catalog logits, but the sampled objective remains an approximation.
- GRU checkpoint selection uses a fixed 20,000-example validation subset for tractability. Final
  validation reports use all 88,159 eligible examples.
- Reported GRU latency is amortized batched throughput, while classical latency is per query. The
  values are labeled but are not a controlled production serving benchmark.
- Validation Item-KNN and Markov outperform the current GRU at Recall@20. This is reported as a
  model result, not hidden by tuning or unsupported claims.
- MMR increases co-visitation diversity and changes rankings, but it is slower and does not
  guarantee better relevance. Its effect depends on alpha and candidate-pool quality.
- The chronological split reflects one historical dataset period and does not simulate online
  catalog refreshes, inventory constraints, delayed feedback, or continual retraining.
- The Streamlit app is a local artifact-only demonstration, not a deployed, load-tested, secured,
  or monitored recommendation service.
