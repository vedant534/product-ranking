# GRU Diagnostics

This report describes the saved `artifacts/gru4rec_best.pt` checkpoint and the current full-data
reports. It separates observed facts from experiments that have not yet been run.

## Executive diagnosis

The current GRU is improving, but it does not beat the Markov or Item-KNN baselines. Training loss
fell every epoch and checkpoint MRR@20 increased every epoch; epoch 5 was both the final and best
epoch. Therefore, there is no evidence that the model has converged or that GRU-based ranking
cannot improve. The evidence instead indicates that the present run is an intentionally small,
lightly trained reference model.

The direct full-vocabulary GRU reaches validation Recall@20 of 0.072664 and test Recall@20 of
0.057691. The hybrid GRU plus MMR path reaches 0.111548 and 0.089588 respectively, but this hybrid
still trails the classical baselines. The hybrid result must not be interpreted as a pure neural
win: it depends on classical candidate generation and co-visitation-based reranking.

## Architecture

| Component | Saved checkpoint value |
|---|---:|
| Train-catalog items | 112,714 |
| Output classes including `<UNK_ITEM>` | 112,715 |
| Padding index | 112,715, outside output classes |
| Item embedding dimension | 32 |
| GRU hidden dimension | 32 |
| GRU layers | 1 |
| Dropout | 0.1 |
| Maximum prefix length | 50 |
| Trainable parameters | 7,332,843 |

The model is `item ID -> embedding -> GRU -> linear catalog logits`. It has no attention, event
type embedding, category feature, time-gap feature, user embedding, or item-content feature. UNK
and items already present in the prefix are excluded from recommendations.

## Training configuration

| Setting | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 512 |
| Configured epochs | 5 |
| Early-stopping patience | 2 |
| Gradient clipping | 5.0 |
| Random seed | 42 |
| Device setting | `auto` |
| Resolved full-run device | Apple MPS |
| Sampled negatives per batch | 1,024 |
| Checkpoint validation sample | 20,000 examples |

Training used sampled cross-entropy, not full softmax. Each batch scored all batch targets plus
1,024 shared, uniformly sampled train-vocabulary negatives. Full-softmax mode exists in code when
`sampled_softmax_negatives` is zero, but it was not used for this full run. Checkpoint validation
and reported direct GRU evaluation use exact full-catalog top-K scoring.

## Data volume

| Split | Generated examples | Ranking-eligible examples | Use in current GRU workflow |
|---|---:|---:|---|
| Train | 713,456 | 421,798 | All eligible examples used for optimization |
| Validation | 140,579 | 88,159 | Fixed 20,000-example subset used for checkpoint selection; all 88,159 used in final reporting |
| Test | 140,176 | 92,892 | All eligible examples used only for reporting |

The item mapping is fitted on train only. Validation has 14,764 unknown targets and test has
23,515 unknown targets. They remain in the denominator and cannot be recommended. This creates an
unavoidable Recall ceiling of 0.8325 on validation and 0.7469 on test for every train-catalog
model.

## Training trajectory

| Epoch | Train sampled loss | Validation Recall@20 | Validation MRR@20 | Validation NDCG@20 |
|---:|---:|---:|---:|---:|
| 1 | 7.167957 | 0.015400 | 0.006148 | 0.008181 |
| 2 | 6.783954 | 0.031850 | 0.012901 | 0.017081 |
| 3 | 6.392461 | 0.048450 | 0.019634 | 0.025995 |
| 4 | 5.995992 | 0.062200 | 0.025427 | 0.033562 |
| 5 | 5.649377 | 0.074100 | 0.030719 | 0.040307 |

The final available training loss is 5.649377. Because this is sampled cross-entropy, it is not a
full-catalog negative log-likelihood and should only be compared with losses from the same sampled
training setup.

The saved best checkpoint is epoch 5, selected by MRR@20 = 0.030719 on the fixed 20,000-example
validation subset. Its exact full-validation MRR@20 is 0.030282. The monotonic loss and metric
curves are evidence for undertraining, not a plateau.

## Full-vocabulary and candidate-pool scoring

There are two distinct inference paths:

1. **Direct `GRU4Rec` offline evaluation:** scores the complete 112,714-item train catalog, then
   excludes UNK and prefix items. Candidate recall does not constrain this result.
2. **Streamlit and `GRU4Rec + MMR`:** Popularity, Markov, and Item-KNN build a shared pool, the GRU
   scores only those candidates, and optional MMR reranks them using Item-KNN co-visitation
   similarity. The default shared pool contains at most 500 items.

The offline `GRU4Rec` row therefore is not the same serving path as the demo's candidate-scored GRU.
A shared-pool GRU result without MMR is not currently recorded as a separate offline row. That
missing comparison should be added before attributing the hybrid improvement specifically to GRU
scoring or to MMR.

## Candidate recall and its effect

| Split | Pool | Recall@100 | Recall@500 | Recall@1000 |
|---|---|---:|---:|---:|
| Validation | Popularity | 0.027507 | 0.079992 | 0.123039 |
| Validation | Markov | 0.222450 | 0.222870 | 0.222870 |
| Validation | Item-KNN | 0.349709 | 0.420309 | 0.432038 |
| Validation | Combined | 0.303724 | 0.396261 | 0.433444 |
| Test | Popularity | 0.024276 | 0.066787 | 0.102485 |
| Test | Markov | 0.173718 | 0.173987 | 0.173987 |
| Test | Item-KNN | 0.283674 | 0.344508 | 0.354315 |
| Test | Combined | 0.245769 | 0.324517 | 0.356629 |

At the demo's 500-item setting, the true target is available for only 39.63% of validation examples
and 32.45% of test examples. These values are hard upper bounds for candidate-scored GRU Recall at
any K. GRU plus MMR moves about 28% of candidate-available targets into the top 20 on both splits
(0.111548 / 0.396261 on validation and 0.089588 / 0.324517 on test). This leaves headroom in both
candidate generation and ranking.

The combined pool has lower Recall@500 than native Item-KNN on both splits. The normalized merge is
discarding some useful Item-KNN candidates to make room for weaker Popularity or Markov candidates.
Increasing the pool from 500 to 1,000 improves combined recall by only 3.72 percentage points on
validation and 3.21 points on test, so pool size alone is not a complete fix.

## Comparison with classical baselines

| Split | Model | Recall@20 | MRR@20 | NDCG@20 |
|---|---|---:|---:|---:|
| Validation | Markov | 0.210438 | 0.101537 | 0.126578 |
| Validation | Item-KNN | 0.230867 | 0.088170 | 0.119928 |
| Validation | Direct full-vocabulary GRU | 0.072664 | 0.030282 | 0.039669 |
| Validation | GRU + MMR hybrid | 0.111548 | 0.041168 | 0.056530 |
| Test | Markov | 0.165655 | 0.079811 | 0.099538 |
| Test | Item-KNN | 0.189338 | 0.072020 | 0.098112 |
| Test | Direct full-vocabulary GRU | 0.057691 | 0.024318 | 0.031724 |
| Test | GRU + MMR hybrid | 0.089588 | 0.032692 | 0.045109 |

Likely reasons for underperformance are:

- **Training stopped while metrics were still rising.** Five epochs are insufficient evidence of
  convergence, and the best checkpoint is the last checkpoint.
- **Low capacity relative to catalog size.** A 32-dimensional embedding and 32-unit hidden state
  must separate more than 112,000 output classes.
- **Sampled-loss mismatch.** Only 1,024 shared random negatives are contrasted per batch, while
  final inference compares against the entire catalog. Easy uniform negatives may not teach the
  fine distinctions needed at the top of a large ranking.
- **Sparse long-tail supervision.** There are 421,798 eligible training examples for 112,714 train
  items, and the frequency distribution is highly skewed. Many output classes have little direct
  supervision.
- **Short sessions favor memorization.** Average eligible prefixes are roughly five to six items.
  First-order transitions and co-visitation directly memorize the dominant local patterns and need
  far less data to estimate them.
- **Restricted inputs.** The GRU sees item IDs only. It cannot use event type, time gap, category,
  or item-property signals that might disambiguate otherwise similar prefixes.
- **Candidate bottleneck in the demo path.** Most test targets never enter the 500-item shared pool,
  and the current merge reduces Item-KNN's native candidate recall.

These are evidence-supported hypotheses, not proven causal explanations. Controlled validation
experiments are required before claiming any one change fixes the gap.

## Honest next experiments

None of the following improvements has been implemented or validated unless explicitly stated.

1. **Train longer.** Raise the epoch limit to approximately 15-30 with patience 3-5, retain
   validation-MRR checkpointing, and stop only after an actual plateau. This is the highest-priority
   experiment because the current curve is still improving.
2. **Tune capacity and optimization.** Compare embedding/hidden sizes 64 and 128, learning rates,
   dropout, batch size, and gradient clipping using validation only. Report compute and latency
   costs alongside metrics.
3. **Check sampled versus full softmax.** Run a controlled subset experiment with full softmax and
   smaller batches, then compare it with 1,024-negative sampled training under the same data and
   seed. If full softmax is impractical, test more negatives and popularity- or hard-negative
   sampling. Full-softmax training mode exists but has not been tested on the full dataset.
4. **Record the missing shared-pool GRU ablation.** Evaluate candidate-scored GRU without MMR, then
   compare direct full-vocabulary GRU, shared-pool GRU, and shared-pool GRU plus MMR. This isolates
   candidate loss, neural ordering, and reranking effects.
5. **Improve candidate generation.** Preserve Item-KNN candidates with per-source quotas or a
   union-first policy, tune source weights on validation, and measure candidate recall before model
   recall. A larger pool can be tested, but current Recall@1000 shows limited gain by itself.
6. **Align serving with full-vocabulary neural scoring.** Direct full-vocabulary scoring is already
   implemented for offline evaluation. Benchmark it in the interactive/serving path, or use neural
   top-N retrieval before classical augmentation, so the demo is not completely bounded by the
   current classical pool.
7. **Try SASRec as a separate model.** Self-attention may model multiple relevant prefix positions
   more directly than one final GRU state. SASRec is not implemented, and it should be compared
   under the same train-only vocabulary, chronological splits, eligibility policy, and evaluator.
8. **Add richer sequence signals only after the core ablations.** Event type and time-gap embeddings
   are plausible extensions. Category metadata should not be presented as a diversity signal and
   must be cutoff-safe if used for modeling.

The current evidence supports further validation experiments, not a claim that GRU will eventually
beat the baselines. It is also a valid outcome if a properly tuned sequential neural model remains
behind Item-KNN or Markov on this sparse, short-session benchmark.
