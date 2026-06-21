# Candidate-GRU Diagnostics

## Validation decision

Candidate-aware MRR@20 change versus the same-pool current GRU: **+0.052322**.

Observed-test gate: **passed**.

| Controlled model | Recall@20 | MRR@20 | NDCG@20 | Coverage@20 |
|---|---:|---:|---:|---:|
| Current GRU - quota_combined | 0.101691 | 0.038378 | 0.052274 | 0.142076 |
| Candidate-aware GRU - quota_combined | 0.226398 | 0.090700 | 0.121109 | 0.511019 |
| Candidate-aware GRU - quota_combined + MMR | 0.211674 | 0.086545 | 0.114499 | 0.492414 |

## Candidate-generation effect

- Recall@100: old combined 0.303724, quota combined 0.350049, change +0.046325.
- Recall@500: old combined 0.396261, quota combined 0.429599, change +0.033337.
- Recall@1000: old combined 0.433444, quota combined 0.454429, change +0.020985.

## Training and cache evidence

- Architecture: embedding 128, hidden 128, GRU layers 1, sampled candidate cross-entropy with the positive at class 0.
- Best checkpoint epoch: 5; checkpoint-selection MRR@20: 0.091003 on 20000 fixed validation examples.
- Final attempted epoch loss: 1.184450 after 10 recorded epochs.
- Train hard-negative cache: 421,798 examples; natural in-sample target-in-pool rate 0.999993.
- Fixed checkpoint-validation pool: 20,000 examples; natural target-in-pool rate 0.461000.
- Full cache footprint: 489.1 MiB. Cache identity is tied to mapping, train source, examples, candidate config, quotas, and seed.
- Neural training and controlled scoring used the configured automatic accelerator; this completed on Apple MPS for the recorded run.

## Baseline comparison

- Versus Markov, candidate-aware GRU wins: recall, coverage.
- Versus Item-KNN, candidate-aware GRU wins: mrr, ndcg.

## Attribution and limitations

- Neural-training effect is measured by the same-pool MRR change (+0.052322).
- Fixed-alpha MMR effect on MRR@20 is -0.004155; MMR is secondary and uses co-visitation similarity, not category diversity.
- Candidate recall bounds every quota-pool neural model; unknown targets remain misses.
- Validation contains 14,764/88,159 unknown targets (16.75%), imposing an unknown-item ceiling before candidate retrieval and ranking errors.
- Batched validation latency/query: control 5.0440 ms, candidate-aware 5.0381 ms, and candidate-aware + MMR 9.3146 ms. These are offline throughput measurements, not production service latency.
- The near-perfect train target-in-pool rate is in-sample and must not be read as held-out candidate recall.
- The existing test split is already observed and cannot provide unbiased final proof.
- No production, business-uplift, or guaranteed neural-win claim is supported.
- Observed-test candidate-GRU evaluation was run under the validation gate.
- Observed-test MRR@20 was 0.073671 for candidate-aware GRU versus 0.030761 for the same-pool control; this is an observed diagnostic, not unbiased final proof.
