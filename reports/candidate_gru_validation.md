# Candidate-GRU Results: Validation

Report status: **validation**  
Candidate mode: **quota_combined**  
Candidate pool size: **1000**  
Metric cutoff: **K=20**  
MMR alpha: **0.8**

The primary control and candidate-aware model use the identical quota pool. MMR is a
secondary result. The full-vocabulary GRU reference is not an equal control.

| Model | Recall@20 | MRR@20 | NDCG@20 | Coverage@20 | Latency/query (ms) |
|---|---:|---:|---:|---:|---:|
| Current GRU - quota_combined | 0.101691 | 0.038378 | 0.052274 | 0.142076 | 5.0440 |
| Candidate-aware GRU - quota_combined | 0.226398 | 0.090700 | 0.121109 | 0.511019 | 5.0381 |
| Candidate-aware GRU - quota_combined + MMR | 0.211674 | 0.086545 | 0.114499 | 0.492414 | 9.3146 |

## Secondary diagnostic: full-vocabulary GRU

This result has different candidate availability and is not directly comparable
to the controlled quota-pool rows.

Recall@20=0.072664, MRR@20=0.030282, NDCG@20=0.039669, Coverage@20=0.102152.
