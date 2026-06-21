# Candidate-GRU Results: Test

Report status: **observed_test**  
Candidate mode: **quota_combined**  
Candidate pool size: **1000**  
Metric cutoff: **K=20**  
MMR alpha: **0.8**

The primary control and candidate-aware model use the identical quota pool. MMR is a
secondary result. The full-vocabulary GRU reference is not an equal control.

| Model | Recall@20 | MRR@20 | NDCG@20 | Coverage@20 | Latency/query (ms) |
|---|---:|---:|---:|---:|---:|
| Current GRU - quota_combined | 0.080922 | 0.030761 | 0.041816 | 0.136416 | 4.9158 |
| Candidate-aware GRU - quota_combined | 0.182804 | 0.073671 | 0.098144 | 0.506370 | 4.9121 |
| Candidate-aware GRU - quota_combined + MMR | 0.171619 | 0.070246 | 0.092886 | 0.486089 | 9.1580 |

## Secondary diagnostic: full-vocabulary GRU

This result has different candidate availability and is not directly comparable
to the controlled quota-pool rows.

Recall@20=0.057691, MRR@20=0.024318, NDCG@20=0.031724, Coverage@20=0.100839.
