# Quota Candidate Recall: Validation

Report status: **validation**  
Examples: **88,159**  
Unknown targets counted as misses: **14,764**  
Fit split: **train**  
Candidate mode: **quota_combined**

Base quota weights are Item-KNN 700, Markov 200, and Popularity 100. Quotas
are scaled to each cutoff using largest-remainder rounding.

| Candidate source | Recall@100 | Recall@500 | Recall@1000 |
|---|---:|---:|---:|
| Item-KNN | 0.349709 | 0.420309 | 0.432038 |
| Combined | 0.303724 | 0.396261 | 0.433444 |
| Quota combined | 0.350049 | 0.429599 | 0.454429 |
