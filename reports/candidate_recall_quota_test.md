# Quota Candidate Recall: Test

Report status: **observed_test**  
Examples: **92,892**  
Unknown targets counted as misses: **23,515**  
Fit split: **train**  
Candidate mode: **quota_combined**

Base quota weights are Item-KNN 700, Markov 200, and Popularity 100. Quotas
are scaled to each cutoff using largest-remainder rounding.

| Candidate source | Recall@100 | Recall@500 | Recall@1000 |
|---|---:|---:|---:|
| Item-KNN | 0.283674 | 0.344508 | 0.354315 |
| Combined | 0.245769 | 0.324517 | 0.356629 |
| Quota combined | 0.283372 | 0.352603 | 0.375511 |
