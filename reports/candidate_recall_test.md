# Candidate Recall: Test

Examples: **92,892**  
Unknown targets counted as misses: **23,515**  
Fit split: **train**

Markov and Item-KNN rows use native candidates without popularity fallback. The
combined row uses the serving score normalization and is truncated to exactly K items.

| Candidate source | Recall@100 | Recall@500 | Recall@1000 |
|---|---:|---:|---:|
| Popularity | 0.024276 | 0.066787 | 0.102485 |
| Markov | 0.173718 | 0.173987 | 0.173987 |
| Item-KNN | 0.283674 | 0.344508 | 0.354315 |
| Combined | 0.245769 | 0.324517 | 0.356629 |
