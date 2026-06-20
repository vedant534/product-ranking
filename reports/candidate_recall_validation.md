# Candidate Recall: Validation

Examples: **88,159**  
Unknown targets counted as misses: **14,764**  
Fit split: **train**

Markov and Item-KNN rows use native candidates without popularity fallback. The
combined row uses the serving score normalization and is truncated to exactly K items.

| Candidate source | Recall@100 | Recall@500 | Recall@1000 |
|---|---:|---:|---:|
| Popularity | 0.027507 | 0.079992 | 0.123039 |
| Markov | 0.222450 | 0.222870 | 0.222870 |
| Item-KNN | 0.349709 | 0.420309 | 0.432038 |
| Combined | 0.303724 | 0.396261 | 0.433444 |
