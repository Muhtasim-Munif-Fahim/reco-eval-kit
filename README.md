# reco-eval-kit

Ranking and beyond-accuracy evaluation toolkit for implicit-feedback
recommender systems. It ships the standard top-K metrics, catalog-level and
list-level quality measures, leave-one-out / leave-last-N protocols, a seeded
synthetic interaction generator, six baseline recommenders to score against,
and a small CLI that ties the whole loop together and writes a markdown report.

## Features

- **Ranking metrics** — Precision@K, Recall@K, MAP@K, NDCG@K, MRR, HitRate@K,
  and InversePopularity@K. Binary or graded relevance per user; NDCG
  normalizes against an ideal DCG built from *all* relevant items (so short
  lists and missed hits are handled correctly); MAP@K caps its normalizer at
  K; duplicate recommendations raise. InversePopularity@K is rank-aware
  novelty: each top-K item contributes `1 / (count + 1)`, discounted by
  `1 / log2(rank + 1)` and normalized by the sum of those discounts through K.
- **Beyond accuracy** — catalog coverage, novelty as mean self-information
  `-log2(p(item))` with a probability floor for unseen items, intra-list
  diversity as mean pairwise cosine distance over item feature vectors,
  unexpectedness as `1 - p(item)` or cosine distance from a user's primitive
  profile, and serendipity as the mean of relevance × unexpectedness in the
  top-K.
- **Protocols** — chronological leave-one-out and leave-last-N splitting with
  minimum-history filters, plus rating thresholding for explicit feedback.
- **Baselines** — popularity ranking (ties broken by item id), seeded random,
  item-item co-occurrence counts with popularity fallback, ItemKNN
  (cosine or Jaccard item-item similarity, top-`n_neighbors` truncation,
  popularity fallback), UserKNN (cosine or Jaccard user-user similarity on
  the binary interaction matrix, top-`n_neighbors` truncation, popularity
  fallback), and a lightweight BPR matrix-factorization model (seeded SGD
  on implicit pairwise triples, popularity fallback for unknown users).
- **Reporting** — markdown tables of all metrics per model, written to disk.

Catalog coverage and intra-list diversity are already part of the toolkit
(`catalog_coverage`, `intra_list_diversity`), with tests in
`tests/test_beyond_accuracy.py`. This change therefore does not add a second
copy of those measures. It adds InversePopularity@K instead: a ranking metric
for novelty as inverse popularity. It is distinct from `novelty`, which
averages self-information and ignores position. A rarer item placed earlier
scores higher than the same item placed later, and a list shorter than K is
penalized because the discount normalizer always runs through K.

## Installation

```bash
pip install -r requirements.txt
```

Optional editable install exposing the `reco-eval-kit` command:

```bash
pip install -e .
```

## Quickstart

### Score one model by hand

```python
import pandas as pd
from reco_eval_kit.metrics import (
    precision_at_k, ndcg_at_k, mrr, inverse_popularity_at_k,
)
from reco_eval_kit.baselines import (
    BPRRecommender, ItemKNNRecommender, PopularityRecommender,
    UserKNNRecommender,
)
from reco_eval_kit.splitting import leave_one_out

interactions = pd.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 2],
    "item_id": [10, 11, 12, 10, 13, 14],
    "timestamp": [1, 2, 3, 4, 5, 6],
})
train, test = leave_one_out(interactions)

model = PopularityRecommender().fit(train)
recs = model.recommend(user_id=1, k=2, exclude={10, 11})
print(recs)                                   # e.g. [12, 13]
print(precision_at_k(recs, relevant={12}, k=2))  # 0.5
print(ndcg_at_k(recs, relevant={12}, k=2))
print(mrr(recs, relevant={13}))
popularity = train["item_id"].value_counts().to_dict()
print(inverse_popularity_at_k(recs, popularity, k=2))

bpr = BPRRecommender(seed=0).fit(train)
print(bpr.recommend(user_id=1, k=2, exclude={10, 11}))

knn = ItemKNNRecommender(similarity="cosine", n_neighbors=20).fit(train)
print(knn.recommend(user_id=1, k=2, exclude={10, 11}))
# ItemKNNRecommender(similarity="jaccard") uses set-overlap weights instead.

user_knn = UserKNNRecommender(similarity="cosine", n_neighbors=20).fit(train)
print(user_knn.recommend(user_id=1, k=2, exclude={10, 11}))
# UserKNNRecommender(similarity="jaccard") weights neighbors by set overlap.
```

### Generate synthetic data with learnable structure

```python
from reco_eval_kit.synthetic import generate_interactions, generate_item_features

interactions = generate_interactions(
    n_users=100, n_items=300, n_interactions=6000, seed=42
)
features = generate_item_features(n_items=300, seed=43)
```

Users belong to latent taste clusters; item draws mix cluster-favored slices
with a Zipf-style global popularity distribution, so popularity,
co-occurrence, ItemKNN, UserKNN, and BPR baselines all find real signal.

### Beyond-accuracy measures

```python
from reco_eval_kit.beyond_accuracy import (
    catalog_coverage, novelty, intra_list_diversity,
    unexpectedness, profile_unexpectedness, serendipity,
)

lists = [[12, 13], [12, 14]]
relevant = [{12}, {14}]
histories = [{10, 11}, {10}]
popularity = train["item_id"].value_counts().to_dict()

catalog_coverage(lists, catalog=300)
novelty(lists, popularity)
intra_list_diversity(lists, features)
unexpectedness(lists, popularity)
profile_unexpectedness(lists, histories, features)
serendipity(lists, relevant, item_popularity=popularity)
```

### End-to-end CLI

```bash
python -m reco_eval_kit --n-users 60 --n-items 150 \
    --n-interactions 2500 --seed 7 --k 10 --output report.md
```

or after `pip install -e .`:

```bash
reco-eval-kit --seed 7 --k 10
```

The CLI generates data, splits leave-one-out, fits all seven baselines,
averages every metric across test users, computes beyond-accuracy measures,
and writes a markdown report that includes each listed model.

### Bundled demo

```bash
python examples/run_demo.py
```

writes `examples/output/demo_report.md`.

## Metric conventions

- Precision@K divides by the cutoff `k`, even when fewer items are recommended.
- Recall@K normalizes by the number of positive-relevance items.
- AP@K divides the running-precision sum by `min(n_relevant, k)`.
- NDCG@K discounts gains by `1/log2(rank + 1)`; the ideal side sorts every
  relevant item's gain descending before truncating at `k`.
- MRR is the reciprocal rank of the first relevant hit; `0.0` without one.
- InversePopularity@K uses `1 / (count + 1)` per item (unseen or zero-count
  items score `1.0`), weights positions by `1 / log2(rank + 1)`, and divides
  by the sum of those weights through `k`.
- Unexpectedness is the mean of `1 - p(item)` over recommended entries.
- Profile unexpectedness is mean cosine distance from each user's primitive
  profile (the mean history vector), clipped to `[0, 1]`.
- Serendipity averages `relevant(i) * unexpected(i)` per user; with a cutoff
  `k` the denominator is `k`.
- ItemKNN binarizes interactions, then weights each candidate by the sum of
  cosine (`|U_i ∩ U_j| / sqrt(|U_i| |U_j|)`) or Jaccard
  (`|U_i ∩ U_j| / |U_i ∪ U_j|`) similarities to the user's history. Each item
  keeps its `n_neighbors` strongest neighbors (default 20). Score ties and
  neighbor ties break by ascending item id. Seen items are dropped, and any
  shortfall is filled from the popularity ranking. The CLI reports this model
  as `item-knn(cosine)`.
- UserKNN binarizes interactions, then scores each unseen item by the sum of
  cosine (`|I_u ∩ I_v| / sqrt(|I_u| |I_v|)`) or Jaccard
  (`|I_u ∩ I_v| / |I_u ∪ I_v|`) similarities to neighbors who interacted with
  it. Each user keeps their `n_neighbors` strongest neighbors (default 20).
  Neighbor ties break by ascending user id; score ties break by ascending
  item id. Seen items are dropped, and any shortfall is filled from the
  popularity ranking. The CLI reports this model as `user-knn(cosine)`.

## Project layout

```
src/reco_eval_kit/
    metrics.py           # Precision/Recall/MAP/NDCG/MRR/HitRate/InversePopularity @K
    beyond_accuracy.py   # coverage, novelty, diversity, unexpectedness, serendipity
    splitting.py         # leave-one-out, leave-last-N, thresholding
    synthetic.py         # seeded interactions and item features
    baselines.py         # popularity, random, co-occurrence, ItemKNN, UserKNN, BPR-MF, PureSVD
    report.py            # markdown rendering
    cli.py               # end-to-end entry point
tests/                   # pytest suite
examples/run_demo.py     # runnable demo
```

## Development

```bash
python -m pytest tests -q
python -m compileall -q src
```

## License

MIT — see [LICENSE](LICENSE).