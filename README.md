# reco-eval-kit

Ranking and beyond-accuracy evaluation toolkit for implicit-feedback
recommender systems. It ships the standard top-K metrics, catalog-level and
list-level quality measures, leave-one-out / leave-last-N protocols, a seeded
synthetic interaction generator, three baseline recommenders to score against,
and a small CLI that ties the whole loop together and writes a markdown report.

## Features

- **Ranking metrics** — Precision@K, Recall@K, MAP@K, NDCG@K, MRR, HitRate@K.
  Binary or graded relevance per user; NDCG normalizes against an ideal DCG
  built from *all* relevant items (so short lists and missed hits are handled
  correctly); MAP@K caps its normalizer at K; duplicate recommendations raise.
- **Beyond accuracy** — catalog coverage, novelty as mean self-information
  `-log2(p(item))` with a probability floor for unseen items, and intra-list
  diversity as mean pairwise cosine distance over item feature vectors.
- **Protocols** — chronological leave-one-out and leave-last-N splitting with
  minimum-history filters, plus rating thresholding for explicit feedback.
- **Baselines** — popularity ranking (ties broken by item id), seeded random,
  and item-item co-occurrence counts with popularity fallback.
- **Reporting** — markdown tables of all metrics per model, written to disk.

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
from reco_eval_kit.metrics import precision_at_k, ndcg_at_k, mrr
from reco_eval_kit.baselines import PopularityRecommender
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
with a Zipf-style global popularity distribution, so popularity and
co-occurrence baselines both find real signal.

### Beyond-accuracy measures

```python
from reco_eval_kit.beyond_accuracy import (
    catalog_coverage, novelty, intra_list_diversity,
)

lists = [[12, 13], [12, 14]]
popularity = train["item_id"].value_counts().to_dict()

catalog_coverage(lists, catalog=300)
novelty(lists, popularity)
intra_list_diversity(lists, features)
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

The CLI generates data, splits leave-one-out, fits all three baselines,
averages every metric across test users, computes beyond-accuracy measures,
and writes a markdown report.

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

## Project layout

```
src/reco_eval_kit/
    metrics.py           # Precision/Recall/MAP/NDCG/MRR/HitRate @K
    beyond_accuracy.py   # coverage, novelty, intra-list diversity
    splitting.py         # leave-one-out, leave-last-N, thresholding
    synthetic.py         # seeded interactions and item features
    baselines.py         # popularity, random, item-item co-occurrence
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