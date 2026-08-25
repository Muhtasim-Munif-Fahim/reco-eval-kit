# reco-eval-kit

Evaluation toolkit for implicit-feedback recommender systems: ranking
metrics, beyond-accuracy measures, standard leave-one-out / leave-last-N
protocols, seeded synthetic interaction data, and baseline recommenders to
score against.

Planned modules:

- Ranking metrics: Precision@K, Recall@K, MAP@K, NDCG@K, MRR, HitRate@K
- Beyond accuracy: catalog coverage, novelty (inverse popularity),
  intra-list diversity
- Protocols: leave-one-out and leave-last-N splitting, relevance thresholding
- Baselines: popularity, seeded random, item-item co-occurrence counts
- Markdown report rendering plus a small end-to-end CLI

This is an early scaffold; usage documentation lands with each module.