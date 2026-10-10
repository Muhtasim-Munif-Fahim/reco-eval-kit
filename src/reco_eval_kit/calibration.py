"""Calibrated recommendations and popularity-bias measures.

Calibration (Steck, RecSys 2018) asks whether a recommendation list reflects
the *proportions* of a user's interests. Someone who watched 70% dramas and
30% comedies is miscalibrated by a top-10 list of dramas only, even when
every item is relevant. With ``p(g|u)`` the genre distribution of the user's
history and ``q(g|u)`` that of the list,

    C_KL(p, q) = sum_g p(g|u) * log( p(g|u) / q~(g|u) ),
    q~ = (1 - alpha) * q + alpha * p

where the small ``alpha`` keeps the divergence finite when the list misses
a genre entirely. Each item spreads unit mass uniformly over its genres.
:func:`calibrated_rerank` greedily builds a top-K list maximizing

    (1 - lambda) * sum(score(i) for i in list) - lambda * C_KL(p, q~(list))

which trades accuracy for calibration as ``lambda`` grows.

Popularity-bias measures (Abdollahpouri et al., 2019) describe how much a
recommender concentrates exposure on already-popular items:

* :func:`gini_index` — Gini coefficient of how often each catalog item is
  recommended (0 = perfectly equal exposure, toward 1 = a few items get all).
* :func:`average_recommendation_popularity` (ARP) — mean training popularity
  of the recommended items.
* :func:`average_long_tail_share` (APLT) — mean share of each list drawn from
  the long tail, where the head is the most popular ``head_fraction`` of
  items (see :func:`long_tail_items`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "average_long_tail_share",
    "average_recommendation_popularity",
    "calibrated_rerank",
    "genre_distribution",
    "gini_index",
    "kl_miscalibration",
    "long_tail_items",
    "miscalibration",
]


def _genres_of(item, item_genres: Mapping) -> tuple:
    genres = item_genres.get(item)
    if genres is None:
        return ()
    if isinstance(genres, str):
        return (genres,)
    return tuple(genres)


def genre_distribution(items, item_genres: Mapping, weights=None) -> dict:
    """Genre distribution of ``items``: each item spreads its weight over its genres.

    Items without genres are skipped. Returns an empty dict when no item
    carries a genre. ``weights`` optionally gives one non-negative weight per
    item (e.g. recency); the default is uniform.
    """
    items = list(items)
    if weights is None:
        weights = [1.0] * len(items)
    else:
        weights = [float(w) for w in weights]
        if len(weights) != len(items):
            raise ValueError("weights must align with items")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
    dist: dict = {}
    total = 0.0
    for item, weight in zip(items, weights):
        genres = _genres_of(item, item_genres)
        if not genres or weight == 0.0:
            continue
        share = weight / len(genres)
        for genre in genres:
            dist[genre] = dist.get(genre, 0.0) + share
        total += weight
    if total == 0.0:
        return {}
    return {genre: value / total for genre, value in dist.items()}


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return alpha


def kl_miscalibration(p: Mapping, q: Mapping, alpha: float = 0.01) -> float:
    """Steck's ``C_KL(p, q~)`` with ``q~ = (1 - alpha) q + alpha p``.

    ``p`` is the target (history) distribution and ``q`` the list
    distribution, both as ``{genre: probability}`` mappings. An empty ``p``
    has nothing to calibrate against and scores ``0.0``.
    """
    alpha = _check_alpha(alpha)
    value = 0.0
    for genre, pg in p.items():
        if pg <= 0.0:
            continue
        qg = (1.0 - alpha) * float(q.get(genre, 0.0)) + alpha * float(pg)
        value += pg * math.log(pg / qg)
    return max(value, 0.0)


def miscalibration(
    recommended_lists,
    user_histories,
    item_genres: Mapping,
    alpha: float = 0.01,
    k=None,
) -> float:
    """Mean ``C_KL`` between each user's history and their top-``k`` list."""
    if len(recommended_lists) != len(user_histories):
        raise ValueError("recommended_lists and user_histories must align per user")
    if k is not None and (isinstance(k, bool) or int(k) != k or k < 1):
        raise ValueError("k must be a positive integer")
    scores = []
    for items, history in zip(recommended_lists, user_histories):
        ranked = list(items) if k is None else list(items)[: int(k)]
        p = genre_distribution(history, item_genres)
        q = genre_distribution(ranked, item_genres)
        scores.append(kl_miscalibration(p, q, alpha))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def calibrated_rerank(
    candidates: Sequence,
    scores: Sequence[float],
    history,
    item_genres: Mapping,
    k: int = 10,
    lam: float = 0.5,
    alpha: float = 0.01,
) -> list:
    """Greedy calibrated top-``k`` re-ranking (Steck 2018).

    ``candidates`` are item ids with relevance ``scores`` (higher is better),
    typically a recommender's top-N. At every step the item that maximizes
    ``(1 - lam) * total_score - lam * C_KL(p, q~)`` of the extended list is
    appended. ``lam = 0`` reproduces the score order; larger ``lam`` buys
    calibration with accuracy. Ties break toward the earlier candidate.
    """
    candidates = list(candidates)
    scores = [float(s) for s in scores]
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    alpha = _check_alpha(alpha)
    p = genre_distribution(history, item_genres)

    selected: list = []
    selected_scores = 0.0
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        best_idx = None
        best_value = -math.inf
        for idx in remaining:
            trial = selected + [candidates[idx]]
            total = selected_scores + scores[idx]
            q = genre_distribution(trial, item_genres)
            value = (1.0 - lam) * total - lam * kl_miscalibration(p, q, alpha)
            if value > best_value + 1e-12:
                best_value = value
                best_idx = idx
        selected.append(candidates[best_idx])
        selected_scores += scores[best_idx]
        remaining.remove(best_idx)
    return selected


# ------------------------------------------------------------ popularity bias
def gini_index(recommended_lists, catalog=None) -> float:
    """Gini coefficient of item exposure across all recommendation lists.

    Items in ``catalog`` that are never recommended count with zero
    exposure; without ``catalog`` only recommended items are considered.
    """
    counts: dict = {}
    for items in recommended_lists:
        for item in items:
            counts[item] = counts.get(item, 0) + 1
    if catalog is not None:
        for item in catalog:
            counts.setdefault(item, 0)
    if not counts:
        raise ValueError("gini_index needs at least one item")
    values = np.sort(np.asarray(list(counts.values()), dtype=float))
    n = values.size
    total = values.sum()
    if total == 0.0 or n == 1:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float(np.sum((2 * ranks - n - 1) * values) / (n * total))


def average_recommendation_popularity(recommended_lists, item_popularity: Mapping) -> float:
    """ARP: mean over users of the mean training popularity of their list."""
    per_user = []
    for items in recommended_lists:
        items = list(items)
        if not items:
            continue
        per_user.append(sum(float(item_popularity.get(i, 0)) for i in items) / len(items))
    if not per_user:
        raise ValueError("average_recommendation_popularity needs a non-empty list")
    return float(sum(per_user) / len(per_user))


def long_tail_items(item_popularity: Mapping, head_fraction: float = 0.2) -> set:
    """Items outside the head: the most popular ``head_fraction`` of the catalog.

    Popularity ties at the head boundary break by item id order, so the head
    always holds ``ceil(head_fraction * n_items)`` items.
    """
    head_fraction = float(head_fraction)
    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must be in (0, 1)")
    if not item_popularity:
        raise ValueError("item_popularity must not be empty")
    ranked = sorted(item_popularity, key=lambda item: (-float(item_popularity[item]), str(item)))
    n_head = max(1, math.ceil(head_fraction * len(ranked)))
    return set(ranked[n_head:])


def average_long_tail_share(
    recommended_lists, item_popularity: Mapping, head_fraction: float = 0.2
) -> float:
    """APLT: mean share of each non-empty list that comes from the long tail.

    Items missing from ``item_popularity`` (never interacted with) count as
    long tail.
    """
    tail = long_tail_items(item_popularity, head_fraction)
    head = set(item_popularity) - tail
    shares = []
    for items in recommended_lists:
        items = list(items)
        if not items:
            continue
        shares.append(sum(1 for i in items if i not in head) / len(items))
    if not shares:
        raise ValueError("average_long_tail_share needs a non-empty list")
    return float(sum(shares) / len(shares))
