"""Ranking quality metrics for top-K recommendation lists.

Accuracy metrics take the ranked list of recommended item ids for one user
and that user's relevant items. Relevance may be given either as a collection
of item ids (binary relevance) or as a mapping ``item_id -> gain`` with
non-negative graded gains. :func:`inverse_popularity_at_k` is the novelty
ranking metric: it takes an item-popularity mapping instead of a relevance
set and scores rank-discounted inverse popularity.

Recommended lists must not contain duplicates; metrics raise
:class:`ValueError` on repeated items because double-counted positions
silently inflate every score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Union

RelevanceSpec = Union[Mapping[object, float], Sequence[object]]


def _validate_recommended(recommended: Sequence) -> list:
    items = list(recommended)
    if len(set(items)) != len(items):
        raise ValueError("recommended list contains duplicate items")
    return items


def _gain(item, relevant: RelevanceSpec) -> float:
    if isinstance(relevant, Mapping):
        gain = relevant.get(item, 0.0)
    else:
        gain = 1.0 if item in relevant else 0.0
    if gain < 0:
        raise ValueError("relevance gains must be non-negative")
    return float(gain)


def _positive_count(relevant: RelevanceSpec) -> int:
    if isinstance(relevant, Mapping):
        return sum(1 for value in relevant.values() if float(value) > 0)
    return len(set(relevant))


def _discount(rank: int) -> float:
    """Logarithmic discount for 1-based rank."""
    return 1.0 / math.log2(rank + 1)


def _check_k(k: int) -> None:
    if int(k) != k or k < 1:
        raise ValueError("k must be an integer >= 1")


def precision_at_k(recommended: Sequence, relevant: RelevanceSpec, k: int) -> float:
    """Fraction of the first k recommendations that are relevant.

    The denominator is always the cutoff k even when fewer than k items are
    recommended; this matches the usual fixed-length top-K convention.
    """
    _check_k(k)
    items = _validate_recommended(recommended)
    hits = sum(1 for item in items[:k] if _gain(item, relevant) > 0)
    return hits / k


def recall_at_k(recommended: Sequence, relevant: RelevanceSpec, k: int) -> float:
    """Fraction of the user's relevant items recovered within the first k."""
    _check_k(k)
    items = _validate_recommended(recommended)
    total = _positive_count(relevant)
    if total == 0:
        raise ValueError("relevant set is empty")
    hits = sum(1 for item in items[:k] if _gain(item, relevant) > 0)
    return hits / total


def hit_rate_at_k(recommended: Sequence, relevant: RelevanceSpec, k: int) -> float:
    """1.0 when at least one relevant item appears within the first k."""
    _check_k(k)
    items = _validate_recommended(recommended)
    return 1.0 if any(_gain(item, relevant) > 0 for item in items[:k]) else 0.0


def average_precision_at_k(
    recommended: Sequence, relevant: RelevanceSpec, k: int
) -> float:
    """Average Precision at k with the normalizer capped at k.

    AP@K = (1 / min(n_relevant, k)) * sum_{i<=k} P@i * rel(i). Capping the
    denominator keeps scores comparable between users with few and many
    relevant items, which matters for leave-one-out test sets.
    """
    _check_k(k)
    items = _validate_recommended(recommended)
    total = _positive_count(relevant)
    if total == 0:
        raise ValueError("relevant set is empty")
    seen = 0
    score = 0.0
    for rank, item in enumerate(items[:k], start=1):
        if _gain(item, relevant) > 0:
            seen += 1
            score += seen / rank
    return score / min(total, k)


def ndcg_at_k(recommended: Sequence, relevant: RelevanceSpec, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Gains are discounted by ``1 / log2(rank + 1)``. The ideal DCG is built
    from *every* relevant item sorted by descending gain and truncated at k,
    so the normalization stays correct when the recommendation list is
    shorter than k or omits relevant items entirely. Returns 0.0 when no
    relevant item carries positive gain.
    """
    _check_k(k)
    items = _validate_recommended(recommended)
    dcg = sum(
        _gain(item, relevant) * _discount(rank)
        for rank, item in enumerate(items[:k], start=1)
    )
    if isinstance(relevant, Mapping):
        ideal_gains = sorted((float(v) for v in relevant.values()), reverse=True)
    else:
        ideal_gains = [1.0] * len(set(relevant))
    idcg = sum(
        gain * _discount(rank)
        for rank, gain in enumerate(ideal_gains[:k], start=1)
    )
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mrr(recommended: Sequence, relevant: RelevanceSpec) -> float:
    """Reciprocal rank of the first relevant item; 0.0 when there is none."""
    items = _validate_recommended(recommended)
    for rank, item in enumerate(items, start=1):
        if _gain(item, relevant) > 0:
            return 1.0 / rank
    return 0.0


def _validate_popularity(item_popularity: Mapping) -> None:
    if not isinstance(item_popularity, Mapping):
        raise TypeError("item_popularity must map item ids to interaction counts")
    for count in item_popularity.values():
        value = float(count)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("popularity counts must be finite and non-negative")


def _inverse_popularity(item, item_popularity: Mapping) -> float:
    """Add-one inverse popularity ``1 / (count + 1)`` in ``(0, 1]``.

    Missing items and non-positive counts score 1.0, the value of an item
    nobody has interacted with. A count of ``n`` scores ``1 / (n + 1)``.
    """
    count = float(item_popularity.get(item, 0.0))
    if count <= 0.0:
        return 1.0
    return 1.0 / (count + 1.0)


def inverse_popularity_at_k(
    recommended: Sequence, item_popularity: Mapping, k: int
) -> float:
    """Rank-discounted inverse-popularity novelty at k.

    Each item contributes ``1 / (count + 1)``, discounted by the same
    logarithmic weight as NDCG, ``1 / log2(rank + 1)``. The score divides by
    the sum of discounts through ``k``, so a shorter list is penalized and a
    full list of unseen items scores 1.0. Placing a rarer item earlier raises
    the score relative to the same items in the opposite order.

    ``item_popularity`` maps item id to interaction count. This is separate
    from :func:`reco_eval_kit.beyond_accuracy.novelty`, which averages
    self-information and ignores rank.
    """
    _check_k(k)
    items = _validate_recommended(recommended)
    _validate_popularity(item_popularity)
    numerator = sum(
        _inverse_popularity(item, item_popularity) * _discount(rank)
        for rank, item in enumerate(items[:k], start=1)
    )
    denominator = sum(_discount(rank) for rank in range(1, k + 1))
    return numerator / denominator


MetricCallable = Callable[..., float]


def mean_metric(
    metric: MetricCallable,
    recommended_lists: Sequence[Sequence],
    relevant_lists: Sequence[RelevanceSpec],
    k: Optional[int] = None,
) -> float:
    """Average a per-user metric over users.

    ``k`` is forwarded for cutoff metrics; pass ``None`` for metrics such as
    MRR that take no cutoff. An empty evaluation set averages to 0.0.
    """
    if len(recommended_lists) != len(relevant_lists):
        raise ValueError("recommended_lists and relevant_lists must align per user")
    values = [
        metric(recs, rels) if k is None else metric(recs, rels, k)
        for recs, rels in zip(recommended_lists, relevant_lists)
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean_inverse_popularity_at_k(
    recommended_lists: Sequence[Sequence],
    item_popularity: Mapping,
    k: int,
) -> float:
    """Average :func:`inverse_popularity_at_k` over users.

    Popularity describes the catalog and is shared across users, so this
    helper does not go through :func:`mean_metric`. An empty evaluation set
    averages to 0.0.
    """
    if len(recommended_lists) == 0:
        return 0.0
    values = [
        inverse_popularity_at_k(items, item_popularity, k) for items in recommended_lists
    ]
    return sum(values) / len(values)