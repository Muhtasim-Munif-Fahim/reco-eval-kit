"""Propensity-weighted (unbiased) offline ranking metrics for MNAR feedback.

Implicit feedback is *missing not at random*: users can only interact with
items they were exposed to, and popular items are exposed far more often.
Averaging Recall@K or NDCG@K over the observed held-out positives therefore
over-rewards recommenders that push popular items, because popular relevant
items are over-represented in the test set.

Yang et al. ("Unbiased offline recommender evaluation for missing-not-at-
random implicit feedback", RecSys 2018) correct this with inverse propensity
scoring. Write ``c(rank_i)`` for a per-item gain of relevant item ``i`` (hit
indicator for recall, ``1/log2(rank + 1)`` for DCG, ``1/rank`` for reciprocal
rank, all zero beyond the cutoff ``k``) and ``p_i`` for the probability that
a relevant item is observed. For a user with observed positives ``S_u``:

* naive: ``(1/|S_u|) * sum_{i in S_u} c_i``
* IPS:   ``(1/|S_u|) * sum_{i in S_u} c_i / p_i`` (unbiased up to the scale of
  the true positive count, can exceed 1)
* SNIPS: the IPS estimate divided by the mean inverse-propensity mass,
  ``sum_u IPS_u / sum_u ((1/|S_u|) sum_{i in S_u} 1 / p_i)``, normalized
  *across* users (in ``[0, 1]``, lower variance; the usual reporting
  choice). Normalizing globally matters: under leave-one-out there is one
  positive per user, so a per-user self-normalization would cancel the
  weights entirely and collapse back to the naive estimate.

Propensities are typically modelled from item popularity with the power law
``p_i ∝ n_i ** ((gamma + 1) / 2)`` (Yang et al., Eq. 9), which
:func:`power_law_propensities` implements with clipping to bound variance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from .metrics import _check_k, _validate_recommended

GAINS = ("hit", "dcg", "rr")


def power_law_propensities(
    item_popularity: Mapping,
    gamma: float = 0.5,
    max_propensity: float = 1.0,
    min_propensity: float = 1e-3,
) -> dict:
    """Observation propensities ``p_i ∝ n_i ** ((gamma + 1) / 2)``.

    The most popular item gets ``max_propensity``; every propensity is then
    clipped below at ``min_propensity`` so a handful of very rare items cannot
    dominate the estimate with huge weights. Items with zero count receive
    ``min_propensity``.
    """
    if not 0.0 < max_propensity <= 1.0:
        raise ValueError("max_propensity must lie in (0, 1]")
    if not 0.0 < min_propensity <= max_propensity:
        raise ValueError("min_propensity must lie in (0, max_propensity]")
    if gamma < -1.0:
        raise ValueError("gamma must be >= -1")
    if not item_popularity:
        return {}
    counts = {item: float(count) for item, count in item_popularity.items()}
    if any(c < 0 or not math.isfinite(c) for c in counts.values()):
        raise ValueError("item popularity counts must be finite and non-negative")
    exponent = (gamma + 1.0) / 2.0
    top = max(counts.values())
    if top <= 0:
        return {item: float(min_propensity) for item in counts}
    out = {}
    for item, count in counts.items():
        p = max_propensity * (count / top) ** exponent if count > 0 else 0.0
        out[item] = float(min(max_propensity, max(min_propensity, p)))
    return out


def _item_gain(rank: Optional[int], k: int, gain: str) -> float:
    if rank is None or rank > k:
        return 0.0
    if gain == "hit":
        return 1.0
    if gain == "dcg":
        return 1.0 / math.log2(rank + 1)
    return 1.0 / rank


@dataclass(frozen=True)
class UserEstimate:
    """Per-user naive / IPS / self-normalized values and the weight mass.

    ``snips`` here is the *per-user* self-normalized value; the aggregate in
    :func:`mean_unbiased_metric_at_k` normalizes across users instead.
    """

    naive: float
    ips: float
    snips: float
    n_observed: int
    weight_sum: float


def unbiased_user_metric(
    recommended: Sequence,
    observed_relevant: Sequence,
    propensities: Mapping,
    k: int,
    gain: str = "hit",
    default_propensity: Optional[float] = None,
) -> Optional[UserEstimate]:
    """Naive, IPS and SNIPS estimates of a per-item ranking gain for one user.

    Returns ``None`` when the user has no observed positives. ``gain`` is
    ``"hit"`` (recall-style), ``"dcg"`` or ``"rr"``. Items missing from
    ``propensities`` use ``default_propensity`` or raise ``KeyError``.
    """
    _check_k(k)
    if gain not in GAINS:
        raise ValueError(f"gain must be one of {GAINS}")
    items = _validate_recommended(recommended)
    ranks = {item: rank for rank, item in enumerate(items, start=1)}
    positives = list(dict.fromkeys(observed_relevant))
    if not positives:
        return None
    naive = ips = weight_sum = 0.0
    for item in positives:
        if item in propensities:
            p = float(propensities[item])
        elif default_propensity is not None:
            p = float(default_propensity)
        else:
            raise KeyError(f"no propensity for item {item!r}")
        if not 0.0 < p <= 1.0:
            raise ValueError("propensities must lie in (0, 1]")
        c = _item_gain(ranks.get(item), k, gain)
        naive += c
        ips += c / p
        weight_sum += 1.0 / p
    n = len(positives)
    return UserEstimate(
        naive=naive / n,
        ips=ips / n,
        snips=ips / weight_sum,
        n_observed=n,
        weight_sum=weight_sum,
    )


@dataclass(frozen=True)
class UnbiasedEstimate:
    """Mean naive / IPS / SNIPS over users plus diagnostics."""

    gain: str
    k: int
    naive: float
    ips: float
    snips: float
    n_users: int
    n_observed: int
    effective_sample_size: float


def mean_unbiased_metric_at_k(
    recommended_lists: Sequence[Sequence],
    observed_relevant_lists: Sequence[Sequence],
    propensities: Mapping,
    k: int,
    gain: str = "hit",
    default_propensity: Optional[float] = None,
) -> UnbiasedEstimate:
    """Average :func:`unbiased_user_metric` over users with observed positives.

    ``effective_sample_size`` is Kish's ``(sum w)^2 / sum w^2`` over all
    observed positives' inverse-propensity weights; a value far below
    ``n_observed`` warns that a few rare items dominate the estimate.
    """
    if len(recommended_lists) != len(observed_relevant_lists):
        raise ValueError("recommended_lists and observed_relevant_lists must align per user")
    per_user = []
    weights = []
    for recs, rels in zip(recommended_lists, observed_relevant_lists):
        est = unbiased_user_metric(recs, rels, propensities, k, gain, default_propensity)
        if est is None:
            continue
        per_user.append(est)
        for item in dict.fromkeys(rels):
            p = propensities.get(item, default_propensity)
            weights.append(1.0 / float(p))
    if not per_user:
        return UnbiasedEstimate(gain, int(k), 0.0, 0.0, 0.0, 0, 0, 0.0)
    m = len(per_user)
    sw = sum(weights)
    sw2 = sum(w * w for w in weights)
    return UnbiasedEstimate(
        gain=gain,
        k=int(k),
        naive=sum(e.naive for e in per_user) / m,
        ips=sum(e.ips for e in per_user) / m,
        snips=sum(e.ips for e in per_user)
        / sum(e.weight_sum / e.n_observed for e in per_user),
        n_users=m,
        n_observed=sum(e.n_observed for e in per_user),
        effective_sample_size=(sw * sw / sw2) if sw2 > 0 else 0.0,
    )


__all__ = [
    "GAINS",
    "UnbiasedEstimate",
    "UserEstimate",
    "mean_unbiased_metric_at_k",
    "power_law_propensities",
    "unbiased_user_metric",
]
