"""Sampled-negative evaluation and its bias correction (Krichene & Rendle, KDD 2020).

Many papers rank each held-out positive against ``m`` randomly sampled
negatives (e.g. 100) instead of the whole catalog, and report HitRate@K or
NDCG@K on that small list. Krichene & Rendle ("On Sampled Metrics for Item
Recommendation", KDD 2020) showed that these *sampled* metrics are biased
estimates of the full-catalog metrics and can even reverse the order of two
models.

Model: one positive and ``n`` candidate negatives (``n = N - 1`` for a catalog
of ``N`` items after removing the positive). Let ``R`` be the positive's
0-based **full** rank (how many of the ``n`` negatives score above it) and
``r`` its rank among ``m`` negatives drawn without replacement. Then
``r | R ~ Hypergeometric(n, R, m)``:

    P(r | R) = C(R, r) C(n - R, m - r) / C(n, m).

A full-catalog metric is a function ``M(R)`` (HitRate@K: ``1[R < K]``;
NDCG@K: ``1[R < K] / log2(R + 2)``). The naive sampled metric applies the
same function to ``r``. This module provides

* :func:`sampled_rank` / :func:`sample_negative_ranks` for computing ``r``,
* :func:`rank_transition_matrix` for ``P(r | R)``,
* :func:`metric_by_rank` for ``M`` as a vector over ranks,
* :func:`correction_weights`, which find a per-sampled-rank value ``f(r)``
  so that ``E[f(r) | R] ≈ M(R)``: ``method="ls"`` solves the weighted least
  squares problem and ``method="bv"`` adds Krichene & Rendle's variance
  penalty ``gamma * E[f(r)^2]``, trading a little bias for much lower
  variance, under a prior ``p(R)`` (uniform by default). Plain least squares
  is badly conditioned when ``m << n`` and its estimates can be wildly
  off; prefer ``"bv"`` in practice;
* :func:`corrected_sampled_metric`, which averages ``f(r)`` over users and
  also reports the naive sampled value, and
* :func:`expected_sampled_metric`, the expected naive sampled metric for
  known full ranks (useful to show the bias).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy.stats import hypergeom

__all__ = [
    "SampledMetricEstimate",
    "full_rank",
    "sampled_rank",
    "sample_negative_ranks",
    "rank_transition_matrix",
    "metric_by_rank",
    "correction_weights",
    "corrected_sampled_metric",
    "expected_sampled_metric",
]

_METRICS = ("hit", "ndcg", "rr")


def _check_counts(n_negatives: int, n_sampled: int) -> None:
    if isinstance(n_negatives, bool) or int(n_negatives) != n_negatives or n_negatives < 1:
        raise ValueError("n_negatives must be a positive integer")
    if isinstance(n_sampled, bool) or int(n_sampled) != n_sampled or not 1 <= n_sampled <= n_negatives:
        raise ValueError("n_sampled must be an integer in [1, n_negatives]")


def full_rank(ranking: Sequence, positive, candidates: Iterable) -> int:
    """0-based rank of ``positive`` among ``candidates`` under ``ranking``.

    ``ranking`` lists items best first. Candidates that do not appear in it are
    ranked after every listed item; if the positive itself is missing it gets
    the worst rank (all other candidates above it).
    """

    position = {item: i for i, item in enumerate(ranking)}
    negatives = [c for c in candidates if c != positive]
    if positive not in position:
        return len(negatives)
    pos = position[positive]
    return sum(1 for c in negatives if position.get(c, np.inf) < pos)


def sampled_rank(positive_score: float, negative_scores: Sequence[float], ties: str = "pessimistic") -> float:
    """Rank of a positive among sampled negatives from raw scores.

    ``ties`` decides how negatives with an equal score count: ``"pessimistic"``
    (above the positive), ``"optimistic"`` (below) or ``"average"`` (half).
    """

    neg = np.asarray(negative_scores, dtype=float)
    above = int(np.sum(neg > positive_score))
    tied = int(np.sum(neg == positive_score))
    if ties == "pessimistic":
        return float(above + tied)
    if ties == "optimistic":
        return float(above)
    if ties == "average":
        return above + tied / 2.0
    raise ValueError("ties must be 'pessimistic', 'optimistic' or 'average'")


def sample_negative_ranks(full_ranks: Sequence[int], n_negatives: int, n_sampled: int,
                          seed: Optional[int] = None) -> np.ndarray:
    """Draw ``r ~ Hypergeometric(n, R, m)`` for each full rank ``R`` (simulation helper).

    This is exactly the distribution of the sampled rank when the ``m``
    negatives are drawn uniformly without replacement from the ``n``
    candidates.
    """

    _check_counts(n_negatives, n_sampled)
    ranks = np.asarray(full_ranks, dtype=int)
    if np.any(ranks < 0) or np.any(ranks > n_negatives):
        raise ValueError("full ranks must lie in [0, n_negatives]")
    rng = np.random.default_rng(seed)
    return rng.hypergeometric(ranks, n_negatives - ranks, n_sampled)


def rank_transition_matrix(n_negatives: int, n_sampled: int) -> np.ndarray:
    """``P[R, r] = P(sampled rank r | full rank R)``; shape ``(n + 1, m + 1)``."""

    _check_counts(n_negatives, n_sampled)
    n, m = int(n_negatives), int(n_sampled)
    R = np.arange(n + 1)[:, None]
    r = np.arange(m + 1)[None, :]
    P = hypergeom.pmf(r, n, R, m)
    P = np.nan_to_num(P, nan=0.0)
    return P / P.sum(axis=1, keepdims=True)


def metric_by_rank(metric: str, k: int, n_ranks: int) -> np.ndarray:
    """``M(R)`` for ``R = 0 .. n_ranks - 1``: ``hit``, ``ndcg`` or ``rr`` (reciprocal rank, uncut if k is None)."""

    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    R = np.arange(n_ranks, dtype=float)
    if k is None:
        if metric != "rr":
            raise ValueError("k is required for hit and ndcg")
        return 1.0 / (R + 1.0)
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    inside = R < k
    if metric == "hit":
        return inside.astype(float)
    if metric == "ndcg":
        return np.where(inside, 1.0 / np.log2(R + 2.0), 0.0)
    return np.where(inside, 1.0 / (R + 1.0), 0.0)


def correction_weights(
    n_negatives: int,
    n_sampled: int,
    metric: str = "hit",
    k: Optional[int] = 10,
    *,
    method: str = "bv",
    gamma: float = 0.01,
    prior: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Per-sampled-rank values ``f(0..m)`` approximating the full metric.

    Minimises ``sum_R p(R) (E[f(r)|R] - M(R))^2 + gamma sum_R p(R) E[f(r)^2|R]``
    (``gamma = 0`` for ``method="ls"``). ``method="naive"`` returns ``M``
    evaluated at the sampled rank, i.e. the uncorrected sampled metric.
    With ``m = n`` (no sampling) and ``gamma = 0`` the solution is ``M`` itself.
    """

    _check_counts(n_negatives, n_sampled)
    n, m = int(n_negatives), int(n_sampled)
    if method == "naive":
        return metric_by_rank(metric, k, m + 1)
    if method not in {"ls", "bv"}:
        raise ValueError("method must be 'naive', 'ls' or 'bv'")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    lam = 0.0 if method == "ls" else float(gamma)
    P = rank_transition_matrix(n, m)
    target = metric_by_rank(metric, k, n + 1)
    if prior is None:
        p = np.full(n + 1, 1.0 / (n + 1))
    else:
        p = np.asarray(prior, dtype=float).ravel()
        if p.shape[0] != n + 1 or np.any(p < 0) or p.sum() <= 0:
            raise ValueError("prior must be a non-negative vector over ranks 0..n_negatives")
        p = p / p.sum()
    PtW = P.T * p[None, :]
    A = PtW @ P + lam * np.diag(PtW.sum(axis=1))
    b = PtW @ target
    # tiny ridge keeps the system solvable for sampled ranks the prior never reaches
    A += 1e-12 * max(np.trace(A), 1e-300) / (m + 1) * np.eye(m + 1)
    return np.linalg.solve(A, b)


@dataclass(frozen=True)
class SampledMetricEstimate:
    """Naive and corrected sampled estimates of a full-catalog metric."""

    metric: str
    k: Optional[int]
    naive: float
    corrected: float
    method: str
    n_users: int
    n_negatives: int  # -1 when users had different candidate counts
    n_sampled: int
    weights: np.ndarray

    def as_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric, "k": self.k, "naive": self.naive,
            "corrected": self.corrected, "method": self.method, "n_users": self.n_users,
            "n_negatives": self.n_negatives, "n_sampled": self.n_sampled,
        }


def corrected_sampled_metric(
    sampled_ranks: Sequence[float],
    n_negatives,
    n_sampled: int,
    metric: str = "hit",
    k: Optional[int] = 10,
    *,
    method: str = "bv",
    gamma: float = 0.01,
    prior: Optional[Sequence[float]] = None,
) -> SampledMetricEstimate:
    """Average naive and corrected values over users' sampled ranks.

    ``n_negatives`` is the candidate-negative count, either shared by all users
    or one value per user (e.g. catalog size minus each user's history); the
    correction weights are solved once per distinct value. Fractional ranks
    from ``ties="average"`` are interpolated linearly between neighbouring
    integer ranks. ``prior`` (only with a shared ``n_negatives``) is the
    assumed distribution of full ranks.
    """

    ranks = np.asarray(sampled_ranks, dtype=float).ravel()
    if ranks.size == 0:
        raise ValueError("sampled_ranks must not be empty")
    if np.any(ranks < 0) or np.any(ranks > n_sampled):
        raise ValueError("sampled ranks must lie in [0, n_sampled]")
    per_user = np.broadcast_to(np.asarray(n_negatives), ranks.shape).astype(int)
    if prior is not None and np.unique(per_user).size > 1:
        raise ValueError("prior requires a single shared n_negatives")
    grid = np.arange(int(n_sampled) + 1, dtype=float)
    naive_w = correction_weights(int(per_user.max()), n_sampled, metric, k, method="naive")
    corrected = np.empty_like(ranks)
    weights = None
    for n in np.unique(per_user):
        sel = per_user == n
        weights = correction_weights(int(n), n_sampled, metric, k, method=method,
                                     gamma=gamma, prior=prior)
        corrected[sel] = np.interp(ranks[sel], grid, weights)
    shared = int(per_user[0]) if np.unique(per_user).size == 1 else -1
    return SampledMetricEstimate(
        metric=metric, k=k,
        naive=float(np.mean(np.interp(ranks, grid, naive_w))),
        corrected=float(np.mean(corrected)),
        method=method, n_users=int(ranks.size), n_negatives=shared,
        n_sampled=int(n_sampled), weights=weights if shared >= 0 else np.array([]),
    )


def expected_sampled_metric(full_ranks: Sequence[int], n_negatives: int, n_sampled: int,
                            metric: str = "hit", k: Optional[int] = 10) -> Dict[str, float]:
    """Full-catalog metric vs. the *expected* naive sampled metric for known full ranks."""

    ranks = np.asarray(full_ranks, dtype=int).ravel()
    if ranks.size == 0:
        raise ValueError("full_ranks must not be empty")
    if np.any(ranks < 0) or np.any(ranks > n_negatives):
        raise ValueError("full ranks must lie in [0, n_negatives]")
    P = rank_transition_matrix(n_negatives, n_sampled)
    naive = metric_by_rank(metric, k, n_sampled + 1)
    full = metric_by_rank(metric, k, n_negatives + 1)
    return {
        "full": float(full[ranks].mean()),
        "expected_sampled": float((P[ranks] @ naive).mean()),
    }
