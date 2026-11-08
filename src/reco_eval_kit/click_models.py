"""Click models and position-bias (propensity) estimation from click logs.

Propensity-weighted metrics (:mod:`reco_eval_kit.unbiased`) need examination
probabilities per rank. This module estimates them from logged impressions
``(query, item, position, click)`` instead of assuming a power law.

* :func:`fit_position_based_model` – the position-based click model
  ``P(click) = theta_k * gamma_{q,d}`` fitted by expectation-maximization
  (Chuklin, Markov & de Rijke 2015; regression-EM setting of Wang et al.,
  WSDM 2018 without the feature regressor). Examination ``theta`` is reported
  relative to rank 1 (the model is identifiable only up to that scale).
* :func:`randomization_propensities` – the RandTop-n intervention estimator
  (Joachims, Swaminathan & Schnabel, WSDM 2017): when the top-n results are
  shuffled uniformly, the click-through rate at each rank is proportional to
  its examination probability.
* :func:`fit_cascade_model` – maximum-likelihood attractiveness of the cascade
  model (Craswell et al. 2008): the user scans top-down and stops at the first
  click, so only results at or above the first click were examined.
* :func:`fit_power_law` – least-squares fit of ``theta_k = k^-eta`` on the log
  scale, to plug an estimated ``eta`` into the power-law helpers.
* :func:`click_log_likelihood` / :func:`click_perplexity` – held-out fit of a
  fitted PBM (per-rank perplexity as in Dupret & Piwowarski 2008).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ClickRecord",
    "PBMResult",
    "CascadeResult",
    "fit_position_based_model",
    "randomization_propensities",
    "fit_cascade_model",
    "fit_power_law",
    "click_log_likelihood",
    "click_perplexity",
]

ClickRecord = tuple  # (query, item, position (1-based), click (0/1))
_EPS = 1e-9


def _records(records: Iterable[Sequence]) -> list[tuple[Hashable, Hashable, int, int]]:
    out = []
    for rec in records:
        if len(rec) != 4:
            raise ValueError("each record must be (query, item, position, click)")
        q, d, k, c = rec
        k = int(k)
        if k < 1:
            raise ValueError("positions are 1-based and must be >= 1")
        if c not in (0, 1, True, False):
            raise ValueError("click must be 0 or 1")
        out.append((q, d, k, int(c)))
    if not out:
        raise ValueError("no click records")
    return out


@dataclass
class PBMResult:
    """Fitted position-based model."""

    examination: np.ndarray  # theta_k for k = 1..K, relative to rank 1
    attractiveness: dict = field(repr=False)  # (query, item) -> gamma
    log_likelihood: float
    n_iter: int
    converged: bool
    raw_examination: np.ndarray = field(repr=False, default=None)

    def propensity(self, position: int) -> float:
        """Relative examination probability at a 1-based rank."""
        if position < 1 or position > len(self.examination):
            raise ValueError("position outside the fitted range")
        return float(self.examination[position - 1])

    def click_probability(self, query, item, position: int) -> float:
        gamma = self.attractiveness.get((query, item), self._prior)
        return float(self.raw_examination[position - 1] * gamma)

    _prior: float = 0.5


def _pbm_loglik(recs, theta, gamma) -> float:
    ll = 0.0
    for q, d, k, c in recs:
        p = float(np.clip(theta[k - 1] * gamma[(q, d)], _EPS, 1 - _EPS))
        ll += np.log(p) if c else np.log1p(-p)
    return ll


def fit_position_based_model(
    records: Iterable[Sequence],
    n_positions: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-7,
    init_theta: float = 0.5,
    init_gamma: float = 0.5,
) -> PBMResult:
    """Fit ``P(C=1 | q, d, k) = theta_k * gamma_{q,d}`` by EM.

    E-step for an unclicked impression: ``P(E=1, R=0 | C=0) =
    theta (1 - gamma) / (1 - theta gamma)`` and ``P(E=0, R=1 | C=0) =
    (1 - theta) gamma / (1 - theta gamma)``; a click implies ``E = R = 1``.
    M-step: ``theta_k`` and ``gamma_{q,d}`` are the means of those posteriors
    over the impressions at rank ``k`` and of pair ``(q, d)``.

    Note the rank structure must vary across a pair's impressions for theta and
    gamma to be separable (e.g. via swaps, rank randomization or re-rankings).
    """
    recs = _records(records)
    K = max(k for _, _, k, _ in recs)
    if n_positions is not None:
        if n_positions < K:
            raise ValueError("n_positions is smaller than the largest logged position")
        K = int(n_positions)
    if not (0 < init_theta < 1 and 0 < init_gamma < 1):
        raise ValueError("initial values must lie in (0, 1)")
    theta = np.full(K, init_theta, dtype=float)
    pairs = {(q, d) for q, d, _, _ in recs}
    gamma = {p: init_gamma for p in pairs}
    pos_count = np.bincount([k - 1 for _, _, k, _ in recs], minlength=K).astype(float)

    prev = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        theta_num = np.zeros(K)
        gamma_num: dict = defaultdict(float)
        gamma_den: dict = defaultdict(float)
        for q, d, k, c in recs:
            t, g = theta[k - 1], gamma[(q, d)]
            if c:
                pe = pr = 1.0
            else:
                denom = max(1.0 - t * g, _EPS)
                pe = t * (1.0 - g) / denom
                pr = (1.0 - t) * g / denom
            theta_num[k - 1] += pe
            gamma_num[(q, d)] += pr
            gamma_den[(q, d)] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            new_theta = np.where(pos_count > 0, theta_num / np.maximum(pos_count, 1), theta)
        theta = np.clip(new_theta, _EPS, 1 - _EPS)
        gamma = {p: float(np.clip(gamma_num[p] / gamma_den[p], _EPS, 1 - _EPS)) for p in pairs}
        ll = _pbm_loglik(recs, theta, gamma)
        if abs(ll - prev) <= tol * max(1.0, abs(ll)):
            converged = True
            break
        prev = ll

    rel = theta / theta[0]
    res = PBMResult(
        examination=rel,
        attractiveness=gamma,
        log_likelihood=float(ll),
        n_iter=it,
        converged=converged,
        raw_examination=theta.copy(),
    )
    res._prior = float(np.mean(list(gamma.values())))
    return res


def randomization_propensities(
    records: Iterable[Sequence],
    n_positions: int | None = None,
    smoothing: float = 0.0,
) -> np.ndarray:
    """RandTop-n propensities ``p_k = CTR_k / CTR_1`` from randomized logs.

    Valid only when the top-n results were shuffled uniformly at random, so
    relevance is independent of rank. ``smoothing`` adds pseudo-counts
    (``(clicks + s) / (impressions + 2 s)``) for sparse ranks.
    """
    recs = _records(records)
    K = n_positions or max(k for _, _, k, _ in recs)
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    clicks = np.zeros(K)
    imps = np.zeros(K)
    for _, _, k, c in recs:
        if k <= K:
            clicks[k - 1] += c
            imps[k - 1] += 1
    if np.any(imps == 0):
        raise ValueError("every position needs at least one impression")
    ctr = (clicks + smoothing) / (imps + 2 * smoothing)
    if ctr[0] <= 0:
        raise ValueError("rank 1 has no clicks; propensities are undefined")
    return ctr / ctr[0]


@dataclass
class CascadeResult:
    attractiveness: dict  # (query, item) -> alpha
    examined: dict  # (query, item) -> number of examined impressions
    log_likelihood: float


def fit_cascade_model(
    sessions: Iterable[tuple[Hashable, Sequence[Hashable], Sequence[int]]],
    prior: tuple[float, float] = (1.0, 1.0),
) -> CascadeResult:
    """Cascade-model attractiveness from sessions ``(query, ranking, clicks)``.

    ``clicks`` lists 0/1 per rank. Results down to the first click (or the
    whole list if nothing was clicked) are examined; the MLE is
    ``alpha_{q,d} = clicks / examined``, smoothed with a Beta ``prior``.
    Sessions with more than one click violate the model; only the first click
    is used.
    """
    a, b = prior
    if a < 0 or b < 0:
        raise ValueError("prior pseudo-counts must be non-negative")
    clicked: dict = defaultdict(float)
    examined: dict = defaultdict(float)
    sess = list(sessions)
    if not sess:
        raise ValueError("no sessions")
    for q, ranking, cl in sess:
        if len(ranking) != len(cl):
            raise ValueError("ranking and clicks must have equal length")
        for d, c in zip(ranking, cl):
            examined[(q, d)] += 1
            if c:
                clicked[(q, d)] += 1
                break
    alpha = {p: (clicked[p] + a) / (examined[p] + a + b) if examined[p] + a + b > 0 else 0.0
             for p in examined}
    ll = 0.0
    for q, ranking, cl in sess:
        for d, c in zip(ranking, cl):
            p = float(np.clip(alpha[(q, d)], _EPS, 1 - _EPS))
            if c:
                ll += np.log(p)
                break
            ll += np.log1p(-p)
    return CascadeResult(attractiveness=dict(alpha), examined=dict(examined), log_likelihood=float(ll))


def fit_power_law(propensities: Sequence[float]) -> float:
    """Least-squares ``eta`` for ``p_k = k^-eta`` (through the origin in log-log)."""
    p = np.asarray(propensities, dtype=float)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("need at least two propensities")
    if np.any(p <= 0):
        raise ValueError("propensities must be positive")
    x = np.log(np.arange(1, len(p) + 1))
    y = np.log(p / p[0])
    return float(-(x @ y) / (x @ x))


def click_log_likelihood(model: PBMResult, records: Iterable[Sequence]) -> float:
    """Mean per-impression log-likelihood of ``records`` under a fitted PBM."""
    recs = _records(records)
    total = 0.0
    for q, d, k, c in recs:
        if k > len(model.raw_examination):
            raise ValueError("position outside the fitted range")
        p = float(np.clip(model.click_probability(q, d, k), _EPS, 1 - _EPS))
        total += np.log(p) if c else np.log1p(-p)
    return total / len(recs)


def click_perplexity(model: PBMResult, records: Iterable[Sequence]) -> np.ndarray:
    """Per-rank perplexity ``2^(-mean log2 P(observed click))``; 1 is perfect, 2 is coin flip."""
    recs = _records(records)
    K = len(model.raw_examination)
    sums = np.zeros(K)
    counts = np.zeros(K)
    for q, d, k, c in recs:
        if k > K:
            raise ValueError("position outside the fitted range")
        p = float(np.clip(model.click_probability(q, d, k), _EPS, 1 - _EPS))
        sums[k - 1] += np.log2(p) if c else np.log2(1 - p)
        counts[k - 1] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, 2.0 ** (-sums / np.maximum(counts, 1)), np.nan)
