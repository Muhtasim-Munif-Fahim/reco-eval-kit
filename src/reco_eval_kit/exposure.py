"""Expected exposure of stochastic rankings (Diaz et al., CIKM 2020).

A deterministic ranker always shows the same items in the same slots, so two
equally relevant items can get very different attention. *Expected exposure*
evaluates a ranking **policy** (a distribution over rankings) by the exposure
each item receives on average, under the RBP browsing model where rank ``i``
(1-based) is examined with probability ``gamma ** (i - 1)`` and ranks beyond
the cutoff ``k`` get none:

* ``epsilon``  -- the policy's expected exposure per item, estimated from
  sampled rankings;
* ``epsilon*`` -- the target exposure of an *ideal* policy that puts every
  relevant item above every non-relevant one in uniformly random order, so
  equally relevant items get equal exposure (``m`` relevant items share the
  first ``m`` slots; non-relevant candidates share the rest);
* ``EE-L = ||epsilon - epsilon*||^2 = EE-D - EE-R + ||epsilon*||^2`` (lower
  is better), with **disparity** ``EE-D = ||epsilon||^2`` (how unequally the
  policy spreads attention; lower is fairer) and **relevance**
  ``EE-R = 2 epsilon . epsilon*`` (how much attention goes to relevant items;
  higher is better).

:func:`plackett_luce_rankings` turns any scored candidate list into a
stochastic policy via Gumbel-max Plackett-Luce sampling with a temperature
(``temperature -> 0`` recovers the deterministic ranking), so the
relevance/disparity trade-off of a model can be traced by sweeping it.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ExpectedExposure",
    "rbp_exposure",
    "target_exposure",
    "policy_exposure",
    "plackett_luce_rankings",
    "expected_exposure",
    "mean_expected_exposure",
]


def _check_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    return gamma


def _check_k(k: int) -> int:
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    return int(k)


def _relevant_set(relevant) -> set:
    if isinstance(relevant, Mapping):
        return {item for item, gain in relevant.items() if float(gain) > 0}
    return set(relevant)


def rbp_exposure(n: int, gamma: float = 0.5, k: Optional[int] = None) -> np.ndarray:
    """Exposure ``gamma ** (i - 1)`` of ranks ``1..n``, zero beyond cutoff ``k``."""
    gamma = _check_gamma(gamma)
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError("n must be a non-negative integer")
    out = gamma ** np.arange(int(n), dtype=float)
    if k is not None:
        out[_check_k(k):] = 0.0
    return out


def target_exposure(
    n_relevant: int, n_candidates: int, gamma: float = 0.5, k: Optional[int] = None
) -> tuple[float, float]:
    """Per-item exposure ``(relevant, non_relevant)`` under the ideal policy.

    The ideal policy ranks the ``n_relevant`` relevant items first in uniformly
    random order, then the remaining candidates in uniformly random order, so
    each group splits its slots' exposure evenly.
    """
    m, n = int(n_relevant), int(n_candidates)
    if m < 0 or n < m:
        raise ValueError("need 0 <= n_relevant <= n_candidates")
    exp = rbp_exposure(n, gamma, k)
    rel = float(exp[:m].mean()) if m else 0.0
    nonrel = float(exp[m:].mean()) if n > m else 0.0
    return rel, nonrel


def policy_exposure(rankings: Sequence[Sequence[Hashable]], gamma: float = 0.5, k: Optional[int] = None) -> dict:
    """Mean RBP exposure per item over sampled rankings (items never shown get 0)."""
    if len(rankings) == 0:
        raise ValueError("rankings must not be empty")
    gamma = _check_gamma(gamma)
    totals: dict = {}
    for ranking in rankings:
        ranking = list(ranking)
        if len(set(ranking)) != len(ranking):
            raise ValueError("a ranking contains duplicate items")
        exp = rbp_exposure(len(ranking), gamma, k)
        for item, e in zip(ranking, exp):
            totals[item] = totals.get(item, 0.0) + float(e)
    n = float(len(rankings))
    return {item: total / n for item, total in totals.items()}


def plackett_luce_rankings(
    candidates: Sequence[Hashable],
    scores: Sequence[float],
    n_samples: int = 100,
    temperature: float = 1.0,
    k: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[list]:
    """Sample rankings from a Plackett-Luce policy ``P(i) ∝ exp(score_i / T)``.

    Uses the Gumbel-max trick: sort ``score / T + Gumbel(0, 1)`` noise.
    ``temperature = 0`` returns ``n_samples`` copies of the deterministic
    score ordering (ties broken by candidate order). Rankings are truncated to
    ``k`` items when given.
    """
    candidates = list(candidates)
    s = np.asarray(scores, dtype=float)
    if s.shape != (len(candidates),):
        raise ValueError("scores must have one value per candidate")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if isinstance(n_samples, bool) or int(n_samples) != n_samples or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    cut = len(candidates) if k is None else min(_check_k(k), len(candidates))
    if temperature == 0:
        order = np.argsort(-s, kind="stable")[:cut]
        return [[candidates[i] for i in order] for _ in range(int(n_samples))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_samples)):
        keys = s / float(temperature) + rng.gumbel(size=s.size)
        order = np.argsort(-keys, kind="stable")[:cut]
        out.append([candidates[i] for i in order])
    return out


@dataclass(frozen=True)
class ExpectedExposure:
    """Expected-exposure decomposition for one query/user."""

    loss: float
    disparity: float
    relevance: float
    target_norm: float
    n_relevant: int
    n_candidates: int

    @property
    def normalized_relevance(self) -> float:
        """``EE-R / (2 ||epsilon*||^2)``: 1 for the ideal policy's attention on relevant items."""
        return self.relevance / (2.0 * self.target_norm) if self.target_norm > 0 else 0.0


def expected_exposure(
    rankings: Sequence[Sequence[Hashable]],
    relevant,
    gamma: float = 0.5,
    k: Optional[int] = None,
    n_candidates: Optional[int] = None,
) -> ExpectedExposure:
    """EE-L / EE-D / EE-R of a stochastic ranking policy for one user.

    Parameters
    ----------
    rankings : sequence of rankings
        Samples from the policy (e.g. :func:`plackett_luce_rankings`). A
        single deterministic ranking is ``[ranking]``.
    relevant : set or mapping
        Relevant items (mapping values ``> 0`` count as relevant).
    gamma : float
        RBP patience.
    k : int, optional
        Exposure cutoff.
    n_candidates : int, optional
        Size of the candidate universe the ideal policy ranks. Defaults to
        the number of distinct items in the samples plus unretrieved relevant
        items. Candidates never shown are counted with zero system exposure.
    """
    gamma = _check_gamma(gamma)
    rel = _relevant_set(relevant)
    eps = policy_exposure(rankings, gamma, k)
    universe = set(eps) | rel
    n = len(universe) if n_candidates is None else int(n_candidates)
    if n < len(universe):
        raise ValueError("n_candidates is smaller than the items seen in rankings plus relevant items")
    t_rel, t_non = target_exposure(len(rel), n, gamma, k)
    disparity = float(sum(e * e for e in eps.values()))
    relevance = 2.0 * float(sum(e * (t_rel if item in rel else t_non) for item, e in eps.items()))
    target_norm = len(rel) * t_rel ** 2 + (n - len(rel)) * t_non ** 2
    loss = max(disparity - relevance + target_norm, 0.0)
    return ExpectedExposure(
        loss=loss,
        disparity=disparity,
        relevance=relevance,
        target_norm=float(target_norm),
        n_relevant=len(rel),
        n_candidates=n,
    )


def mean_expected_exposure(
    rankings_per_user: Sequence[Sequence[Sequence[Hashable]]],
    relevant_per_user: Sequence,
    gamma: float = 0.5,
    k: Optional[int] = None,
    n_candidates=None,
) -> dict:
    """Mean ``loss``/``disparity``/``relevance``/``normalized_relevance`` over users.

    ``n_candidates`` may be ``None``, an int shared by all users, or one value
    per user. Users without relevant items are skipped.
    """
    if len(rankings_per_user) != len(relevant_per_user):
        raise ValueError("need one set of rankings per user")
    if n_candidates is None or isinstance(n_candidates, (int, np.integer)):
        sizes = [n_candidates] * len(rankings_per_user)
    else:
        sizes = list(n_candidates)
        if len(sizes) != len(rankings_per_user):
            raise ValueError("n_candidates must be a scalar or one value per user")
    rows = [
        expected_exposure(r, rel, gamma=gamma, k=k, n_candidates=n)
        for r, rel, n in zip(rankings_per_user, relevant_per_user, sizes)
        if _relevant_set(rel)
    ]
    if not rows:
        return {"loss": 0.0, "disparity": 0.0, "relevance": 0.0, "normalized_relevance": 0.0, "users": 0}
    return {
        "loss": float(np.mean([r.loss for r in rows])),
        "disparity": float(np.mean([r.disparity for r in rows])),
        "relevance": float(np.mean([r.relevance for r in rows])),
        "normalized_relevance": float(np.mean([r.normalized_relevance for r in rows])),
        "users": len(rows),
    }
