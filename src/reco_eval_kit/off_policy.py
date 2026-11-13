"""Off-policy evaluation (OPE) of recommendation policies from logged bandit feedback.

A deployed (logging) policy ``mu`` showed action ``a_i`` in context ``x_i`` with
known propensity ``p_i = mu(a_i | x_i)`` and observed reward ``r_i`` (click,
purchase, ...). We want the value ``V(pi) = E_x E_{a ~ pi(.|x)} [r]`` of a new
target policy ``pi`` without deploying it. Inputs are plain arrays:

* ``actions`` ``(n,)`` integer action indices,
* ``rewards`` ``(n,)``,
* ``pscore`` ``(n,)`` logging propensities of the logged actions,
* ``action_dist`` ``(n, n_actions)`` target-policy probabilities ``pi(a | x_i)``
  (a deterministic policy is a one-hot matrix),
* ``estimated_rewards`` ``(n, n_actions)`` reward-model predictions
  ``q_hat(x_i, a)`` for the model-based estimators.

Estimators (``w_i = pi(a_i | x_i) / p_i``):

* **IPS** (Horvitz-Thompson): ``mean(w_i r_i)``; unbiased when ``mu`` covers ``pi``.
* **SNIPS** (Swaminathan & Joachims 2015): ``sum(w_i r_i) / sum(w_i)``; biased
  but consistent with much lower variance.
* **Clipped IPS**: weights capped at ``max_weight``.
* **DM** (direct method): ``mean_i sum_a pi(a|x_i) q_hat(x_i, a)``.
* **DR** (doubly robust; Dudik, Langford & Li 2011):
  ``DM_i + w_i (r_i - q_hat(x_i, a_i))``; unbiased if *either* the
  propensities or the reward model are correct.
* **Switch-DR** (Wang, Agarwal & Dudik 2017): uses the DR correction only for
  samples with ``w_i <= tau`` and the model elsewhere (``tau=inf`` -> DR,
  ``tau=0`` -> DM).

Every estimator returns an :class:`OPEEstimate` with a normal-approximation
confidence interval from the per-sample contributions (for SNIPS via the
delta method), and Kish's effective sample size of the importance weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Optional

import numpy as np

__all__ = [
    "OPEEstimate",
    "importance_weights",
    "ips_estimate",
    "snips_estimate",
    "clipped_ips_estimate",
    "direct_method_estimate",
    "doubly_robust_estimate",
    "switch_dr_estimate",
    "evaluate_policy",
]


@dataclass(frozen=True)
class OPEEstimate:
    """Point estimate of a policy value with a normal-approximation CI."""

    estimator: str
    value: float
    std_error: float
    ci_low: float
    ci_high: float
    n: int
    effective_sample_size: float

    def as_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "value": self.value,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "effective_sample_size": self.effective_sample_size,
        }


def _check(actions, rewards, pscore, action_dist):
    a = np.asarray(actions)
    r = np.asarray(rewards, dtype=float)
    p = np.asarray(pscore, dtype=float)
    pi = np.asarray(action_dist, dtype=float)
    if a.ndim != 1 or r.ndim != 1 or p.ndim != 1:
        raise ValueError("actions, rewards and pscore must be 1-D")
    n = a.shape[0]
    if n == 0:
        raise ValueError("need at least one logged sample")
    if r.shape[0] != n or p.shape[0] != n:
        raise ValueError("actions, rewards and pscore must have the same length")
    if pi.ndim != 2 or pi.shape[0] != n:
        raise ValueError("action_dist must have shape (n, n_actions)")
    if not np.issubdtype(a.dtype, np.integer):
        if not np.all(np.equal(np.mod(a, 1), 0)):
            raise ValueError("actions must be integer indices")
        a = a.astype(int)
    if a.min() < 0 or a.max() >= pi.shape[1]:
        raise ValueError("actions must index columns of action_dist")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(p)) and np.all(np.isfinite(pi))):
        raise ValueError("inputs must be finite")
    if np.any(p <= 0.0) or np.any(p > 1.0):
        raise ValueError("pscore must be in (0, 1]")
    if np.any(pi < 0.0) or not np.allclose(pi.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("action_dist rows must be probability distributions")
    return a, r, p, pi


def _check_q(estimated_rewards, shape) -> np.ndarray:
    q = np.asarray(estimated_rewards, dtype=float)
    if q.shape != shape:
        raise ValueError("estimated_rewards must have the same shape as action_dist")
    if not np.all(np.isfinite(q)):
        raise ValueError("estimated_rewards must be finite")
    return q


def _z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def _kish(w: np.ndarray) -> float:
    s2 = float(np.sum(w * w))
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0


def _from_contributions(name, contrib, w, confidence) -> OPEEstimate:
    n = contrib.shape[0]
    value = float(contrib.mean())
    se = float(contrib.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = _z(confidence)
    return OPEEstimate(name, value, se, value - z * se, value + z * se, n, _kish(w))


def importance_weights(actions, pscore, action_dist) -> np.ndarray:
    """``w_i = pi(a_i | x_i) / mu(a_i | x_i)``."""
    a, _, p, pi = _check(actions, np.zeros(len(actions)), pscore, action_dist)
    return pi[np.arange(a.shape[0]), a] / p


def ips_estimate(actions, rewards, pscore, action_dist, confidence: float = 0.95) -> OPEEstimate:
    """Inverse propensity scoring ``mean(w r)``."""
    a, r, p, pi = _check(actions, rewards, pscore, action_dist)
    w = pi[np.arange(a.shape[0]), a] / p
    return _from_contributions("ips", w * r, w, confidence)


def clipped_ips_estimate(
    actions, rewards, pscore, action_dist, max_weight: float = 10.0, confidence: float = 0.95
) -> OPEEstimate:
    """IPS with importance weights clipped at ``max_weight`` (biased, lower variance)."""
    if not max_weight > 0:
        raise ValueError("max_weight must be positive")
    a, r, p, pi = _check(actions, rewards, pscore, action_dist)
    w = np.minimum(pi[np.arange(a.shape[0]), a] / p, float(max_weight))
    return _from_contributions("clipped_ips", w * r, w, confidence)


def snips_estimate(actions, rewards, pscore, action_dist, confidence: float = 0.95) -> OPEEstimate:
    """Self-normalised IPS ``sum(w r) / sum(w)`` with a delta-method CI."""
    a, r, p, pi = _check(actions, rewards, pscore, action_dist)
    w = pi[np.arange(a.shape[0]), a] / p
    n = a.shape[0]
    sw = float(w.sum())
    if sw <= 0.0:
        raise ValueError("all importance weights are zero: target policy never takes a logged action")
    value = float(np.sum(w * r) / sw)
    # Influence function of a ratio estimator: w_i (r_i - V) / mean(w).
    infl = w * (r - value) / (sw / n)
    se = float(infl.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = _z(confidence)
    return OPEEstimate("snips", value, se, value - z * se, value + z * se, n, _kish(w))


def direct_method_estimate(action_dist, estimated_rewards, confidence: float = 0.95) -> OPEEstimate:
    """Direct method ``mean_i sum_a pi(a|x_i) q_hat(x_i, a)``.

    The interval reflects only context sampling, not reward-model error.
    """
    pi = np.asarray(action_dist, dtype=float)
    if pi.ndim != 2 or pi.shape[0] == 0:
        raise ValueError("action_dist must have shape (n, n_actions)")
    if np.any(pi < 0.0) or not np.allclose(pi.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("action_dist rows must be probability distributions")
    q = _check_q(estimated_rewards, pi.shape)
    contrib = np.sum(pi * q, axis=1)
    return _from_contributions("dm", contrib, np.ones(pi.shape[0]), confidence)


def _dr_parts(actions, rewards, pscore, action_dist, estimated_rewards):
    a, r, p, pi = _check(actions, rewards, pscore, action_dist)
    q = _check_q(estimated_rewards, pi.shape)
    idx = np.arange(a.shape[0])
    w = pi[idx, a] / p
    dm = np.sum(pi * q, axis=1)
    correction = w * (r - q[idx, a])
    return w, dm, correction


def doubly_robust_estimate(
    actions, rewards, pscore, action_dist, estimated_rewards, confidence: float = 0.95
) -> OPEEstimate:
    """Doubly robust ``DM_i + w_i (r_i - q_hat(x_i, a_i))`` (Dudik et al. 2011)."""
    w, dm, corr = _dr_parts(actions, rewards, pscore, action_dist, estimated_rewards)
    return _from_contributions("dr", dm + corr, w, confidence)


def switch_dr_estimate(
    actions, rewards, pscore, action_dist, estimated_rewards, tau: float = 10.0,
    confidence: float = 0.95,
) -> OPEEstimate:
    """Switch-DR (Wang et al. 2017): DR correction only where ``w_i <= tau``."""
    if not tau >= 0:
        raise ValueError("tau must be non-negative")
    w, dm, corr = _dr_parts(actions, rewards, pscore, action_dist, estimated_rewards)
    keep = w <= tau
    return _from_contributions("switch_dr", dm + np.where(keep, corr, 0.0), np.where(keep, w, 0.0), confidence)


def evaluate_policy(
    actions, rewards, pscore, action_dist, estimated_rewards: Optional[np.ndarray] = None,
    *, max_weight: float = 10.0, tau: float = 10.0, confidence: float = 0.95,
) -> dict:
    """Run every applicable estimator; returns ``{name: OPEEstimate}``.

    Model-based estimators (DM, DR, Switch-DR) are included only when
    ``estimated_rewards`` is given.
    """
    out = {
        "ips": ips_estimate(actions, rewards, pscore, action_dist, confidence),
        "snips": snips_estimate(actions, rewards, pscore, action_dist, confidence),
        "clipped_ips": clipped_ips_estimate(actions, rewards, pscore, action_dist, max_weight, confidence),
    }
    if estimated_rewards is not None:
        out["dm"] = direct_method_estimate(action_dist, estimated_rewards, confidence)
        out["dr"] = doubly_robust_estimate(actions, rewards, pscore, action_dist, estimated_rewards, confidence)
        out["switch_dr"] = switch_dr_estimate(
            actions, rewards, pscore, action_dist, estimated_rewards, tau, confidence
        )
    return out
