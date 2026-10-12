"""Paired significance testing for comparing recommenders on the same users.

Two models evaluated on the same test users produce *paired* per-user
scores, so the right question is whether the per-user differences
``d_u = score_b(u) - score_a(u)`` are centred away from zero. Comparing the
two means with an unpaired test throws away that pairing and loses power.

:func:`paired_comparison` reports, for the differences:

* a two-sided **randomization (sign-flip) test**. Under the null of
  exchangeable systems each ``d_u`` is equally likely to have either sign.
  The p-value uses the ``(b + 1) / (B + 1)`` correction, so it is never 0
  (Smucker, Allan & Carterette, CIKM 2007, recommend this test for IR
  evaluation).
* a **paired t-test** and a **Wilcoxon signed-rank** test (from ``scipy``),
* a **paired bootstrap** percentile confidence interval for the mean
  difference, and
* the standardized effect size ``d_z = mean(d) / sd(d)``.

:func:`holm_adjust` applies the Holm–Bonferroni step-down correction when
several models are compared against one baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

MetricCallable = Callable[..., float]


def per_user_scores(
    metric: MetricCallable,
    recommended_lists: Sequence[Sequence],
    relevant_lists: Sequence,
    k: Optional[int] = None,
) -> np.ndarray:
    """Per-user metric values (the un-averaged form of ``mean_metric``)."""
    if len(recommended_lists) != len(relevant_lists):
        raise ValueError("recommended_lists and relevant_lists must align per user")
    return np.array(
        [
            metric(recs, rels) if k is None else metric(recs, rels, k)
            for recs, rels in zip(recommended_lists, relevant_lists)
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class PairedComparison:
    """Paired test of system B against system A on the same users."""

    n_users: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    confidence: float
    randomization_p: float
    t_p: float
    wilcoxon_p: float
    effect_size_dz: float
    n_resamples: int

    def significant(self, alpha: float = 0.05) -> bool:
        """True when the randomization p-value is below ``alpha``."""
        return self.randomization_p < alpha

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _check_scores(scores_a, scores_b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("scores_a and scores_b must be 1-D and the same length")
    if a.size < 2:
        raise ValueError("need at least two paired users")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("scores must be finite")
    return a, b


def paired_comparison(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare paired per-user scores of two systems (B minus A)."""
    a, b = _check_scores(scores_a, scores_b)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    d = b - a
    n = d.size
    observed = float(d.mean())
    rng = np.random.default_rng(seed)

    # Sign-flip randomization test, processed in chunks to bound memory.
    extreme = 0
    remaining = int(n_resamples)
    tol = 1e-12 * max(1.0, abs(observed))
    while remaining > 0:
        m = min(remaining, max(1, 2_000_000 // n))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
        means = (signs * d).mean(axis=1)
        extreme += int(np.sum(np.abs(means) >= abs(observed) - tol))
        remaining -= m
    rand_p = (extreme + 1) / (n_resamples + 1)

    # Paired bootstrap percentile interval for the mean difference.
    boot = np.empty(int(n_resamples))
    done = 0
    while done < n_resamples:
        m = min(int(n_resamples) - done, max(1, 2_000_000 // n))
        idx = rng.integers(0, n, size=(m, n))
        boot[done:done + m] = d[idx].mean(axis=1)
        done += m
    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = (float(x) for x in np.quantile(boot, [tail, 1.0 - tail]))

    sd = float(d.std(ddof=1))
    if sd == 0.0:
        t_p = 1.0 if observed == 0.0 else 0.0
        dz = 0.0 if observed == 0.0 else float(np.sign(observed) * np.inf)
    else:
        t_p = float(stats.ttest_rel(b, a).pvalue)
        dz = observed / sd
    if np.all(d == 0.0):
        w_p = 1.0
    else:
        w_p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)

    return PairedComparison(
        n_users=int(n),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=float(confidence),
        randomization_p=float(rand_p),
        t_p=t_p,
        wilcoxon_p=w_p,
        effect_size_dz=float(dz),
        n_resamples=int(n_resamples),
    )


def compare_recommenders(
    metric: MetricCallable,
    recommended_a: Sequence[Sequence],
    recommended_b: Sequence[Sequence],
    relevant_lists: Sequence,
    k: Optional[int] = None,
    **kwargs,
) -> PairedComparison:
    """Score two recommenders' lists per user with ``metric`` and compare them."""
    if len(recommended_a) != len(recommended_b):
        raise ValueError("both systems must be evaluated on the same users")
    scores_a = per_user_scores(metric, recommended_a, relevant_lists, k)
    scores_b = per_user_scores(metric, recommended_b, relevant_lists, k)
    return paired_comparison(scores_a, scores_b, **kwargs)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be 1-D")
    if np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError("p_values must be in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def render_significance_section(
    comparisons: dict,
    baseline: str,
    metric_label: str,
    alpha: float = 0.05,
) -> str:
    """Markdown table of paired comparisons against ``baseline`` with Holm-adjusted p-values."""
    names = list(comparisons)
    adjusted = holm_adjust([comparisons[n].randomization_p for n in names]) if names else []
    lines = [
        f"## Paired significance vs {baseline} ({metric_label})",
        "",
        "Sign-flip randomization test on per-user differences (model − baseline); "
        "p-values Holm-adjusted across models.",
        "",
        "| Model | Mean diff | CI low | CI high | p (rand.) | p (Holm) | d_z | Significant |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, p_holm in zip(names, adjusted):
        c = comparisons[name]
        lines.append(
            f"| {name} | {c.mean_diff:+.4f} | {c.ci_low:+.4f} | {c.ci_high:+.4f} | "
            f"{c.randomization_p:.4f} | {p_holm:.4f} | {c.effect_size_dz:+.3f} | "
            f"{'yes' if p_holm < alpha else 'no'} |"
        )
    return "\n".join(lines) + "\n"
