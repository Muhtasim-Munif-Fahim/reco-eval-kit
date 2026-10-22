"""User-group (consumer-side) fairness of recommendation quality.

Average NDCG hides *who* the recommender serves well. Recommenders usually
serve active users much better than users with short histories
(Li, Chen, Fu, Ge & Zhang, "User-oriented Fairness in Recommendation",
WWW 2021, found this on every dataset they tested), and a model can raise the
mean while widening the gap. This module segments users and audits per-group
quality:

* :func:`activity_groups` splits users by training-interaction count. With
  ``"advantaged"`` it uses Li et al.'s split (the top ``top_fraction`` most
  active users vs the rest); with ``"quantile"`` it uses ``n_groups``
  equal-frequency activity buckets.
* :func:`cold_start_groups` labels users with at most ``max_interactions``
  training interactions as ``"cold"``.
* :func:`group_fairness_report` summarises per-user scores by group:
  per-group mean with a t confidence interval, the largest absolute gap
  (``UGF``, Li et al.'s user-group fairness metric for two groups), the
  min/max ratio, the coefficient of variation and the Gini index of group
  means. It tests "all groups are equal" with a label-permutation test on
  the gap (exact for any number of groups) plus Welch's t (two groups) or
  Kruskal–Wallis (more groups).
* :func:`compare_group_fairness` audits several models on the same users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "GroupFairnessReport",
    "activity_groups",
    "cold_start_groups",
    "group_fairness_report",
    "compare_group_fairness",
    "render_group_fairness_section",
]


def _activity(interactions: pd.DataFrame, users: Optional[Sequence], user_col: str) -> pd.Series:
    if user_col not in interactions.columns:
        raise ValueError(f"interactions must have a {user_col!r} column")
    counts = interactions.groupby(user_col).size()
    if users is not None:
        counts = counts.reindex(list(users), fill_value=0)
    return counts


def activity_groups(
    interactions: pd.DataFrame,
    users: Optional[Sequence] = None,
    method: str = "advantaged",
    top_fraction: float = 0.05,
    n_groups: int = 3,
    user_col: str = "user_id",
) -> Dict[object, str]:
    """Map each user to an activity group from training interaction counts.

    ``method="advantaged"`` labels the ``ceil(top_fraction * n_users)`` most
    active users ``"advantaged"`` and the rest ``"disadvantaged"`` (ties at
    the cut-off are broken by user order). ``method="quantile"`` labels
    ``n_groups`` equal-frequency buckets ``"Q1"`` (least active) ...
    ``"Qn"``. ``users`` restricts / orders the population; users absent from
    ``interactions`` count as zero activity.
    """
    counts = _activity(interactions, users, user_col)
    n = len(counts)
    if n < 2:
        raise ValueError("need at least two users to form groups")
    order = np.argsort(-counts.to_numpy(), kind="mergesort")
    if method == "advantaged":
        if not 0.0 < top_fraction < 1.0:
            raise ValueError("top_fraction must be in (0, 1)")
        n_top = min(max(int(np.ceil(top_fraction * n)), 1), n - 1)
        labels = np.array(["disadvantaged"] * n, dtype=object)
        labels[order[:n_top]] = "advantaged"
    elif method == "quantile":
        if not 2 <= n_groups <= n:
            raise ValueError("n_groups must be between 2 and the number of users")
        ascending = order[::-1]
        labels = np.empty(n, dtype=object)
        for g, chunk in enumerate(np.array_split(ascending, n_groups)):
            labels[chunk] = f"Q{g + 1}"
    else:
        raise ValueError("method must be 'advantaged' or 'quantile'")
    return dict(zip(counts.index.tolist(), labels.tolist()))


def cold_start_groups(
    interactions: pd.DataFrame,
    users: Optional[Sequence] = None,
    max_interactions: int = 5,
    user_col: str = "user_id",
) -> Dict[object, str]:
    """Label users with ``<= max_interactions`` training interactions ``"cold"``, else ``"warm"``."""
    if max_interactions < 0:
        raise ValueError("max_interactions must be non-negative")
    counts = _activity(interactions, users, user_col)
    return {u: ("cold" if c <= max_interactions else "warm") for u, c in counts.items()}


def _gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    if v.size == 0 or v.sum() <= 0:
        return 0.0
    n = v.size
    return float((2.0 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


@dataclass(frozen=True)
class GroupFairnessReport:
    """Per-group quality and disparity summary for one model."""

    groups: Dict[str, Dict[str, float]]
    overall_mean: float
    gap: float
    best_group: str
    worst_group: str
    ratio: float
    coefficient_of_variation: float
    gini: float
    permutation_p: float
    parametric_test: str
    parametric_p: float
    n_users: int
    extra: Dict[str, float] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.groups).T


def _gap(scores: np.ndarray, codes: np.ndarray, n_groups: int) -> float:
    sums = np.bincount(codes, weights=scores, minlength=n_groups)
    cnts = np.bincount(codes, minlength=n_groups)
    means = sums / cnts
    return float(means.max() - means.min())


def group_fairness_report(
    scores: Sequence[float],
    groups: Sequence,
    ci: float = 0.95,
    n_permutations: int = 2000,
    seed: Optional[int] = None,
) -> GroupFairnessReport:
    """Summarise per-user ``scores`` by ``groups`` (one label per user).

    The permutation p-value is ``(b + 1) / (B + 1)`` where ``b`` counts label
    shuffles whose max-minus-min group-mean gap is at least the observed one.
    """
    s = np.asarray(scores, dtype=float).ravel()
    g = np.asarray(list(groups), dtype=object).ravel()
    if s.size != g.size:
        raise ValueError("scores and groups must have one entry per user")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    labels, codes = np.unique(g.astype(str), return_inverse=True)
    k = labels.size
    if k < 2:
        raise ValueError("need at least two groups")
    cnts = np.bincount(codes, minlength=k)
    if cnts.min() < 1:
        raise ValueError("every group needs at least one user")
    per_group: Dict[str, Dict[str, float]] = {}
    means = np.zeros(k)
    for j, name in enumerate(labels):
        x = s[codes == j]
        m = float(x.mean())
        means[j] = m
        if x.size > 1:
            se = float(x.std(ddof=1) / np.sqrt(x.size))
            half = float(stats.t.ppf(0.5 + ci / 2, x.size - 1) * se)
        else:
            se, half = float("nan"), float("nan")
        per_group[str(name)] = {"n_users": float(x.size), "mean": m, "se": se,
                                "ci_low": m - half, "ci_high": m + half}
    best = str(labels[int(np.argmax(means))])
    worst = str(labels[int(np.argmin(means))])
    gap = float(means.max() - means.min())
    ratio = float(means.min() / means.max()) if means.max() > 0 else 1.0
    cv = float(means.std() / means.mean()) if means.mean() > 0 else 0.0
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(n_permutations)):
        if _gap(s, rng.permutation(codes), k) >= gap - 1e-12:
            exceed += 1
    perm_p = (exceed + 1) / (int(n_permutations) + 1)
    if k == 2:
        a, b = s[codes == 0], s[codes == 1]
        if a.size > 1 and b.size > 1 and (a.var() > 0 or b.var() > 0):
            par_p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        else:
            par_p = float("nan")
        test = "welch_t"
    else:
        try:
            par_p = float(stats.kruskal(*[s[codes == j] for j in range(k)]).pvalue)
        except ValueError:
            par_p = float("nan")
        test = "kruskal_wallis"
    return GroupFairnessReport(groups=per_group, overall_mean=float(s.mean()), gap=gap,
                               best_group=best, worst_group=worst, ratio=ratio,
                               coefficient_of_variation=cv, gini=_gini(means),
                               permutation_p=float(perm_p), parametric_test=test,
                               parametric_p=par_p, n_users=int(s.size))


def compare_group_fairness(
    scores_by_model: Mapping[str, Sequence[float]],
    groups: Sequence,
    **kwargs,
) -> Dict[str, GroupFairnessReport]:
    """:func:`group_fairness_report` for every model on the same users and groups."""
    return {name: group_fairness_report(scores, groups, **kwargs) for name, scores in scores_by_model.items()}


def render_group_fairness_section(
    reports: Mapping[str, GroupFairnessReport],
    metric_name: str = "NDCG@K",
    grouping: str = "user activity",
) -> str:
    """Markdown table: per-group means, gap, ratio and permutation p per model."""
    if not reports:
        return ""
    group_names = sorted({g for r in reports.values() for g in r.groups})
    header = ["Model", "Overall"] + group_names + ["Gap (UGF)", "Min/Max", "Perm. p"]
    lines = [f"## User-group fairness ({metric_name} by {grouping})", "",
             "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for name, r in reports.items():
        cells = [name, f"{r.overall_mean:.4f}"]
        cells += [f"{r.groups[g]['mean']:.4f}" if g in r.groups else "–" for g in group_names]
        cells += [f"{r.gap:.4f}", f"{r.ratio:.3f}", f"{r.permutation_p:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    first = next(iter(reports.values()))
    sizes = ", ".join(f"{g}: {int(v['n_users'])}" for g, v in first.groups.items())
    lines += ["", f"Group sizes — {sizes}. Gap = best minus worst group mean; "
              "the permutation test shuffles group labels.", ""]
    return "\n".join(lines)
