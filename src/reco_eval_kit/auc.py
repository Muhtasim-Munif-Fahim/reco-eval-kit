"""Score-based ranking metrics: per-user AUC, GAUC, R-precision and F1@k.

Top-K cutoff metrics ignore everything below the cutoff. Pairwise AUC looks
at the whole scored candidate list, and it is the standard offline metric
for CTR / candidate-ranking models. Pooling every impression into one AUC
mixes users with very different base rates and rewards a model for ranking
*users* rather than items. Group AUC (GAUC, Zhu et al. 2017, "Optimized Cost
per Click in Taobao Display Advertising") instead averages per-user AUCs
weighted by impressions (or clicks), which removes that bias.

* :func:`user_auc` – Mann–Whitney AUC of one user's scored candidates; tied
  scores count half, matching ``sklearn.metrics.roc_auc_score``;
* :func:`list_auc` – AUC of a ranked list against a catalog, where every
  unrecommended catalog item ties at the bottom;
* :func:`gauc` – impression-, click- or uniformly-weighted mean of per-user
  AUCs, skipping users with only positives or only negatives;
* :func:`r_precision` – precision at ``R = |relevant|``;
* :func:`f1_at_k` – harmonic mean of precision@k and recall@k.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from .metrics import (
    RelevanceSpec,
    _check_k,
    _gain,
    _positive_count,
    _validate_recommended,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "GAUCResult",
    "user_auc",
    "list_auc",
    "gauc",
    "r_precision",
    "f1_at_k",
]

_WEIGHTINGS = ("impressions", "clicks", "uniform")


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _auc_counts(scores: Mapping, positives: Collection) -> tuple[float, int, int]:
    items = list(scores)
    values = []
    for item in items:
        value = float(scores[item])
        if value != value:  # NaN
            raise ValueError("scores must not contain NaN")
        values.append(value)
    pos_set = set(positives)
    labels = [item in pos_set for item in items]
    n_pos = sum(labels)
    n_neg = len(items) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, n_neg
    ranks = _average_ranks(values)
    rank_sum = sum(r for r, is_pos in zip(ranks, labels) if is_pos)
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc, n_pos, n_neg


def user_auc(scores: Mapping, positives: Collection) -> float:
    """AUC of one user's candidate scores against their positive items.

    ``scores`` maps candidate item -> model score (higher = more relevant).
    Positives missing from ``scores`` are ignored: only scored candidates
    count. Raises ``ValueError`` when the candidates contain no positives
    or no negatives, because AUC is undefined there.
    """
    auc, n_pos, n_neg = _auc_counts(scores, positives)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one positive and one negative candidate")
    return auc


def list_auc(recommended: Sequence, relevant: RelevanceSpec, catalog: Collection) -> float:
    """AUC of a ranked recommendation list over a full catalog.

    Rank ``r`` gets score ``-r``. Catalog items that were not recommended
    tie below every recommended item, so a relevant item that was never
    recommended counts half against each unrecommended negative. Graded
    relevance is binarised at gain > 0.
    """
    items = _validate_recommended(recommended)
    catalog_set = set(catalog)
    missing = [item for item in items if item not in catalog_set]
    if missing:
        raise ValueError("recommended items must belong to the catalog")
    scores = {item: -float(len(items) + 1) for item in catalog_set}
    for rank, item in enumerate(items, start=1):
        scores[item] = -float(rank)
    positives = {item for item in catalog_set if _gain(item, relevant) > 0}
    return user_auc(scores, positives)


@dataclass(frozen=True)
class GAUCResult:
    """Group AUC with bookkeeping about which users contributed."""

    gauc: float
    weighting: str
    n_users: int
    n_skipped: int
    total_weight: float
    per_user: dict = field(default_factory=dict)


def gauc(
    user_scores: Mapping[object, Mapping],
    user_positives: Mapping[object, Collection],
    *,
    weighting: str = "impressions",
) -> GAUCResult:
    """Group AUC: a weighted mean of per-user AUCs.

    ``user_scores[u]`` maps candidate item -> score for user ``u``, and
    ``user_positives[u]`` holds that user's positive items (users absent from
    it have none). ``weighting`` is ``"impressions"`` (number of scored
    candidates, the usual GAUC), ``"clicks"`` (number of positives) or
    ``"uniform"``. Users whose candidates are all positive or all negative
    have no defined AUC; they are skipped and counted in ``n_skipped``.
    Raises ``ValueError`` when no user has a defined AUC.
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}")
    per_user: dict = {}
    num = 0.0
    den = 0.0
    skipped = 0
    for user, scores in user_scores.items():
        auc, n_pos, n_neg = _auc_counts(scores, user_positives.get(user, ()))
        if n_pos == 0 or n_neg == 0:
            skipped += 1
            continue
        if weighting == "impressions":
            w = float(n_pos + n_neg)
        elif weighting == "clicks":
            w = float(n_pos)
        else:
            w = 1.0
        per_user[user] = auc
        num += w * auc
        den += w
    if not per_user:
        raise ValueError("no user has both positive and negative candidates")
    return GAUCResult(
        gauc=num / den,
        weighting=weighting,
        n_users=len(per_user),
        n_skipped=skipped,
        total_weight=den,
        per_user=per_user,
    )


def r_precision(recommended: Sequence, relevant: RelevanceSpec) -> float:
    """Precision at cutoff ``R``, the user's number of relevant items.

    At that cutoff precision equals recall, and a perfect ranking scores
    1.0 no matter how many relevant items the user has. Raises
    ``ValueError`` when the relevant set is empty.
    """
    total = _positive_count(relevant)
    if total == 0:
        raise ValueError("relevant set is empty")
    return precision_at_k(recommended, relevant, total)


def f1_at_k(recommended: Sequence, relevant: RelevanceSpec, k: int) -> float:
    """Harmonic mean of precision@k and recall@k (0.0 when both are zero)."""
    _check_k(k)
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)
