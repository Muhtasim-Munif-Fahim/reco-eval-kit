"""How similar are two recommenders' ranked lists?

Accuracy metrics compare a list with held-out items. These measures compare two
models' lists directly. Use them to see whether a new model really changes what
users see, how stable a model is across retrains or seeds, or whether two
"different" baselines recommend the same head items:

* :func:`rank_biased_overlap` -- Rank-Biased Overlap (Webber, Moffat & Zobel,
  "A Similarity Measure for Indefinite Rankings", TOIS 2010). Agreement at
  depth ``d`` is ``A_d = |S[:d] & T[:d]| / d``. RBO weights it geometrically,
  ``(1 - p) * sum_d p**(d-1) * A_d``, so ``p`` sets how top-heavy it is (about
  86% of the weight sits in the first ``1 / (1 - p)`` ranks). The two
  observable quantities for finite prefixes are returned: ``min`` (the lower
  bound when the unseen tails are assumed disjoint, Webber eq. 11) and ``ext``
  (the point estimate that extrapolates the agreement seen so far, eq. 32,
  including the uneven-length case), plus the residual ``res`` so that
  ``[min, min + res]`` brackets the true RBO of the full rankings.
* :func:`jaccard_at_k` and :func:`overlap_at_k` -- set agreement of the top k.
* :func:`kendall_tau_topk_distance` -- Fagin, Kumar & Sivakumar's ``K^(p)``
  Kendall distance between top-k lists ("Comparing top k lists", SIAM J.
  Discrete Math. 2003), normalised to ``[0, 1]`` by the distance between two
  disjoint lists of the same lengths. Pairs whose order is unknown (both items
  in one list and neither in the other) cost ``penalty`` (``p = 0`` is the
  optimistic and ``p = 0.5`` the neutral choice).
* :func:`compare_rankings` averages all of these over users, and
  :func:`pairwise_rank_similarity` builds a model-by-model matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Hashable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "RBOResult",
    "rank_biased_overlap",
    "jaccard_at_k",
    "overlap_at_k",
    "kendall_tau_topk_distance",
    "compare_rankings",
    "pairwise_rank_similarity",
    "render_rank_similarity_section",
]


@dataclass(frozen=True)
class RBOResult:
    """Rank-biased overlap bounds and estimate for two finite prefixes."""

    min: float
    res: float
    ext: float
    depth: int

    @property
    def max(self) -> float:
        return self.min + self.res


def _dedup(ranking: Sequence[Hashable], k: Optional[int]) -> List[Hashable]:
    seen = set()
    out = []
    for item in ranking:
        if item in seen:
            raise ValueError(f"ranking contains duplicate item {item!r}")
        seen.add(item)
        out.append(item)
    if k is not None:
        if k < 1:
            raise ValueError("k must be >= 1")
        out = out[:k]
    return out


def _check_p(p: float) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")


def rank_biased_overlap(
    s: Sequence[Hashable],
    t: Sequence[Hashable],
    p: float = 0.9,
    k: Optional[int] = None,
) -> RBOResult:
    """Rank-biased overlap of two rankings (optionally truncated at ``k``).

    Returns :class:`RBOResult` with ``min``, ``res``, ``ext`` and the evaluated
    ``depth`` (length of the longer list). Two identical lists have
    ``ext == 1``, and disjoint lists have ``min == ext == 0``. Two empty lists
    count as identical.
    """
    _check_p(p)
    s = _dedup(s, k)
    t = _dedup(t, k)
    if len(s) > len(t):
        s, t = t, s  # s is the shorter list
    sl, ll = len(s), len(t)
    if ll == 0:
        return RBOResult(min=1.0, res=0.0, ext=1.0, depth=0)
    if sl == 0:
        # Nothing observed for one ranking: no agreement is known.
        return RBOResult(min=0.0, res=1.0, ext=0.0, depth=ll)

    # X[d] = overlap at depth d (1-indexed). Beyond the short list's end, the
    # short list is treated as fixed (its prefix S[:s]).
    x = np.zeros(ll + 1)
    seen_s, seen_t = set(), set()
    overlap = 0
    for d in range(1, ll + 1):
        if d <= sl:
            a = s[d - 1]
            b = t[d - 1]
            if a == b:
                overlap += 1
            else:
                if a in seen_t:
                    overlap += 1
                if b in seen_s:
                    overlap += 1
            seen_s.add(a)
            seen_t.add(b)
        else:
            if t[d - 1] in seen_s:
                overlap += 1
            seen_t.add(t[d - 1])
        x[d] = overlap

    depths = np.arange(1, ll + 1, dtype=float)
    weights = p ** depths
    xd = x[1:]
    xs = x[sl]
    xl = x[ll]

    # Extrapolated estimate, Webber et al. eq. 32 (reduces to eq. 30 when sl == ll).
    term = float(np.sum(xd / depths * weights))
    if ll > sl:
        tail = depths[sl:]
        term += float(np.sum(xs * (tail - sl) / (sl * tail) * (p ** tail)))
    ext = (1 - p) / p * term + ((xl - xs) / ll + xs / sl) * p ** ll

    # Lower bound, eq. 11 generalised to uneven lists by evaluating at the
    # longer depth with the observed overlaps (unseen tails assumed disjoint).
    rmin = (1 - p) / p * (
        float(np.sum((xd - xl) / depths * weights)) - xl * math.log(1 - p)
    )
    # Residual: RBO_max - RBO_min, where RBO_max credits every unseen item
    # with the best agreement still possible (Webber eq. 12-13; the same
    # construction covers uneven lengths).
    res = max(0.0, 1.0 - rmin - _rbo_certain_loss(xd, depths, p, ll, sl))
    ext = float(min(max(ext, rmin), rmin + res))
    return RBOResult(min=float(rmin), res=float(res), ext=ext, depth=ll)


def _rbo_certain_loss(xd, depths, p, ll, sl) -> float:
    """RBO mass that is *certainly* lost by observed disagreement.

    For ``d <= sl`` the observed overlap ``X_d`` is final. For
    ``sl < d <= ll`` the short list's ``d - sl`` unseen items could still
    match, and beyond ``ll`` both lists' unseen items could, so the overlap is
    at most ``X_ll + (d - sl) + (d - ll)``. The loss at each depth is
    ``(1 - max_agreement_d)``, weighted by ``(1 - p) * p**(d - 1)``.
    """
    loss = 0.0
    for d, x in zip(depths, xd):
        possible = x + (d - sl if d > sl else 0.0)
        agree = min(1.0, possible / d)
        loss += (1 - p) * p ** (d - 1) * (1 - agree)
    xl = xd[-1]
    d = ll + 1
    while True:
        possible = xl + (ll - sl) + 2 * (d - ll)
        if possible >= d:
            break
        loss += (1 - p) * p ** (d - 1) * (1 - possible / d)
        d += 1
    return loss


def overlap_at_k(s: Sequence[Hashable], t: Sequence[Hashable], k: int) -> float:
    """``|S[:k] & T[:k]| / k`` (shared items among the top k of each list)."""
    s = _dedup(s, k)
    t = _dedup(t, k)
    return len(set(s) & set(t)) / k


def jaccard_at_k(s: Sequence[Hashable], t: Sequence[Hashable], k: int) -> float:
    """Jaccard similarity of the two top-k item sets (1.0 if both are empty)."""
    a = set(_dedup(s, k))
    b = set(_dedup(t, k))
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def kendall_tau_topk_distance(
    s: Sequence[Hashable],
    t: Sequence[Hashable],
    k: Optional[int] = None,
    penalty: float = 0.5,
) -> float:
    """Normalised Fagin et al. ``K^(p)`` distance between two top-k lists.

    0 means identical lists and 1 means disjoint lists. Over every pair of
    items in the union of the two lists:

    1. both items in both lists: cost 1 if the lists order them differently;
    2. both in one list, exactly one in the other: cost 1 if the list holding
       both ranks the missing item ahead (the other list implicitly ranks the
       item it contains first);
    3. each item in a different list only: cost 1;
    4. both in one list, neither in the other: cost ``penalty``.
    """
    if not 0.0 <= penalty <= 1.0:
        raise ValueError("penalty must be in [0, 1]")
    s = _dedup(s, k)
    t = _dedup(t, k)
    if not s and not t:
        return 0.0
    rs = {item: r for r, item in enumerate(s)}
    rt = {item: r for r, item in enumerate(t)}
    dist = 0.0
    for i, j in combinations(list(dict.fromkeys(s + t)), 2):
        i_s, j_s, i_t, j_t = i in rs, j in rs, i in rt, j in rt
        if i_s and j_s and i_t and j_t:
            if (rs[i] - rs[j]) * (rt[i] - rt[j]) < 0:
                dist += 1
        elif i_s and j_s and (i_t or j_t):
            present = i if i_t else j
            missing = j if i_t else i
            if rs[missing] < rs[present]:
                dist += 1
        elif i_t and j_t and (i_s or j_s):
            present = i if i_s else j
            missing = j if i_s else i
            if rt[missing] < rt[present]:
                dist += 1
        elif (i_s and j_t and not i_t and not j_s) or (j_s and i_t and not j_t and not i_s):
            dist += 1
        else:  # both in exactly one list, neither in the other
            dist += penalty
    n, m = len(s), len(t)
    worst = n * m + penalty * (n * (n - 1) / 2 + m * (m - 1) / 2)
    if worst == 0:
        return 0.0 if s == t else 1.0
    return dist / worst


def compare_rankings(
    lists_a: Sequence[Sequence[Hashable]],
    lists_b: Sequence[Sequence[Hashable]],
    k: int = 10,
    p: float = 0.9,
    penalty: float = 0.5,
) -> Dict[str, float]:
    """Mean per-user agreement between two models' recommendation lists."""
    if len(lists_a) != len(lists_b):
        raise ValueError("lists_a and lists_b must cover the same users")
    if not lists_a:
        raise ValueError("need at least one user")
    rows = []
    for a, b in zip(lists_a, lists_b):
        r = rank_biased_overlap(a, b, p=p, k=k)
        rows.append(
            (
                r.ext,
                r.min,
                jaccard_at_k(a, b, k),
                overlap_at_k(a, b, k),
                kendall_tau_topk_distance(a, b, k=k, penalty=penalty),
            )
        )
    arr = np.asarray(rows, dtype=float)
    keys = ("RBO_ext", "RBO_min", "Jaccard@K", "Overlap@K", "KendallTopK")
    return {key: float(v) for key, v in zip(keys, arr.mean(axis=0))}


def pairwise_rank_similarity(
    lists_by_model: Mapping[str, Sequence[Sequence[Hashable]]],
    k: int = 10,
    p: float = 0.9,
    metric: str = "RBO_ext",
    penalty: float = 0.5,
) -> pd.DataFrame:
    """Symmetric model-by-model matrix of mean ``metric`` (see :func:`compare_rankings`)."""
    names = list(lists_by_model)
    valid = ("RBO_ext", "RBO_min", "Jaccard@K", "Overlap@K", "KendallTopK")
    if metric not in valid:
        raise ValueError(f"metric must be one of {valid}")
    diag = 0.0 if metric == "KendallTopK" else 1.0
    mat = pd.DataFrame(diag, index=names, columns=names, dtype=float)
    for a, b in combinations(names, 2):
        value = compare_rankings(lists_by_model[a], lists_by_model[b], k=k, p=p, penalty=penalty)[metric]
        mat.loc[a, b] = mat.loc[b, a] = value
    return mat


def render_rank_similarity_section(matrix: pd.DataFrame, k: int, p: float) -> str:
    """Markdown section with the model-by-model RBO matrix and the closest pairs."""
    names = list(matrix.index)
    lines = [
        f"## Ranking similarity between models (RBO@{k}, p={p:g})",
        "",
        "Rank-biased overlap of each pair of models' top-k lists, averaged over "
        "users (1 = identical lists, 0 = disjoint).",
        "",
        "| model | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for a in names:
        lines.append(f"| {a} | " + " | ".join(f"{matrix.loc[a, b]:.3f}" for b in names) + " |")
    pairs = sorted(
        ((matrix.loc[a, b], a, b) for a, b in combinations(names, 2)), reverse=True
    )
    if pairs:
        lines += ["", "Most similar pairs:", ""]
        for value, a, b in pairs[:3]:
            lines.append(f"- {a} vs {b}: {value:.3f}")
    return "\n".join(lines) + "\n"
