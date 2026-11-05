"""Novelty- and diversity-aware ranking metrics over subtopics (aspects).

Standard NDCG rewards a list of ten relevant items even when all ten cover the
same need (ten sequels of one film). The metrics here take, for each user or
query, a mapping ``item -> set of relevant subtopics`` (genres, intents,
aspects) and reward covering *different* subtopics early.

* :func:`alpha_ndcg_at_k` -- alpha-nDCG (Clarke et al., "Novelty and
  Diversity in Information Retrieval Evaluation", SIGIR 2008). The gain of
  the item at rank ``r`` is ``sum_s J(item, s) * (1 - alpha) ** c_s``, where
  ``c_s`` counts the earlier items already covering subtopic ``s``. Gains are
  discounted by ``1 / log2(r + 1)`` and normalized by a greedily built ideal
  list (exact ideal ordering is NP-hard; the greedy ideal is the standard
  choice and is what the reference ``ndeval`` tool uses). ``alpha = 0`` is
  NDCG over subtopic counts; ``alpha -> 1`` rewards only new subtopics.
* :func:`subtopic_recall_at_k` -- S-recall (Zhai, Cohen & Lafferty, 2003):
  the fraction of the user's relevant subtopics covered in the top ``k``.
* :func:`mean_alpha_ndcg_at_k` / :func:`mean_subtopic_recall_at_k` -- the
  per-user means over parallel lists of rankings and judgments.

Users with no relevant subtopics are undefined for both metrics and are
skipped by the means (``nan`` for the per-user functions).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "alpha_dcg_at_k",
    "alpha_ndcg_at_k",
    "ideal_alpha_dcg_at_k",
    "mean_alpha_ndcg_at_k",
    "mean_subtopic_recall_at_k",
    "subtopic_recall_at_k",
]


def _check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return k


def _check_alpha(alpha) -> float:
    alpha = float(alpha)
    if not 0.0 <= alpha < 1.0 or math.isnan(alpha):
        raise ValueError("alpha must be in [0, 1)")
    return alpha


def _judgments(item_subtopics: Mapping) -> dict:
    if not isinstance(item_subtopics, Mapping):
        raise TypeError("item_subtopics must map item -> iterable of subtopics")
    out = {}
    for item, subs in item_subtopics.items():
        if isinstance(subs, (str, bytes)):
            subs = [subs]
        subs = frozenset(subs)
        if subs:
            out[item] = subs
    return out


def _check_ranking(recommended: Sequence) -> list:
    ranking = list(recommended)
    if len(set(ranking)) != len(ranking):
        raise ValueError("recommended list contains duplicate items")
    return ranking


def _gain(subs, counts: dict, alpha: float) -> float:
    return sum((1.0 - alpha) ** counts.get(s, 0) for s in subs)


def _alpha_dcg(ranking, judged: dict, k: int, alpha: float) -> float:
    counts: dict = {}
    total = 0.0
    for rank, item in enumerate(ranking[:k], start=1):
        subs = judged.get(item)
        if not subs:
            continue
        total += _gain(subs, counts, alpha) / math.log2(rank + 1)
        for s in subs:
            counts[s] = counts.get(s, 0) + 1
    return total


def alpha_dcg_at_k(recommended: Sequence, item_subtopics: Mapping, k: int, alpha: float = 0.5) -> float:
    """Unnormalized alpha-DCG@k of one ranked list."""
    k, alpha = _check_k(k), _check_alpha(alpha)
    return _alpha_dcg(_check_ranking(recommended), _judgments(item_subtopics), k, alpha)


def ideal_alpha_dcg_at_k(item_subtopics: Mapping, k: int, alpha: float = 0.5) -> float:
    """alpha-DCG@k of the greedy ideal ranking of all judged items.

    At each rank the item with the largest marginal gain is appended; ties
    are broken by the judgment order, so the result is deterministic.
    """
    k, alpha = _check_k(k), _check_alpha(alpha)
    remaining = list(_judgments(item_subtopics).items())
    counts: dict = {}
    total = 0.0
    for rank in range(1, k + 1):
        if not remaining:
            break
        best_idx, best_gain = 0, -1.0
        for idx, (_, subs) in enumerate(remaining):
            g = _gain(subs, counts, alpha)
            if g > best_gain:
                best_idx, best_gain = idx, g
        _, subs = remaining.pop(best_idx)
        total += best_gain / math.log2(rank + 1)
        for s in subs:
            counts[s] = counts.get(s, 0) + 1
    return total


def alpha_ndcg_at_k(recommended: Sequence, item_subtopics: Mapping, k: int, alpha: float = 0.5) -> float:
    """alpha-nDCG@k of one ranked list (``nan`` when nothing is relevant).

    ``item_subtopics`` maps each relevant item to the subtopics it is
    relevant to *for this user / query*. Items absent from the mapping have
    zero gain. Because the ideal is built greedily the score can, in rare
    cases, slightly exceed 1; it is clipped to 1.
    """
    k, alpha = _check_k(k), _check_alpha(alpha)
    judged = _judgments(item_subtopics)
    ideal = ideal_alpha_dcg_at_k(judged, k, alpha)
    if ideal <= 0.0:
        return float("nan")
    return min(1.0, _alpha_dcg(_check_ranking(recommended), judged, k, alpha) / ideal)


def subtopic_recall_at_k(recommended: Sequence, item_subtopics: Mapping, k: int) -> float:
    """Fraction of relevant subtopics covered by the top ``k`` (S-recall@k)."""
    k = _check_k(k)
    judged = _judgments(item_subtopics)
    universe = frozenset().union(*judged.values()) if judged else frozenset()
    if not universe:
        return float("nan")
    covered = set()
    for item in _check_ranking(recommended)[:k]:
        covered |= judged.get(item, frozenset())
    return len(covered) / len(universe)


def _mean(values) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _paired(recommended_lists, subtopic_judgments):
    lists, judgments = list(recommended_lists), list(subtopic_judgments)
    if len(lists) != len(judgments):
        raise ValueError("recommended_lists and subtopic_judgments must have the same length")
    return zip(lists, judgments)


def mean_alpha_ndcg_at_k(recommended_lists, subtopic_judgments, k: int, alpha: float = 0.5) -> float:
    """Mean alpha-nDCG@k over users, skipping users with no judgments."""
    return _mean(alpha_ndcg_at_k(r, j, k, alpha) for r, j in _paired(recommended_lists, subtopic_judgments))


def mean_subtopic_recall_at_k(recommended_lists, subtopic_judgments, k: int) -> float:
    """Mean S-recall@k over users, skipping users with no judgments."""
    return _mean(subtopic_recall_at_k(r, j, k) for r, j in _paired(recommended_lists, subtopic_judgments))
