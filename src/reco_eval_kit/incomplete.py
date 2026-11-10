"""Metrics that stay robust when relevance judgments are incomplete.

Offline recommender test sets only label a small part of the catalog. A
standard metric counts every *unlabelled* item as non-relevant, so a new model
that surfaces good but never-shown items gets penalised. These metrics take
judgments as a mapping ``item -> gain``: ``gain > 0`` is relevant, ``gain == 0``
is *judged* non-relevant, and items missing from the mapping are *unjudged*.

* :func:`judged_at_k` -- fraction of the top-k that carries a judgment (report
  it next to any metric to see how much of the list was actually assessed).
* :func:`bpref` -- Buckley & Voorhees (SIGIR 2004) binary preference, computed
  from judged items only, following the ``trec_eval`` convention::

      bpref = (1/R) * sum_{relevant r retrieved} (1 - min(n_r, R) / min(R, N))

  where ``R``/``N`` are the numbers of judged relevant / non-relevant items
  and ``n_r`` counts judged non-relevant items ranked above ``r``.
* :func:`condense` plus :func:`condensed_ndcg_at_k` /
  :func:`condensed_ap_at_k` -- Sakai's (SIGIR 2007) condensed-list metrics:
  drop unjudged items from the ranking, then score it as usual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .metrics import _check_k, _validate_recommended, average_precision_at_k, ndcg_at_k

__all__ = [
    "bpref",
    "condense",
    "condensed_ap_at_k",
    "condensed_ndcg_at_k",
    "judged_at_k",
]


def _judgments(judged) -> dict:
    if isinstance(judged, Mapping):
        out = {}
        for item, gain in judged.items():
            gain = float(gain)
            if gain < 0:
                raise ValueError("relevance gains must be non-negative")
            out[item] = gain
        return out
    if isinstance(judged, (str, bytes)):
        raise TypeError("judgments must be a mapping item -> gain")
    raise TypeError("judgments must be a mapping item -> gain (0 for judged non-relevant)")


def judged_at_k(recommended: Sequence, judged, k: int) -> float:
    """Fraction of the first ``k`` positions occupied by judged items.

    ``judged`` may be a judgment mapping or any collection of judged item ids.
    The denominator is ``k`` (short lists count missing slots as unjudged).
    """
    _check_k(k)
    items = _validate_recommended(recommended)
    keys = judged.keys() if isinstance(judged, Mapping) else set(judged)
    return sum(1 for item in items[:k] if item in keys) / k


def bpref(recommended: Sequence, judgments: Mapping) -> float:
    """Binary preference (bpref) over judged items; unjudged items are ignored.

    Raises ``ValueError`` when there is no judged relevant item.
    """
    items = _validate_recommended(recommended)
    qrels = _judgments(judgments)
    R = sum(1 for g in qrels.values() if g > 0)
    N = len(qrels) - R
    if R == 0:
        raise ValueError("judgments contain no relevant item")
    denom = min(R, N)
    nonrel_above = 0
    total = 0.0
    for item in items:
        if item not in qrels:
            continue
        if qrels[item] > 0:
            if nonrel_above > 0 and denom > 0:
                total += 1.0 - min(nonrel_above, R) / denom
            else:
                total += 1.0
        else:
            nonrel_above += 1
    return total / R


def condense(recommended: Sequence, judgments) -> list:
    """Return the ranking with unjudged items removed (order preserved)."""
    items = _validate_recommended(recommended)
    keys = judgments.keys() if isinstance(judgments, Mapping) else set(judgments)
    return [item for item in items if item in keys]


def condensed_ndcg_at_k(recommended: Sequence, judgments: Mapping, k: int) -> float:
    """nDCG@k on the condensed list (unjudged items removed first)."""
    qrels = _judgments(judgments)
    return ndcg_at_k(condense(recommended, qrels), qrels, k)


def condensed_ap_at_k(recommended: Sequence, judgments: Mapping, k: int) -> float:
    """AP@k on the condensed list (unjudged items removed first)."""
    qrels = _judgments(judgments)
    relevant = {item: g for item, g in qrels.items() if g > 0}
    if not relevant:
        raise ValueError("judgments contain no relevant item")
    return average_precision_at_k(condense(recommended, qrels), relevant, k)
