"""Inter-user (cross-list) diversity measures.

Per-list measures such as intra-list diversity say nothing about whether
every user gets the *same* list. This module measures how recommendations
differ across users and over time:

* :func:`personalization` -- ``1 - mean pairwise similarity`` between users'
  top-k lists (cosine of binary item vectors, as popularised by
  ``recmetrics``, or Jaccard). 0 means everyone sees the same items, 1 means
  no two users share an item.
* :func:`recommendation_entropy` -- Shannon entropy of how recommendation
  slots are spread over items (aggregate diversity, Adomavicius & Kwon 2012),
  optionally normalised to [0, 1] by ``log(catalog size)``.
* :func:`temporal_diversity` -- mean fraction of a user's current top-k that
  was not in their previous top-k (Lathia et al., SIGIR 2010).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from typing import Optional

import numpy as np

__all__ = ["personalization", "recommendation_entropy", "temporal_diversity"]


def _truncate(lists, k: Optional[int]):
    if k is not None and (int(k) != k or k < 1):
        raise ValueError("k must be a positive integer or None")
    out = []
    for items in lists:
        items = list(dict.fromkeys(items))  # de-duplicate, keep order
        out.append(items[:k] if k is not None else items)
    return out


def personalization(
    recommended_lists: Sequence[Sequence[Hashable]],
    k: Optional[int] = None,
    similarity: str = "cosine",
    max_users: Optional[int] = None,
    random_state: Optional[int] = None,
) -> float:
    """Personalization: ``1 - mean pairwise similarity`` of users' lists.

    Parameters
    ----------
    recommended_lists:
        One ranked list per user.
    k:
        Truncate each list to its top ``k`` items (``None`` keeps all).
    similarity:
        ``"cosine"`` (binary item vectors) or ``"jaccard"``.
    max_users:
        If set and there are more users, a uniform random subset of this size
        is used (the pairwise cost is quadratic in users).
    random_state:
        Seed for the user subsample.
    """
    if similarity not in ("cosine", "jaccard"):
        raise ValueError("similarity must be 'cosine' or 'jaccard'")
    lists = _truncate(recommended_lists, k)
    if max_users is not None:
        if int(max_users) != max_users or max_users < 2:
            raise ValueError("max_users must be an integer >= 2")
        if len(lists) > max_users:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(lists), size=max_users, replace=False)
            lists = [lists[i] for i in idx]
    if len(lists) < 2:
        raise ValueError("personalization needs at least two users")
    if any(len(items) == 0 for items in lists):
        raise ValueError("every recommended list must be non-empty")
    vocab = {item: j for j, item in enumerate({i for items in lists for i in items})}
    M = np.zeros((len(lists), len(vocab)))
    for u, items in enumerate(lists):
        M[u, [vocab[i] for i in items]] = 1.0
    inter = M @ M.T
    sizes = M.sum(axis=1)
    if similarity == "cosine":
        sim = inter / np.sqrt(np.outer(sizes, sizes))
    else:
        sim = inter / (sizes[:, None] + sizes[None, :] - inter)
    iu = np.triu_indices(len(lists), k=1)
    return float(1.0 - sim[iu].mean())


def recommendation_entropy(
    recommended_lists: Sequence[Sequence[Hashable]],
    k: Optional[int] = None,
    catalog=None,
    normalize: bool = False,
) -> float:
    """Shannon entropy (nats) of item frequencies across all recommendation slots.

    With ``normalize=True`` the entropy is divided by ``log(n)``, where ``n`` is
    the catalog size (``catalog`` as a collection or an int) or, when
    ``catalog`` is None, the number of distinct recommended items.
    """
    lists = _truncate(recommended_lists, k)
    counts = Counter(item for items in lists for item in items)
    total = sum(counts.values())
    if total == 0:
        raise ValueError("recommendation_entropy needs at least one recommended item")
    p = np.array(list(counts.values()), dtype=float) / total
    h = float(-(p * np.log(p)).sum())
    if not normalize:
        return h
    if catalog is None:
        n = len(counts)
    else:
        n = catalog if isinstance(catalog, int) else len(set(catalog))
        if n < len(counts):
            raise ValueError("catalog is smaller than the set of recommended items")
    return h / math.log(n) if n > 1 else 0.0


def temporal_diversity(
    previous: Mapping[Hashable, Sequence[Hashable]],
    current: Mapping[Hashable, Sequence[Hashable]],
    k: Optional[int] = None,
) -> float:
    """Mean ``|L_t \\ L_{t-1}| / |L_t|`` over users present in both mappings."""
    users = [u for u in current if u in previous]
    if not users:
        raise ValueError("previous and current share no users")
    values = []
    for u in users:
        (cur,) = _truncate([current[u]], k)
        (prev,) = _truncate([previous[u]], k)
        if not cur:
            continue
        prev_set = set(prev)
        values.append(sum(1 for i in cur if i not in prev_set) / len(cur))
    if not values:
        raise ValueError("all current lists are empty")
    return float(np.mean(values))
