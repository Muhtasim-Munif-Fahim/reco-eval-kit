"""Beyond-accuracy evaluation measures.

These metrics describe recommendation lists themselves rather than (or in
addition to) overlap with held-out items: how much of the catalog is reached,
how uncommon the recommended items are, how varied a single list is, how
unexpected items are relative to popularity or a user's primitive profile,
and how often those unexpected items are also relevant (serendipity).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd


def catalog_coverage(recommended_lists, catalog) -> float:
    """Fraction of distinct catalog items that appear in any list.

    ``catalog`` is either a collection of item ids or an integer catalog
    size. Duplicated recommendations across users count once.
    """
    size = catalog if isinstance(catalog, int) else len(set(catalog))
    if size <= 0:
        raise ValueError("catalog must be non-empty")
    recommended = {item for items in recommended_lists for item in items}
    return len(recommended) / size


def novelty(recommended_lists, item_popularity, floor: float = 1e-6) -> float:
    """Mean self-information ``-log2(p(item))`` over all recommended entries.

    ``item_popularity`` maps item id to its interaction count; probabilities
    are counts normalized over the observed total. Items absent from the
    mapping receive the floor probability, i.e. maximal novelty, so
    long-tail catalog items stay in the average instead of being skipped.
    """
    total = float(sum(item_popularity.values()))
    values = []
    for items in recommended_lists:
        for item in items:
            count = float(item_popularity.get(item, 0))
            if total > 0 and count > 0:
                prob = count / total
            else:
                prob = floor
            values.append(-math.log2(max(prob, floor)))
    if not values:
        raise ValueError("novelty needs at least one recommended item")
    return sum(values) / len(values)


def _cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if norm == 0.0:
        return 1.0
    similarity = float(np.dot(vec_a, vec_b) / norm)
    return 1.0 - min(1.0, max(-1.0, similarity))


def _feature_lookup(item_features: pd.DataFrame) -> dict:
    if not isinstance(item_features, pd.DataFrame) or item_features.empty:
        raise ValueError("item_features must be a non-empty DataFrame")
    matrix = item_features.to_numpy(dtype=float)
    return {item: matrix[pos] for pos, item in enumerate(item_features.index)}


def _lookup_vector(item, lookup: dict) -> np.ndarray:
    try:
        return lookup[item]
    except KeyError as exc:
        raise ValueError(f"item_features is missing item {exc.args[0]!r}") from exc


def _unexp_from_popularity(item, item_popularity, total: float) -> float:
    """Linear unexpectedness ``1 - p(item)`` in ``[0, 1]``.

    Items absent from the popularity mapping, and any run whose observed
    counts sum to zero, are treated as maximally unexpected.
    """
    if total <= 0:
        return 1.0
    count = max(0.0, float(item_popularity.get(item, 0.0)))
    return 1.0 - min(1.0, count / total)


def _profile_mean(history, lookup: dict):
    vectors = [_lookup_vector(item, lookup) for item in history]
    if not vectors:
        return None
    return np.mean(np.stack(vectors, axis=0), axis=0)


def _unexp_from_profile(item, profile, lookup: dict) -> float:
    """Cosine distance from the primitive profile, clipped to ``[0, 1]``.

    An empty history has no primitive expectation, so every item scores 1.0.
    Opposite-direction vectors clip at 1.0 so unexpectedness stays on the
    same scale as the popularity variant.
    """
    if profile is None:
        return 1.0
    return min(1.0, _cosine_distance(_lookup_vector(item, lookup), profile))


def _is_relevant(item, relevant) -> bool:
    if isinstance(relevant, Mapping):
        return float(relevant.get(item, 0.0)) > 0.0
    return item in relevant


def _check_k(k: int) -> int:
    if int(k) != k or k < 1:
        raise ValueError("k must be an integer >= 1")
    return int(k)


def intra_list_diversity(recommended_lists, item_features: pd.DataFrame) -> float:
    """Mean pairwise cosine distance between item feature vectors per list.

    ``item_features`` is a numeric dataframe indexed by item id. Pairs are
    pooled across all lists; lists shorter than two items contribute no
    pairs, and a run without any pair scores 0.0. Items missing from the
    feature table raise :class:`ValueError` because silently assuming zero
    vectors would understate diversity. Zero-length feature vectors are
    treated as maximally distant since they carry no direction.
    """
    lookup = _feature_lookup(item_features)
    distances = []
    for items in recommended_lists:
        for left, right in combinations(items, 2):
            distances.append(
                _cosine_distance(_lookup_vector(left, lookup), _lookup_vector(right, lookup))
            )
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def unexpectedness(recommended_lists, item_popularity) -> float:
    """Mean popularity unexpectedness ``1 - p(item)`` over recommended entries.

    ``item_popularity`` maps item id to its interaction count; probabilities
    are counts normalized over the observed total. Unseen items (and a
    mapping whose counts sum to zero) score 1.0, i.e. fully unexpected.
    Unlike :func:`novelty`, this is a linear ``[0, 1]`` complement of
    popularity rather than self-information in bits.
    """
    total = float(sum(item_popularity.values()))
    values = [
        _unexp_from_popularity(item, item_popularity, total)
        for items in recommended_lists
        for item in items
    ]
    if not values:
        raise ValueError("unexpectedness needs at least one recommended item")
    return sum(values) / len(values)


def profile_unexpectedness(recommended_lists, user_histories, item_features: pd.DataFrame) -> float:
    """Mean cosine distance of recommended items from each user's primitive profile.

    ``user_histories`` must align with ``recommended_lists`` (one history of
    item ids per list). The primitive profile is the mean feature vector of
    that history; an empty history yields unexpectedness 1.0 because there
    is nothing expected. Items missing from ``item_features`` raise
    :class:`ValueError`.
    """
    if len(recommended_lists) != len(user_histories):
        raise ValueError("recommended_lists and user_histories must align per user")
    lookup = _feature_lookup(item_features)
    values = []
    for items, history in zip(recommended_lists, user_histories):
        profile = _profile_mean(history, lookup)
        for item in items:
            values.append(_unexp_from_profile(item, profile, lookup))
    if not values:
        raise ValueError("profile unexpectedness needs at least one recommended item")
    return sum(values) / len(values)


def serendipity(
    recommended_lists,
    relevant_lists,
    item_popularity=None,
    user_histories=None,
    item_features=None,
    k=None,
) -> float:
    """Mean share of top-K items that are both relevant and unexpected.

    Unexpectedness is ``1 - p(item)`` when ``item_popularity`` is given, or
    cosine distance from the primitive user profile when ``user_histories``
    and ``item_features`` are given. Provide exactly one of those two
    sources. Relevance follows the ranking-metric convention: a collection
    of item ids, or a mapping whose positive gains count as relevant.

    Each user's score is the mean of ``relevant(i) * unexpected(i)`` over
    the first ``k`` recommendations (or the full list when ``k`` is omitted).
    When ``k`` is set the denominator is ``k``, matching Precision@K for
    short lists. Users with an empty list and no cutoff score 0.0; an empty
    evaluation set averages to 0.0.
    """
    if len(recommended_lists) != len(relevant_lists):
        raise ValueError("recommended_lists and relevant_lists must align per user")
    if k is not None:
        k = _check_k(k)

    use_popularity = item_popularity is not None
    use_profile = user_histories is not None or item_features is not None
    if use_popularity == use_profile:
        raise ValueError(
            "provide either item_popularity or user_histories with item_features"
        )
    if use_profile and (user_histories is None or item_features is None):
        raise ValueError("primitive-profile unexpectedness needs user_histories and item_features")
    if use_profile and len(user_histories) != len(recommended_lists):
        raise ValueError("recommended_lists and user_histories must align per user")

    if use_popularity:
        total = float(sum(item_popularity.values()))
        lookup = None
        profiles = None
    else:
        total = None
        lookup = _feature_lookup(item_features)
        profiles = [_profile_mean(history, lookup) for history in user_histories]

    scores = []
    for idx, (items, relevant) in enumerate(zip(recommended_lists, relevant_lists)):
        ranked = list(items) if k is None else list(items)[:k]
        if not ranked and k is None:
            scores.append(0.0)
            continue
        denom = len(ranked) if k is None else k
        contrib = 0.0
        for item in ranked:
            if not _is_relevant(item, relevant):
                continue
            if use_popularity:
                contrib += _unexp_from_popularity(item, item_popularity, total)
            else:
                contrib += _unexp_from_profile(item, profiles[idx], lookup)
        scores.append(contrib / denom)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)