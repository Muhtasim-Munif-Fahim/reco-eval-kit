"""Beyond-accuracy evaluation measures.

These metrics describe recommendation lists themselves rather than overlap
with held-out items: how much of the catalog is reached, how uncommon the
recommended items are, and how varied a single list is.
"""

from __future__ import annotations

import math
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


def intra_list_diversity(recommended_lists, item_features: pd.DataFrame) -> float:
    """Mean pairwise cosine distance between item feature vectors per list.

    ``item_features`` is a numeric dataframe indexed by item id. Pairs are
    pooled across all lists; lists shorter than two items contribute no
    pairs, and a run without any pair scores 0.0. Items missing from the
    feature table raise :class:`ValueError` because silently assuming zero
    vectors would understate diversity. Zero-length feature vectors are
    treated as maximally distant since they carry no direction.
    """
    if not isinstance(item_features, pd.DataFrame) or item_features.empty:
        raise ValueError("item_features must be a non-empty DataFrame")
    matrix = item_features.to_numpy(dtype=float)
    lookup = {item: matrix[pos] for pos, item in enumerate(item_features.index)}
    distances = []
    for items in recommended_lists:
        for left, right in combinations(items, 2):
            try:
                vec_left = lookup[left]
                vec_right = lookup[right]
            except KeyError as exc:
                raise ValueError(f"item_features is missing item {exc.args[0]!r}") from exc
            distances.append(_cosine_distance(vec_left, vec_right))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)