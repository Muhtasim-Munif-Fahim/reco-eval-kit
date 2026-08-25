"""Baseline recommenders that give the metrics something to score."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd


def _validate(interactions: pd.DataFrame) -> None:
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame")
    missing = [c for c in ("user_id", "item_id") if c not in interactions.columns]
    if missing:
        raise KeyError(f"interactions is missing required columns: {missing}")


class PopularityRecommender:
    """Ranks items by interaction count, breaking ties by ascending item id."""

    def __init__(self) -> None:
        self.popularity_order_: list = []

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        _validate(interactions)
        counts = interactions["item_id"].value_counts()
        self.popularity_order_ = sorted(
            counts.index.tolist(), key=lambda item: (-int(counts[item]), item)
        )
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        banned = set(exclude)
        return [item for item in self.popularity_order_ if item not in banned][:k]


class RandomRecommender:
    """Uniform random recommendations from the fitted catalog.

    The seed fixes the sampling stream, so two instances fitted on the same
    catalog produce identical rankings call after call.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.catalog_: list = []
        self._rng: Optional[np.random.Generator] = None

    def fit(self, interactions: pd.DataFrame) -> "RandomRecommender":
        _validate(interactions)
        self.catalog_ = sorted(set(interactions["item_id"].tolist()))
        self._rng = np.random.default_rng(self.seed)
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if self._rng is None:
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude)
        candidates = [item for item in self.catalog_ if item not in banned]
        if not candidates:
            return []
        picks = self._rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        return [candidates[int(pos)] for pos in picks]


class ItemItemCooccurrenceRecommender:
    """Scores candidates by raw co-occurrence counts with the user's history.

    Two items co-occur once per shared user (per-user item sets are
    deduplicated first). Ties break by ascending item id; unknown users or
    exhausted neighbor scores fall back to the popularity ranking so every
    user still receives k items.
    """

    def __init__(self) -> None:
        self.neighbors_: dict = {}
        self.histories_: dict = {}
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "ItemItemCooccurrenceRecommender":
        _validate(interactions)
        pair_counts: defaultdict = defaultdict(int)
        histories = interactions.groupby("user_id")["item_id"].agg(lambda s: sorted(set(s)))
        for items in histories:
            for left, right in combinations(items, 2):
                pair_counts[(left, right)] += 1
        neighbors: defaultdict = defaultdict(dict)
        for (left, right), count in pair_counts.items():
            neighbors[left][right] = count
            neighbors[right][left] = count
        self.neighbors_ = dict(neighbors)
        self.histories_ = {user: set(items) for user, items in histories.items()}
        self._fallback.fit(interactions)
        return self

    def recommend(self, user_id, k: int = 10, exclude=()) -> list:
        banned = set(exclude)
        seen = self.histories_.get(user_id, set())
        blocked = banned | seen
        scores: defaultdict = defaultdict(int)
        for history_item in seen:
            for candidate, count in self.neighbors_.get(history_item, {}).items():
                if candidate not in blocked:
                    scores[candidate] += count
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        recommendations = [item for item, _ in ranked][:k]
        if len(recommendations) < k:
            remaining = self._fallback.recommend(user_id, k, exclude=blocked | set(recommendations))
            recommendations.extend(remaining[: k - len(recommendations)])
        return recommendations