"""Baseline recommenders that give the metrics something to score."""

from __future__ import annotations

import math
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


class ItemKNNRecommender:
    """Item-item k-nearest-neighbor collaborative filtering for implicit feedback.

    Builds a binary user-item matrix (repeated interactions collapse to one)
    and scores a candidate by the sum of its similarities to items in the
    user's history. Similarity is cosine or Jaccard over those binary item
    columns. Only the ``n_neighbors`` strongest neighbors are kept per item;
    ties break by ascending item id. Unknown users or exhausted neighbor
    scores fall back to the popularity ranking so every user still receives
    k items.
    """

    def __init__(self, similarity: str = "cosine", n_neighbors: int = 20) -> None:
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        self.similarity = similarity
        self.n_neighbors = n_neighbors
        self.neighbors_: Optional[dict] = None
        self.histories_: dict = {}
        self.catalog_: list = []
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRecommender":
        _validate(interactions)
        self._fallback.fit(interactions)
        pairs = interactions[["user_id", "item_id"]].drop_duplicates()
        users = sorted(set(pairs["user_id"].tolist()))
        self.catalog_ = sorted(set(pairs["item_id"].tolist()))
        self.histories_ = {
            user: set(items) for user, items in pairs.groupby("user_id")["item_id"]
        }
        self.neighbors_ = {}
        n_users = len(users)
        n_items = len(self.catalog_)
        if n_users == 0 or n_items == 0:
            return self

        user_index = {user: idx for idx, user in enumerate(users)}
        item_index = {item: idx for idx, item in enumerate(self.catalog_)}
        matrix = np.zeros((n_users, n_items), dtype=np.float64)
        for user, item in zip(pairs["user_id"].tolist(), pairs["item_id"].tolist()):
            matrix[user_index[user], item_index[item]] = 1.0

        gram = matrix.T @ matrix
        degrees = np.diag(gram).copy()
        if self.similarity == "cosine":
            norms = np.sqrt(degrees)
            denom = norms[:, None] * norms[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                similarity = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        else:
            union = degrees[:, None] + degrees[None, :] - gram
            with np.errstate(divide="ignore", invalid="ignore"):
                similarity = np.divide(gram, union, out=np.zeros_like(gram), where=union > 0)
        np.fill_diagonal(similarity, 0.0)

        neighbors: dict = {}
        for idx, item in enumerate(self.catalog_):
            positive = np.flatnonzero(similarity[idx] > 0)
            ranked = sorted(
                ((self.catalog_[int(col)], float(similarity[idx, col])) for col in positive),
                key=lambda kv: (-kv[1], kv[0]),
            )[: self.n_neighbors]
            if ranked:
                neighbors[item] = dict(ranked)
        self.neighbors_ = neighbors
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if self.neighbors_ is None:
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude)
        seen = self.histories_.get(user_id, set())
        blocked = banned | seen
        scores: defaultdict = defaultdict(float)
        for history_item in seen:
            for candidate, weight in self.neighbors_.get(history_item, {}).items():
                if candidate not in blocked:
                    scores[candidate] += weight
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        recommendations = [item for item, _ in ranked][:k]
        if len(recommendations) < k:
            remaining = self._fallback.recommend(
                user_id, k, exclude=blocked | set(recommendations)
            )
            recommendations.extend(remaining[: k - len(recommendations)])
        return recommendations


class UserKNNRecommender:
    """User-user k-nearest-neighbor collaborative filtering for implicit feedback.

    Builds a binary user-item matrix (repeated interactions collapse to one)
    and scores a candidate by the sum of similarities to neighbors who
    interacted with it. Similarity is cosine or Jaccard over those binary
    user rows. Only the ``n_neighbors`` strongest neighbors are kept per
    user; ties break by ascending user id. Unknown users or exhausted
    neighbor scores fall back to the popularity ranking so every user still
    receives k items. Seen items are always excluded.
    """

    def __init__(self, similarity: str = "cosine", n_neighbors: int = 20) -> None:
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        n_neighbors = int(n_neighbors)
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        self.similarity = similarity
        self.n_neighbors = n_neighbors
        self.neighbors_: Optional[dict] = None
        self.histories_: dict = {}
        self.catalog_: list = []
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "UserKNNRecommender":
        _validate(interactions)
        self._fallback.fit(interactions)
        pairs = interactions[["user_id", "item_id"]].drop_duplicates()
        users = sorted(set(pairs["user_id"].tolist()))
        self.catalog_ = sorted(set(pairs["item_id"].tolist()))
        self.histories_ = {
            user: set(items) for user, items in pairs.groupby("user_id")["item_id"]
        }
        self.neighbors_ = {}
        n_users = len(users)
        n_items = len(self.catalog_)
        if n_users == 0 or n_items == 0:
            return self

        user_index = {user: idx for idx, user in enumerate(users)}
        item_index = {item: idx for idx, item in enumerate(self.catalog_)}
        matrix = np.zeros((n_users, n_items), dtype=np.float64)
        for user, item in zip(pairs["user_id"].tolist(), pairs["item_id"].tolist()):
            matrix[user_index[user], item_index[item]] = 1.0

        gram = matrix @ matrix.T
        degrees = np.diag(gram).copy()
        if self.similarity == "cosine":
            norms = np.sqrt(degrees)
            denom = norms[:, None] * norms[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                similarity = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        else:
            union = degrees[:, None] + degrees[None, :] - gram
            with np.errstate(divide="ignore", invalid="ignore"):
                similarity = np.divide(gram, union, out=np.zeros_like(gram), where=union > 0)
        np.fill_diagonal(similarity, 0.0)

        neighbors: dict = {}
        for idx, user in enumerate(users):
            positive = np.flatnonzero(similarity[idx] > 0)
            ranked = sorted(
                ((users[int(col)], float(similarity[idx, col])) for col in positive),
                key=lambda kv: (-kv[1], kv[0]),
            )[: self.n_neighbors]
            if ranked:
                neighbors[user] = dict(ranked)
        self.neighbors_ = neighbors
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if self.neighbors_ is None:
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude)
        seen = self.histories_.get(user_id, set())
        blocked = banned | seen
        scores: defaultdict = defaultdict(float)
        for neighbor, weight in self.neighbors_.get(user_id, {}).items():
            for candidate in self.histories_.get(neighbor, ()):
                if candidate not in blocked:
                    scores[candidate] += weight
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        recommendations = [item for item, _ in ranked][:k]
        if len(recommendations) < k:
            remaining = self._fallback.recommend(
                user_id, k, exclude=blocked | set(recommendations)
            )
            recommendations.extend(remaining[: k - len(recommendations)])
        return recommendations


def _bpr_coefficient(score_diff: float) -> float:
    """Return σ(-(x_ui - x_uj)) with overflow protection.

    This is the BPR gradient weight for a sampled (user, positive, negative)
    triple: large positive margins contribute almost nothing, large negative
    margins contribute a weight of 1.
    """
    if score_diff > 35.0:
        return 0.0
    if score_diff < -35.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(score_diff))


class BPRRecommender:
    """Bayesian Personalized Ranking matrix factorization for implicit feedback.

    Learns user and item embeddings (plus item bias) by SGD on sampled
    (user, observed item, unobserved item) triples so that observed items
    rank above unobserved ones. The seed fixes initialization and the
    sampling stream. Unknown users fall back to the popularity ranking so
    every user still receives k items.
    """

    def __init__(
        self,
        n_factors: int = 16,
        n_epochs: int = 30,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        n_negatives: int = 1,
        seed: int = 0,
    ) -> None:
        self.n_factors = int(n_factors)
        self.n_epochs = int(n_epochs)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.n_negatives = int(n_negatives)
        self.seed = int(seed)
        self.catalog_: list = []
        self.histories_: dict = {}
        self.user_factors_: Optional[np.ndarray] = None
        self.item_factors_: Optional[np.ndarray] = None
        self.item_bias_: Optional[np.ndarray] = None
        self._user_index: dict = {}
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "BPRRecommender":
        _validate(interactions)
        self._fallback.fit(interactions)
        pairs = interactions[["user_id", "item_id"]].drop_duplicates()
        users = sorted(set(pairs["user_id"].tolist()))
        self.catalog_ = sorted(set(pairs["item_id"].tolist()))
        self.histories_ = {
            user: set(items) for user, items in pairs.groupby("user_id")["item_id"]
        }
        self._user_index = {user: idx for idx, user in enumerate(users)}
        item_index = {item: idx for idx, item in enumerate(self.catalog_)}

        n_users = len(users)
        n_items = len(self.catalog_)
        if n_users == 0 or n_items == 0:
            self.user_factors_ = np.zeros((0, self.n_factors))
            self.item_factors_ = np.zeros((0, self.n_factors))
            self.item_bias_ = np.zeros(0)
            return self

        rng = np.random.default_rng(self.seed)
        user_factors = rng.normal(0.0, 0.1, size=(n_users, self.n_factors))
        item_factors = rng.normal(0.0, 0.1, size=(n_items, self.n_factors))
        item_bias = np.zeros(n_items, dtype=float)

        seen_mask = np.zeros((n_users, n_items), dtype=bool)
        observed: list = []
        for user, item in zip(pairs["user_id"].tolist(), pairs["item_id"].tolist()):
            u_idx = self._user_index[user]
            i_idx = item_index[item]
            if not seen_mask[u_idx, i_idx]:
                seen_mask[u_idx, i_idx] = True
                observed.append((u_idx, i_idx))
        observed_pairs = np.array(observed, dtype=np.int64)
        seen_counts = seen_mask.sum(axis=1)
        lr = self.learning_rate
        reg = self.regularization

        for _ in range(self.n_epochs):
            rng.shuffle(observed_pairs)
            for u_idx, i_idx in observed_pairs:
                u_idx = int(u_idx)
                i_idx = int(i_idx)
                if int(seen_counts[u_idx]) >= n_items:
                    continue
                for _neg in range(self.n_negatives):
                    j_idx = int(rng.integers(0, n_items))
                    while seen_mask[u_idx, j_idx]:
                        j_idx = int(rng.integers(0, n_items))
                    pu = user_factors[u_idx].copy()
                    qi = item_factors[i_idx].copy()
                    qj = item_factors[j_idx].copy()
                    bi = item_bias[i_idx]
                    bj = item_bias[j_idx]
                    score_diff = float(np.dot(pu, qi - qj) + bi - bj)
                    weight = _bpr_coefficient(score_diff)
                    user_factors[u_idx] += lr * (weight * (qi - qj) - reg * pu)
                    item_factors[i_idx] += lr * (weight * pu - reg * qi)
                    item_factors[j_idx] += lr * (-weight * pu - reg * qj)
                    item_bias[i_idx] += lr * (weight - reg * bi)
                    item_bias[j_idx] += lr * (-weight - reg * bj)

        self.user_factors_ = user_factors
        self.item_factors_ = item_factors
        self.item_bias_ = item_bias
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if self.user_factors_ is None or self.item_factors_ is None or self.item_bias_ is None:
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude) | self.histories_.get(user_id, set())
        user_idx = self._user_index.get(user_id)
        if user_idx is None:
            return self._fallback.recommend(user_id, k, exclude=banned)
        scores = self.user_factors_[user_idx] @ self.item_factors_.T + self.item_bias_
        ranked = sorted(
            (
                (item, score)
                for item, score in zip(self.catalog_, scores)
                if item not in banned
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return [item for item, _ in ranked][:k]