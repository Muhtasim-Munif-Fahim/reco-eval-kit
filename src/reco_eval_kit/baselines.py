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


class PureSVDRecommender:
    """Truncated SVD matrix factorization for implicit feedback (PureSVD).

    Builds a binary user-item matrix (repeated interactions collapse to
    one) and factors it with a compact SVD ``R ≈ U Σ Vᵀ``. Users are
    scored by reconstructing ``u Σ Vᵀ`` against every item. Unlike BPR,
    there is no sampling loop: the ranking is the truncated reconstruction
    of the observed matrix (Cremonesi, Koren & Turrin, 2010). Unknown
    users fall back to the popularity ranking so every user still receives
    k items. Score ties break by ascending item id.
    """

    def __init__(self, n_factors: int = 16) -> None:
        n_factors = int(n_factors)
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        self.n_factors = n_factors
        self.catalog_: list = []
        self.histories_: dict = {}
        self.user_factors_: Optional[np.ndarray] = None
        self.item_factors_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self._user_index: dict = {}
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "PureSVDRecommender":
        _validate(interactions)
        self._fallback.fit(interactions)
        pairs = interactions[["user_id", "item_id"]].drop_duplicates()
        users = sorted(set(pairs["user_id"].tolist()))
        self.catalog_ = sorted(set(pairs["item_id"].tolist()))
        self.histories_ = {
            user: set(items) for user, items in pairs.groupby("user_id")["item_id"]
        }
        self._user_index = {user: idx for idx, user in enumerate(users)}
        n_users = len(users)
        n_items = len(self.catalog_)
        if n_users == 0 or n_items == 0:
            self.user_factors_ = np.zeros((0, 0))
            self.item_factors_ = np.zeros((0, 0))
            self.singular_values_ = np.zeros(0)
            return self

        item_index = {item: idx for idx, item in enumerate(self.catalog_)}
        matrix = np.zeros((n_users, n_items), dtype=np.float64)
        for user, item in zip(pairs["user_id"].tolist(), pairs["item_id"].tolist()):
            matrix[self._user_index[user], item_index[item]] = 1.0

        rank = min(self.n_factors, n_users, n_items)
        # Compact SVD; full_matrices=False keeps U (n_users, r) and Vt (r, n_items).
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        u = u[:, :rank]
        s = s[:rank]
        vt = vt[:rank]
        self.user_factors_ = u
        self.item_factors_ = vt.T
        self.singular_values_ = s
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if (
            self.user_factors_ is None
            or self.item_factors_ is None
            or self.singular_values_ is None
        ):
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude) | self.histories_.get(user_id, set())
        user_idx = self._user_index.get(user_id)
        if user_idx is None:
            return self._fallback.recommend(user_id, k, exclude=banned)
        # Reconstruct row: (u_i * Σ) @ Vᵀ
        scores = (self.user_factors_[user_idx] * self.singular_values_) @ self.item_factors_.T
        ranked = sorted(
            (
                (item, score)
                for item, score in zip(self.catalog_, scores)
                if item not in banned
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return [item for item, _ in ranked][:k]


class WRMFRecommender:
    """Weighted Regularized Matrix Factorization for implicit feedback (WRMF / ALS).

    Follows Hu, Koren & Volinsky (2008). Observations become binary preferences
    ``p_ui ∈ {0, 1}`` with confidence ``c_ui = 1 + alpha * r_ui`` (``r_ui`` is
    the interaction count). User and item factors alternate closed-form ridge
    solves until ``n_epochs`` passes complete. Unknown users fall back to the
    popularity ranking. Score ties break by ascending item id.
    """

    def __init__(
        self,
        n_factors: int = 16,
        n_epochs: int = 10,
        alpha: float = 40.0,
        regularization: float = 0.1,
        seed: int = 0,
    ) -> None:
        n_factors = int(n_factors)
        n_epochs = int(n_epochs)
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        if n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        if float(regularization) < 0.0:
            raise ValueError("regularization must be non-negative")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        self.seed = int(seed)
        self.catalog_: list = []
        self.histories_: dict = {}
        self.user_factors_: Optional[np.ndarray] = None
        self.item_factors_: Optional[np.ndarray] = None
        self._user_index: dict = {}
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "WRMFRecommender":
        _validate(interactions)
        self._fallback.fit(interactions)
        pairs = interactions[["user_id", "item_id"]].copy()
        users = sorted(set(pairs["user_id"].tolist()))
        self.catalog_ = sorted(set(pairs["item_id"].tolist()))
        # Binary histories for exclusion; counts feed confidence.
        dedup = pairs.drop_duplicates()
        self.histories_ = {
            user: set(items) for user, items in dedup.groupby("user_id")["item_id"]
        }
        self._user_index = {user: idx for idx, user in enumerate(users)}
        n_users = len(users)
        n_items = len(self.catalog_)
        if n_users == 0 or n_items == 0:
            self.user_factors_ = np.zeros((0, self.n_factors))
            self.item_factors_ = np.zeros((0, self.n_factors))
            return self

        item_index = {item: idx for idx, item in enumerate(self.catalog_)}
        counts = np.zeros((n_users, n_items), dtype=np.float64)
        for user, item in zip(pairs["user_id"].tolist(), pairs["item_id"].tolist()):
            counts[self._user_index[user], item_index[item]] += 1.0
        preference = (counts > 0.0).astype(np.float64)
        confidence = 1.0 + self.alpha * counts

        rng = np.random.default_rng(self.seed)
        X = rng.normal(0.0, 0.1, size=(n_users, self.n_factors))
        Y = rng.normal(0.0, 0.1, size=(n_items, self.n_factors))
        eye = np.eye(self.n_factors)
        reg = self.regularization

        for _ in range(self.n_epochs):
            YtY = Y.T @ Y
            for u in range(n_users):
                c_u = confidence[u]
                p_u = preference[u]
                # Cu is diagonal; use the Cu - I trick: Yt Y + Yt (Cu-I) Y + λI
                Cu_minus_I = c_u - 1.0
                A = YtY + (Y.T * Cu_minus_I) @ Y + reg * eye
                b = (Y.T * (c_u * p_u)) @ np.ones(n_items)
                X[u] = np.linalg.solve(A, b)
            XtX = X.T @ X
            for i in range(n_items):
                c_i = confidence[:, i]
                p_i = preference[:, i]
                Cu_minus_I = c_i - 1.0
                A = XtX + (X.T * Cu_minus_I) @ X + reg * eye
                b = (X.T * (c_i * p_i)) @ np.ones(n_users)
                Y[i] = np.linalg.solve(A, b)

        self.user_factors_ = X
        self.item_factors_ = Y
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude) | self.histories_.get(user_id, set())
        user_idx = self._user_index.get(user_id)
        if user_idx is None:
            return self._fallback.recommend(user_id, k, exclude=banned)
        scores = self.user_factors_[user_idx] @ self.item_factors_.T
        ranked = sorted(
            (
                (item, score)
                for item, score in zip(self.catalog_, scores)
                if item not in banned
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return [item for item, _ in ranked][:k]



class EASERecommender:
    """Embarrassingly Shallow Autoencoders for Sparse Data (Steck, 2019).

    Builds a binary user–item matrix ``X`` and the item–item Gram
    ``G = XᵀX``. The closed-form item–item weight matrix is

        ``P = (G + λ I)^{-1}``
        ``B = I - P · diag(1 / diag(P))``

    which forces ``diag(B) = 0`` so an item never reconstructs itself.
    Users are scored by ``x_u @ B``. Unknown users fall back to the
    popularity ranking. Score ties break by ascending item id.
    """

    def __init__(self, l2: float = 500.0) -> None:
        l2 = float(l2)
        if not (l2 > 0.0):
            raise ValueError("l2 must be positive")
        self.l2 = l2
        self.catalog_: list = []
        self.histories_: dict = {}
        self.item_weights_: Optional[np.ndarray] = None
        self._user_index: dict = {}
        self._user_rows: Optional[np.ndarray] = None
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "EASERecommender":
        _validate(interactions)
        self._fallback.fit(interactions)
        pairs = interactions[["user_id", "item_id"]].drop_duplicates()
        users = sorted(set(pairs["user_id"].tolist()))
        self.catalog_ = sorted(set(pairs["item_id"].tolist()))
        self.histories_ = {
            user: set(items) for user, items in pairs.groupby("user_id")["item_id"]
        }
        self._user_index = {user: idx for idx, user in enumerate(users)}
        n_users = len(users)
        n_items = len(self.catalog_)
        if n_users == 0 or n_items == 0:
            self.item_weights_ = np.zeros((0, 0))
            self._user_rows = np.zeros((0, 0))
            return self

        item_index = {item: idx for idx, item in enumerate(self.catalog_)}
        X = np.zeros((n_users, n_items), dtype=np.float64)
        for user, item in zip(pairs["user_id"].tolist(), pairs["item_id"].tolist()):
            X[self._user_index[user], item_index[item]] = 1.0

        G = X.T @ X
        # P = (G + λ I)^{-1}
        G = G + self.l2 * np.eye(n_items)
        try:
            P = np.linalg.inv(G)
        except np.linalg.LinAlgError as exc:
            raise ValueError("EASE Gram matrix is singular; increase l2") from exc
        diag = np.diag(P).copy()
        if np.any(np.abs(diag) < 1e-18):
            raise ValueError("EASE inverse has a near-zero diagonal; increase l2")
        # B = I - P @ diag(1/diag(P))  ⇒  zero diagonal, B_ij = -P_ij / P_ii
        B = -P / diag[np.newaxis, :]
        np.fill_diagonal(B, 0.0)
        self.item_weights_ = B
        self._user_rows = X
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if self.item_weights_ is None or self._user_rows is None:
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude) | self.histories_.get(user_id, set())
        user_idx = self._user_index.get(user_id)
        if user_idx is None:
            return self._fallback.recommend(user_id, k, exclude=banned)
        scores = self._user_rows[user_idx] @ self.item_weights_
        ranked = sorted(
            (
                (item, score)
                for item, score in zip(self.catalog_, scores)
                if item not in banned
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return [item for item, _ in ranked][:k]



class SlimRecommender:
    """Sparse Linear Methods for top-N recommendation (Ning & Karypis, 2011).

    Learns an item–item weight matrix ``W`` by solving, for each item ``j``,
    an elastic-net / coordinate-descent regression of column ``j`` on all
    other columns of the binary user–item matrix ``X``, with ``W_jj = 0``
    so an item never reconstructs itself. Users are scored by ``x_u @ W``.
    Unknown users fall back to the popularity ranking. Score ties break by
    ascending item id.
    """

    def __init__(
        self,
        l1_reg: float = 0.1,
        l2_reg: float = 0.1,
        n_iter: int = 20,
        tol: float = 1e-4,
        non_negative: bool = True,
    ) -> None:
        l1_reg = float(l1_reg)
        l2_reg = float(l2_reg)
        n_iter = int(n_iter)
        tol = float(tol)
        if l1_reg < 0.0:
            raise ValueError("l1_reg must be non-negative")
        if l2_reg < 0.0:
            raise ValueError("l2_reg must be non-negative")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1")
        if not (tol >= 0.0):
            raise ValueError("tol must be non-negative")
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.n_iter = n_iter
        self.tol = tol
        self.non_negative = bool(non_negative)
        self.catalog_: list = []
        self.histories_: dict = {}
        self.item_weights_: Optional[np.ndarray] = None
        self._user_index: dict = {}
        self._user_rows: Optional[np.ndarray] = None
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "SlimRecommender":
        _validate(interactions)
        self._fallback.fit(interactions)
        pairs = interactions[["user_id", "item_id"]].drop_duplicates()
        users = sorted(set(pairs["user_id"].tolist()))
        self.catalog_ = sorted(set(pairs["item_id"].tolist()))
        self.histories_ = {
            user: set(items) for user, items in pairs.groupby("user_id")["item_id"]
        }
        self._user_index = {user: idx for idx, user in enumerate(users)}
        n_users = len(users)
        n_items = len(self.catalog_)
        if n_users == 0 or n_items == 0:
            self.item_weights_ = np.zeros((0, 0))
            self._user_rows = np.zeros((0, 0))
            return self

        item_index = {item: idx for idx, item in enumerate(self.catalog_)}
        X = np.zeros((n_users, n_items), dtype=np.float64)
        for user, item in zip(pairs["user_id"].tolist(), pairs["item_id"].tolist()):
            X[self._user_index[user], item_index[item]] = 1.0

        W = _slim_coordinate_descent(
            X,
            l1_reg=self.l1_reg,
            l2_reg=self.l2_reg,
            n_iter=self.n_iter,
            tol=self.tol,
            non_negative=self.non_negative,
        )
        self.item_weights_ = W
        self._user_rows = X
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if self.item_weights_ is None or self._user_rows is None:
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude) | self.histories_.get(user_id, set())
        user_idx = self._user_index.get(user_id)
        if user_idx is None:
            return self._fallback.recommend(user_id, k, exclude=banned)
        scores = self._user_rows[user_idx] @ self.item_weights_
        ranked = sorted(
            (
                (item, score)
                for item, score in zip(self.catalog_, scores)
                if item not in banned
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return [item for item, _ in ranked][:k]


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def _slim_coordinate_descent(
    X: np.ndarray,
    l1_reg: float,
    l2_reg: float,
    n_iter: int,
    tol: float,
    non_negative: bool,
) -> np.ndarray:
    """Per-item elastic-net coordinate descent with a zero diagonal.

    For each column ``j`` solve
    ``min_w ||X w - X[:, j]||^2 + l1 ||w||1 + l2 ||w||^2`` subject to
    ``w_j = 0`` (and optionally ``w >= 0``).
    """
    n_users, n_items = X.shape
    # Gram matrix G = XᵀX; column norms on the diagonal.
    G = X.T @ X
    W = np.zeros((n_items, n_items), dtype=np.float64)
    for j in range(n_items):
        # Residual starts as the target column (w = 0).
        # Coordinate updates use the soft-thresholded Gram row.
        w = np.zeros(n_items, dtype=np.float64)
        # Working residual r = X[:, j] - X @ w  (initially X[:, j])
        # For efficiency, keep Xt_r = X.T @ r = G[:, j] - G @ w
        xt_r = G[:, j].copy()
        for _ in range(n_iter):
            max_delta = 0.0
            for k in range(n_items):
                if k == j:
                    continue
                g_kk = float(G[k, k])
                if g_kk <= 0.0:
                    continue
                # Coordinate update: soft-threshold (xt_r_k + g_kk * w_k)
                # then divide by (g_kk + l2).
                rho = float(xt_r[k]) + g_kk * float(w[k])
                new_wk = _soft_threshold(rho, l1_reg) / (g_kk + l2_reg)
                if non_negative and new_wk < 0.0:
                    new_wk = 0.0
                delta = new_wk - float(w[k])
                if delta != 0.0:
                    # xt_r -= delta * G[:, k]
                    xt_r -= delta * G[:, k]
                    w[k] = new_wk
                    max_delta = max(max_delta, abs(delta))
            if max_delta < tol:
                break
        w[j] = 0.0
        W[:, j] = w
    np.fill_diagonal(W, 0.0)
    return W



class NMFRecommender:
    """Non-negative matrix factorization recommender for implicit feedback.

    Factors a binary user–item matrix ``X ≈ W @ H`` with Lee–Seung
    multiplicative updates (Frobenius objective). Users are scored by
    ``W[u] @ H``. Unknown users fall back to the popularity ranking.
    Score ties break by ascending item id.
    """

    def __init__(
        self,
        n_factors: int = 16,
        n_epochs: int = 50,
        seed: int = 0,
        eps: float = 1e-9,
    ) -> None:
        n_factors = int(n_factors)
        n_epochs = int(n_epochs)
        seed = int(seed)
        eps = float(eps)
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        if n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        if not (eps > 0.0):
            raise ValueError("eps must be positive")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.seed = seed
        self.eps = eps
        self.catalog_: list = []
        self.histories_: dict = {}
        self.user_factors_: Optional[np.ndarray] = None
        self.item_factors_: Optional[np.ndarray] = None
        self._user_index: dict = {}
        self._fallback = PopularityRecommender()

    def fit(self, interactions: pd.DataFrame) -> "NMFRecommender":
        _validate(interactions)
        self._fallback.fit(interactions)
        pairs = interactions[["user_id", "item_id"]].drop_duplicates()
        users = sorted(set(pairs["user_id"].tolist()))
        self.catalog_ = sorted(set(pairs["item_id"].tolist()))
        self.histories_ = {
            user: set(items) for user, items in pairs.groupby("user_id")["item_id"]
        }
        self._user_index = {user: idx for idx, user in enumerate(users)}
        n_users = len(users)
        n_items = len(self.catalog_)
        k = self.n_factors
        if n_users == 0 or n_items == 0:
            self.user_factors_ = np.zeros((0, k))
            self.item_factors_ = np.zeros((k, 0))
            return self

        item_index = {item: idx for idx, item in enumerate(self.catalog_)}
        X = np.zeros((n_users, n_items), dtype=np.float64)
        for user, item in zip(pairs["user_id"].tolist(), pairs["item_id"].tolist()):
            X[self._user_index[user], item_index[item]] = 1.0

        rng = np.random.default_rng(self.seed)
        # Positive random init scaled by data mean.
        scale = max(float(X.mean()), self.eps)
        W = rng.random((n_users, k)) * scale + self.eps
        H = rng.random((k, n_items)) * scale + self.eps
        for _ in range(self.n_epochs):
            # H <- H * (W.T @ X) / (W.T @ W @ H)
            numer_h = W.T @ X
            denom_h = (W.T @ W) @ H + self.eps
            H *= numer_h / denom_h
            # W <- W * (X @ H.T) / (W @ H @ H.T)
            numer_w = X @ H.T
            denom_w = W @ (H @ H.T) + self.eps
            W *= numer_w / denom_w
        self.user_factors_ = W
        self.item_factors_ = H
        return self

    def recommend(self, user_id=None, k: int = 10, exclude=()) -> list:
        if self.user_factors_ is None or self.item_factors_ is None:
            raise RuntimeError("fit must be called before recommend")
        banned = set(exclude) | self.histories_.get(user_id, set())
        user_idx = self._user_index.get(user_id)
        if user_idx is None:
            return self._fallback.recommend(user_id, k, exclude=banned)
        scores = self.user_factors_[user_idx] @ self.item_factors_
        ranked = sorted(
            (
                (item, score)
                for item, score in zip(self.catalog_, scores)
                if item not in banned
            ),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return [item for item, _ in ranked][:k]
