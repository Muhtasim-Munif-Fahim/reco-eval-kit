"""Diversity-aware re-ranking: MMR and greedy DPP MAP inference.

Accuracy-optimized top-K lists are often near-duplicates (five sequels of
the same film). Both re-rankers take a recommender's candidate list with
relevance scores and item feature vectors, and build a top-``k`` list that
trades a little relevance for less redundancy. Similarity between items is
cosine similarity of their feature vectors (any precomputed similarity
matrix can be passed instead).

* :func:`mmr_rerank` -- Maximal Marginal Relevance (Carbonell & Goldstein,
  SIGIR 1998). Each step appends the candidate maximizing
  ``lam * rel(i) - (1 - lam) * max_{j in S} sim(i, j)``.
  ``lam = 1`` keeps the score order; smaller ``lam`` penalizes items similar
  to anything already chosen.
* :func:`dpp_rerank` -- determinantal point process with kernel
  ``L = diag(q) S diag(q)``, ``q_i = exp(alpha * rel(i))`` and
  ``alpha = theta / (2 (1 - theta))``, whose log-determinant equals
  ``theta / (1 - theta) * sum rel + log det S_Y`` up to a constant
  (Chen, Zhang & Zhou, "Fast Greedy MAP Inference for Determinantal Point
  Process to Improve Recommendation Diversity", NeurIPS 2018). Greedy MAP
  inference uses their incremental Cholesky updates, ``O(k^2 N)`` overall.
  Unlike MMR's max-similarity penalty, ``log det S_Y`` rewards the *volume*
  spanned by the whole list.

Scores are min-max normalized to ``[0, 1]`` by default so ``lam`` / ``theta``
mean the same thing for any recommender.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["cosine_similarity_matrix", "mmr_rerank", "dpp_rerank", "rank_scores"]


def rank_scores(n: int) -> list[float]:
    """Relevance proxy for a ranked list without scores: ``1 - r / n``."""
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError("n must be a non-negative integer")
    return [1.0 - r / n for r in range(int(n))]


def cosine_similarity_matrix(items: Sequence, item_features: pd.DataFrame) -> np.ndarray:
    """Pairwise cosine similarity of ``items`` from rows of ``item_features``.

    Items with an all-zero feature vector are similar only to themselves.
    """
    if not isinstance(item_features, pd.DataFrame) or item_features.empty:
        raise ValueError("item_features must be a non-empty DataFrame")
    missing = [item for item in items if item not in item_features.index]
    if missing:
        raise ValueError(f"item_features is missing item {missing[0]!r}")
    X = item_features.loc[list(items)].to_numpy(dtype=float)
    norms = np.linalg.norm(X, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    U = X / safe[:, None]
    S = np.clip(U @ U.T, -1.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S


def _prepare(candidates, scores, k, item_features, similarity, normalize):
    candidates = list(candidates)
    rel = np.asarray([float(s) for s in scores], dtype=float)
    if len(candidates) != rel.size:
        raise ValueError("candidates and scores must have the same length")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidates must be unique")
    if not np.all(np.isfinite(rel)):
        raise ValueError("scores must be finite")
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    if (item_features is None) == (similarity is None):
        raise ValueError("provide exactly one of item_features or similarity")
    if similarity is not None:
        S = np.asarray(similarity, dtype=float)
        if S.shape != (len(candidates), len(candidates)):
            raise ValueError("similarity must be (n_candidates, n_candidates)")
        if not np.allclose(S, S.T):
            raise ValueError("similarity must be symmetric")
    else:
        S = cosine_similarity_matrix(candidates, item_features)
    if normalize and rel.size:
        lo, hi = rel.min(), rel.max()
        rel = (rel - lo) / (hi - lo) if hi > lo else np.ones_like(rel)
    return candidates, rel, S, min(int(k), len(candidates))


def mmr_rerank(
    candidates: Sequence,
    scores: Sequence[float],
    k: int = 10,
    lam: float = 0.7,
    *,
    item_features: pd.DataFrame | None = None,
    similarity=None,
    normalize: bool = True,
) -> list:
    """Maximal Marginal Relevance top-``k`` re-ranking.

    Greedily appends ``argmax lam * rel(i) - (1 - lam) * max_{j in S} sim(i, j)``
    (the first pick is simply the most relevant item). Ties break toward the
    earlier candidate. Pass ``item_features`` (rows indexed by item id) or a
    precomputed symmetric ``similarity`` matrix aligned with ``candidates``.
    """
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    candidates, rel, S, k = _prepare(candidates, scores, k, item_features, similarity, normalize)
    n = len(candidates)
    if n == 0:
        return []
    max_sim = np.full(n, -np.inf)
    available = np.ones(n, dtype=bool)
    chosen: list[int] = []
    for _ in range(k):
        penalty = np.where(np.isfinite(max_sim), max_sim, 0.0)
        value = lam * rel - (1.0 - lam) * penalty
        value[~available] = -np.inf
        best = int(np.argmax(value))
        chosen.append(best)
        available[best] = False
        max_sim = np.maximum(max_sim, S[:, best])
    return [candidates[i] for i in chosen]


def dpp_rerank(
    candidates: Sequence,
    scores: Sequence[float],
    k: int = 10,
    theta: float = 0.7,
    *,
    item_features: pd.DataFrame | None = None,
    similarity=None,
    normalize: bool = True,
    epsilon: float = 1e-10,
) -> list:
    """Greedy DPP MAP re-ranking with fast incremental Cholesky updates.

    ``theta`` in ``[0, 1)`` weights relevance against diversity: the kernel
    is ``L = diag(q) S diag(q)`` with ``q = exp(theta / (2 (1 - theta)) * rel)``,
    so ``theta = 0`` maximizes ``log det S_Y`` alone and ``theta -> 1``
    approaches the score order. Each step adds the item with the largest
    marginal gain in ``log det L_Y`` (Chen et al., 2018, Algorithm 1). When
    no remaining item adds volume above ``epsilon`` (e.g. exact duplicates
    of chosen items), the list is completed in relevance order so ``k``
    items are always returned when available.
    """
    theta = float(theta)
    if not 0.0 <= theta < 1.0:
        raise ValueError("theta must be in [0, 1)")
    candidates, rel, S, k = _prepare(candidates, scores, k, item_features, similarity, normalize)
    n = len(candidates)
    if n == 0:
        return []
    alpha = theta / (2.0 * (1.0 - theta))
    q = np.exp(alpha * (rel - rel.max()))  # shift for stability; scales L by a constant
    L = q[:, None] * S * q[None, :]
    C = np.zeros((k, n))
    d2 = np.diag(L).copy()
    available = np.ones(n, dtype=bool)
    chosen: list[int] = []
    for t in range(k):
        gains = np.where(available, d2, -np.inf)
        j = int(np.argmax(gains))
        if not gains[j] > epsilon:
            break
        chosen.append(j)
        available[j] = False
        if t + 1 == k:
            break
        dj = math.sqrt(d2[j])
        e = (L[j, :] - C[:t, j] @ C[:t, :]) / dj
        C[t, :] = e
        d2 = d2 - e * e
    if len(chosen) < k:
        for i in np.argsort(-rel, kind="stable"):
            if available[i]:
                chosen.append(int(i))
                available[i] = False
                if len(chosen) == k:
                    break
    return [candidates[i] for i in chosen]
