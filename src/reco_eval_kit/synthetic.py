"""Seeded generators for synthetic implicit-feedback data."""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ("user_id", "item_id", "timestamp")


def generate_interactions(
    n_users: int = 100,
    n_items: int = 200,
    n_interactions: int = 5000,
    seed: int = 42,
    popularity_exponent: float = 1.0,
    cluster_mix: float = 0.7,
    n_clusters: int = 4,
) -> pd.DataFrame:
    """Sample a chronological interaction stream with learnable structure.

    Every user belongs to a taste cluster. An interaction's item is drawn
    either from the cluster-favored slice of a Zipf-style global popularity
    distribution or from the global distribution itself (probability
    ``cluster_mix``), so popularity, co-occurrence, ItemKNN, and BPR baselines
    find real signal. Timestamps are stream positions, which makes last-item
    holdout splits well defined.
    """
    if n_users < 1 or n_items < 1 or n_interactions < 1:
        raise ValueError("n_users, n_items and n_interactions must be positive")
    if not 0.0 <= cluster_mix <= 1.0:
        raise ValueError("cluster_mix must lie in [0, 1]")
    if popularity_exponent < 0:
        raise ValueError("popularity_exponent must be non-negative")
    rng = np.random.default_rng(seed)
    n_clusters = max(1, min(int(n_clusters), n_users))
    clusters = rng.integers(0, n_clusters, size=n_users)

    weights = 1.0 / np.arange(1, n_items + 1, dtype=float) ** popularity_exponent
    weights /= weights.sum()

    users = rng.integers(0, n_users, size=n_interactions).astype(np.int64)
    items = rng.choice(n_items, size=n_interactions, p=weights)
    use_cluster = rng.random(size=n_interactions) < cluster_mix

    block = max(1, n_items // n_clusters)
    user_cluster = clusters[users]
    for cluster in range(n_clusters):
        start = cluster * block
        end = n_items if cluster == n_clusters - 1 else start + block
        local = weights[start:end] / weights[start:end].sum()
        mask = use_cluster & (user_cluster == cluster)
        picks = int(mask.sum())
        if picks:
            items[mask] = rng.choice(np.arange(start, end), size=picks, p=local)

    frame = pd.DataFrame(
        {
            "user_id": users,
            "item_id": items.astype(np.int64),
            "timestamp": np.arange(n_interactions, dtype=np.int64),
        }
    )
    return frame[list(COLUMNS)]


def generate_item_features(
    n_items: int = 200,
    n_features: int = 16,
    seed: int = 7,
    noise_scale: float = 0.35,
) -> pd.DataFrame:
    """Seeded item feature matrix for diversity-aware evaluation.

    The first half of the dimensions is shared structure plus moderate noise,
    the rest pure noise, giving intra-list diversity scores that sit strictly
    between the degenerate extremes.
    """
    if n_items < 1 or n_features < 1:
        raise ValueError("n_items and n_features must be positive")
    rng = np.random.default_rng(seed)
    structured = max(1, n_features // 2)
    base = rng.normal(size=(n_items, structured))
    noisy = base + rng.normal(scale=noise_scale, size=(n_items, structured))
    filler = rng.normal(size=(n_items, n_features - structured))
    matrix = np.hstack([noisy, filler])
    columns = [f"f{position}" for position in range(matrix.shape[1])]
    return pd.DataFrame(matrix, index=pd.RangeIndex(n_items), columns=columns)