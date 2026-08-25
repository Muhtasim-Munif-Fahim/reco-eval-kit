"""Train/test splitting protocols for implicit-feedback evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS = ("user_id", "item_id")


def _validate(interactions: pd.DataFrame, require: Iterable[str]) -> None:
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame")
    missing = [column for column in require if column not in interactions.columns]
    if missing:
        raise KeyError(f"interactions is missing required columns: {missing}")


def _order_chronologically(interactions, timestamp_col):
    """Stable sort by user then timestamp; equal timestamps keep input order."""
    return interactions.sort_values(["user_id", timestamp_col], kind="mergesort")


def leave_one_out(
    interactions: pd.DataFrame,
    timestamp_col: str = "timestamp",
    min_interactions: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent interaction as the test row.

    Users with fewer than ``min_interactions`` interactions are dropped so
    every retained user keeps at least one training row. Returns
    ``(train_df, test_df)`` with reset indices.
    """
    require = (*REQUIRED_COLUMNS, timestamp_col)
    _validate(interactions, require)
    if min_interactions < 2:
        raise ValueError("min_interactions must be at least 2")
    ordered = _order_chronologically(interactions, timestamp_col)
    sizes = ordered.groupby("user_id")["item_id"].transform("size")
    kept = ordered[sizes >= min_interactions]
    test_index = kept.groupby("user_id").tail(1).index
    train = kept.drop(index=test_index).reset_index(drop=True)
    test = kept.loc[test_index].reset_index(drop=True)
    return train, test


def leave_last_n(
    interactions: pd.DataFrame,
    n: int,
    timestamp_col: str = "timestamp",
    min_train: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's last ``n`` interactions as the test set.

    Users with fewer than ``n + min_train`` interactions are dropped so the
    train side never starves. Returns ``(train_df, test_df)``.
    """
    require = (*REQUIRED_COLUMNS, timestamp_col)
    _validate(interactions, require)
    if n < 1:
        raise ValueError("n must be at least 1")
    ordered = _order_chronologically(interactions, timestamp_col)
    sizes = ordered.groupby("user_id")["item_id"].transform("size")
    kept = ordered[sizes >= n + min_train]
    test_index = kept.groupby("user_id").tail(n).index
    train = kept.drop(index=test_index).reset_index(drop=True)
    test = kept.loc[test_index].reset_index(drop=True)
    return train, test


def threshold_relevant(
    interactions: pd.DataFrame,
    rating_col: str = "rating",
    threshold: float = 4.0,
) -> pd.DataFrame:
    """Keep only interactions whose rating meets the threshold."""
    _validate(interactions, (*REQUIRED_COLUMNS, rating_col))
    mask = interactions[rating_col] >= threshold
    return interactions.loc[mask].reset_index(drop=True)