import numpy as np
import pandas as pd
import pytest

from reco_eval_kit.synthetic import generate_interactions, generate_item_features


def test_generator_returns_requested_shape_and_columns():
    frame = generate_interactions(n_users=20, n_items=30, n_interactions=500, seed=1)
    assert list(frame.columns) == ["user_id", "item_id", "timestamp"]
    assert len(frame) == 500


def test_same_seed_reproduces_identical_streams():
    first = generate_interactions(n_users=15, n_items=25, n_interactions=400, seed=3)
    second = generate_interactions(n_users=15, n_items=25, n_interactions=400, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_the_stream():
    first = generate_interactions(n_users=15, n_items=25, n_interactions=400, seed=3)
    second = generate_interactions(n_users=15, n_items=25, n_interactions=400, seed=4)
    assert not first["item_id"].equals(second["item_id"])


def test_ids_stay_within_requested_bounds():
    frame = generate_interactions(n_users=10, n_items=40, n_interactions=800, seed=5)
    assert frame["user_id"].between(0, 9).all()
    assert frame["item_id"].between(0, 39).all()


def test_timestamps_form_a_sequential_stream():
    frame = generate_interactions(n_users=8, n_items=20, n_interactions=300, seed=6)
    assert np.array_equal(frame["timestamp"].to_numpy(), np.arange(300))


def test_popularity_exponent_creates_head_concentration():
    frame = generate_interactions(
        n_users=50, n_items=60, n_interactions=4000, seed=8, popularity_exponent=1.2
    )
    counts = frame["item_id"].value_counts()
    assert counts.iloc[0] > counts.median() * 5


def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError):
        generate_interactions(n_users=0)


def test_item_features_have_expected_shape_and_index():
    features = generate_item_features(n_items=12, n_features=5, seed=2)
    assert features.shape == (12, 5)
    assert list(features.index) == list(range(12))