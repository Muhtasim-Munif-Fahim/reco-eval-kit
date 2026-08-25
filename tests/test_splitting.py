import pandas as pd
import pytest

from reco_eval_kit.splitting import leave_last_n, leave_one_out, threshold_relevant


def make_frame(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])


FRAME = make_frame(
    [
        ("u1", "b", 2),
        ("u2", "y", 5),
        ("u1", "a", 1),
        ("u3", "z", 9),
        ("u1", "c", 3),
        ("u2", "x", 1),
    ]
)


def test_leave_one_out_holds_out_latest_interaction_per_user():
    train, test = leave_one_out(FRAME)
    assert set(zip(test["user_id"], test["item_id"])) == {("u1", "c"), ("u2", "y")}
    assert len(train) == 3


def test_leave_one_out_sorts_by_timestamp_not_row_order():
    train, _ = leave_one_out(FRAME)
    held_for_u1 = set(train.loc[train["user_id"] == "u1", "item_id"])
    assert held_for_u1 == {"a", "b"}


def test_leave_one_out_drops_users_below_min_interactions():
    _, test = leave_one_out(FRAME, min_interactions=3)
    assert set(test["user_id"]) == {"u1"}


def test_leave_one_out_requires_min_interactions_of_two():
    with pytest.raises(ValueError):
        leave_one_out(FRAME, min_interactions=1)


def test_leave_last_n_holds_out_n_most_recent_rows():
    frame = make_frame(
        [
            ("u1", "a", 1),
            ("u1", "b", 2),
            ("u1", "c", 3),
            ("u1", "d", 4),
            ("u2", "p", 1),
            ("u2", "q", 2),
            ("u2", "r", 3),
        ]
    )
    train, test = leave_last_n(frame, n=2)
    assert set(test["item_id"]) == {"c", "d", "q", "r"}
    assert set(train["item_id"]) == {"a", "b", "p"}


def test_leave_last_n_drops_users_without_training_history():
    frame = make_frame([("u1", "a", 1), ("u1", "b", 2), ("u1", "c", 3), ("u2", "x", 1), ("u2", "y", 2)])
    train, test = leave_last_n(frame, n=2)
    assert set(test["user_id"]) == {"u1"}
    assert set(test["item_id"]) == {"b", "c"}
    assert set(train["item_id"]) == {"a"}


def test_leave_last_n_rejects_non_positive_n():
    with pytest.raises(ValueError):
        leave_last_n(FRAME, n=0)


def test_threshold_relevant_keeps_only_meeting_rows():
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": [10, 11, 12],
            "rating": [2.0, 4.0, 5.0],
        }
    )
    kept = threshold_relevant(frame, threshold=4.0)
    assert list(kept["item_id"]) == [11, 12]


def test_threshold_relevant_rejects_missing_rating_column():
    with pytest.raises(KeyError):
        threshold_relevant(make_frame([("u1", "a", 1)]))


def test_splitter_rejects_missing_timestamp_column():
    with pytest.raises(KeyError):
        leave_one_out(pd.DataFrame({"user_id": [1], "item_id": [1]}))