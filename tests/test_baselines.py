import numpy as np
import pandas as pd
import pytest

from reco_eval_kit.baselines import (
    BPRRecommender,
    ItemItemCooccurrenceRecommender,
    PopularityRecommender,
    RandomRecommender,
)


def make_frame(pairs):
    return pd.DataFrame(pairs, columns=["user_id", "item_id"])


INTERACTIONS = make_frame(
    [
        ("u1", 10), ("u1", 11),
        ("u2", 10), ("u2", 11),
        ("u3", 10), ("u3", 12),
        ("u4", 14),
    ]
)


def test_popularity_orders_by_decreasing_count():
    model = PopularityRecommender().fit(INTERACTIONS)
    assert model.recommend(k=10) == [10, 11, 12, 14]


def test_popularity_breaks_count_ties_by_ascending_item_id():
    frame = make_frame([("u1", 7), ("u2", 3)])
    assert PopularityRecommender().fit(frame).recommend(k=2) == [3, 7]


def test_popularity_respects_exclusions_and_cutoff():
    model = PopularityRecommender().fit(INTERACTIONS)
    recs = model.recommend(k=2, exclude={10})
    assert recs == [11, 12]
    assert all(item not in {10} for item in recs)


def test_random_with_same_seed_is_reproducible():
    first = RandomRecommender(seed=11).fit(INTERACTIONS)
    second = RandomRecommender(seed=11).fit(INTERACTIONS)
    for _ in range(3):
        assert first.recommend(k=4) == second.recommend(k=4)


def test_random_with_different_seed_differs():
    first = RandomRecommender(seed=1).fit(INTERACTIONS)
    second = RandomRecommender(seed=2).fit(INTERACTIONS)
    assert first.recommend(k=4) != second.recommend(k=4)


def test_random_never_returns_seen_items():
    model = RandomRecommender(seed=3).fit(INTERACTIONS)
    recs = model.recommend(k=10, exclude={10, 11})
    assert set(recs).isdisjoint({10, 11})


def test_random_returns_every_candidate_when_k_exceeds_catalog():
    model = RandomRecommender(seed=4).fit(INTERACTIONS)
    recs = model.recommend(k=100, exclude={10})
    assert len(recs) == 3 and set(recs) == {11, 12, 14}


def test_cooccurrence_prefers_the_strong_partner():
    model = ItemItemCooccurrenceRecommender().fit(INTERACTIONS)
    recs = model.recommend("u3", k=3, exclude=set())
    # u3 already saw item 10; item 11 co-occurs twice vs item 12 once
    assert recs[0] == 11
    assert 10 not in recs


def test_cooccurrence_unknown_user_falls_back_to_popularity():
    model = ItemItemCooccurrenceRecommender().fit(INTERACTIONS)
    assert model.recommend("nobody", k=2) == [10, 11]


def test_cooccurrence_fills_short_neighbor_scores_from_popularity():
    model = ItemItemCooccurrenceRecommender().fit(INTERACTIONS)
    recs = model.recommend("u4", k=3)
    assert len(recs) == 3
    assert 14 not in recs


def test_fit_rejects_frames_missing_required_columns():
    with pytest.raises(KeyError):
        PopularityRecommender().fit(pd.DataFrame({"item_id": [1]}))


def test_bpr_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit must be called"):
        BPRRecommender().recommend("u1", k=2)


def test_bpr_same_seed_is_reproducible():
    first = BPRRecommender(n_epochs=8, seed=11).fit(INTERACTIONS)
    second = BPRRecommender(n_epochs=8, seed=11).fit(INTERACTIONS)
    assert first.recommend("u1", k=3, exclude={10, 11}) == second.recommend(
        "u1", k=3, exclude={10, 11}
    )


def test_bpr_different_seed_differs():
    first = BPRRecommender(n_epochs=8, seed=1).fit(INTERACTIONS)
    second = BPRRecommender(n_epochs=8, seed=2).fit(INTERACTIONS)
    assert not (
        np.allclose(first.user_factors_, second.user_factors_)
        and np.allclose(first.item_factors_, second.item_factors_)
    )


def test_bpr_never_returns_seen_items_and_respects_cutoff():
    model = BPRRecommender(n_epochs=5, seed=3).fit(INTERACTIONS)
    recs = model.recommend("u1", k=2, exclude={10, 11})
    assert len(recs) == 2
    assert set(recs).isdisjoint({10, 11})
    assert set(model.recommend("u1", k=10)).isdisjoint({10, 11})


def test_bpr_unknown_user_falls_back_to_popularity():
    model = BPRRecommender(n_epochs=5, seed=0).fit(INTERACTIONS)
    assert model.recommend("nobody", k=2) == [10, 11]


def test_bpr_ranks_held_out_cluster_item_above_the_other_cluster():
    pairs = []
    for user in ("a1", "a2", "a3", "a4"):
        for item in (1, 2, 3, 4):
            pairs.append((user, item))
    for user in ("b1", "b2", "b3", "b4"):
        for item in (10, 11, 12, 13):
            pairs.append((user, item))
    pairs = [(user, item) for user, item in pairs if not (user == "a1" and item == 4)]
    model = BPRRecommender(n_factors=8, n_epochs=40, seed=0).fit(make_frame(pairs))
    recs = model.recommend("a1", k=4, exclude={1, 2, 3})
    assert recs[0] == 4