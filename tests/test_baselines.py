import numpy as np
import pandas as pd
import pytest

from reco_eval_kit.baselines import (
    BPRRecommender,
    ItemItemCooccurrenceRecommender,
    ItemKNNRecommender,
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


def test_item_knn_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit must be called"):
        ItemKNNRecommender().recommend("u1", k=2)


def test_item_knn_rejects_unknown_similarity_and_empty_neighborhood():
    with pytest.raises(ValueError, match="similarity"):
        ItemKNNRecommender(similarity="pearson")
    with pytest.raises(ValueError, match="n_neighbors"):
        ItemKNNRecommender(n_neighbors=0)


@pytest.mark.parametrize("similarity", ["cosine", "jaccard"])
def test_item_knn_prefers_higher_similarity_over_popularity(similarity):
    # Rare co-occurs with A three times; Pop co-occurs once but is more popular.
    pairs = [("q", 10)]
    for idx in range(3):
        pairs.append((f"r{idx}", 10))
        pairs.append((f"r{idx}", 20))
    pairs.append(("w", 10))
    pairs.append(("w", 2))
    for idx in range(5):
        pairs.append((f"p{idx}", 2))
    model = ItemKNNRecommender(similarity=similarity, n_neighbors=10).fit(make_frame(pairs))
    assert model.recommend("q", k=2) == [20, 2]
    assert 10 not in model.recommend("q", k=10)


def test_item_knn_cosine_and_jaccard_values_and_disagreement():
    cosine = ItemKNNRecommender(similarity="cosine").fit(INTERACTIONS)
    jaccard = ItemKNNRecommender(similarity="jaccard").fit(INTERACTIONS)
    assert cosine.neighbors_[10][11] == pytest.approx(2 / np.sqrt(3 * 2))
    assert jaccard.neighbors_[10][11] == pytest.approx(2 / 3)
    assert cosine.neighbors_[11][10] == pytest.approx(cosine.neighbors_[10][11])

    # |H|=30, |X|=5 with 2 shared, |Y|=25 with 4 shared.
    # Cosine ranks X (1) first; Jaccard ranks Y (2) first.
    pairs = [("t", 100)]
    pairs += [(f"h{i}", 100) for i in range(23)]
    pairs += [(f"sx{i}", 100) for i in range(2)]
    pairs += [(f"sx{i}", 1) for i in range(2)]
    pairs += [(f"ox{i}", 1) for i in range(3)]
    pairs += [(f"sy{i}", 100) for i in range(4)]
    pairs += [(f"sy{i}", 2) for i in range(4)]
    pairs += [(f"oy{i}", 2) for i in range(21)]
    frame = make_frame(pairs)
    assert ItemKNNRecommender(similarity="cosine").fit(frame).recommend("t", k=1) == [1]
    assert ItemKNNRecommender(similarity="jaccard").fit(frame).recommend("t", k=1) == [2]


def test_item_knn_deduplicates_repeated_interactions():
    once = make_frame([("u1", 1), ("u1", 2), ("u2", 1), ("u2", 2)])
    twice = make_frame(
        [("u1", 1), ("u1", 1), ("u1", 2), ("u2", 1), ("u2", 2), ("u2", 2)]
    )
    first = ItemKNNRecommender(similarity="cosine").fit(once)
    second = ItemKNNRecommender(similarity="cosine").fit(twice)
    assert first.neighbors_[1][2] == pytest.approx(second.neighbors_[1][2])
    assert first.neighbors_[1][2] == pytest.approx(1.0)


def test_item_knn_keeps_only_nearest_neighbors_and_breaks_ties_by_item_id():
    model = ItemKNNRecommender(similarity="cosine", n_neighbors=1).fit(INTERACTIONS)
    assert list(model.neighbors_[10]) == [11]
    assert model.recommend("u3", k=2) == [11, 14]

    tied = make_frame(
        [
            ("u", 10),
            ("u", 11),
            ("a", 10),
            ("a", 1),
            ("b", 11),
            ("b", 2),
            ("b", 2),
            ("b", 2),
        ]
    )
    ranked = ItemKNNRecommender(similarity="cosine").fit(tied)
    # Item 2 is more popular, but the similarity scores tie, so item id wins.
    assert ranked.recommend("u", k=2) == [1, 2]


def test_item_knn_unknown_user_falls_back_to_popularity():
    model = ItemKNNRecommender().fit(INTERACTIONS)
    assert model.recommend("nobody", k=2) == [10, 11]


@pytest.mark.parametrize("similarity", ["cosine", "jaccard"])
def test_item_knn_ranks_held_out_cluster_item_above_the_other_cluster(similarity):
    pairs = []
    for user in ("a1", "a2", "a3", "a4"):
        for item in (1, 2, 3, 4):
            pairs.append((user, item))
    for user in ("b1", "b2", "b3", "b4"):
        for item in (10, 11, 12, 13):
            pairs.append((user, item))
    pairs = [(user, item) for user, item in pairs if not (user == "a1" and item == 4)]
    model = ItemKNNRecommender(similarity=similarity, n_neighbors=20).fit(make_frame(pairs))
    recs = model.recommend("a1", k=4, exclude={1, 2, 3})
    assert recs[0] == 4
    assert set(recs).isdisjoint({1, 2, 3})


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