import math

import numpy as np
import pytest

from reco_eval_kit.calibration import (
    average_long_tail_share,
    average_recommendation_popularity,
    calibrated_rerank,
    genre_distribution,
    gini_index,
    kl_miscalibration,
    long_tail_items,
    miscalibration,
)

GENRES = {
    1: ["drama"], 2: ["drama"], 3: ["drama"], 4: ["drama"],
    5: ["comedy"], 6: ["comedy"], 7: ["drama", "comedy"], 8: ["horror"],
}


def test_genre_distribution_splits_multi_genre_items():
    dist = genre_distribution([1, 5, 7], GENRES)
    assert dist == pytest.approx({"drama": 0.5, "comedy": 0.5})
    assert genre_distribution([99], GENRES) == {}
    weighted = genre_distribution([1, 5], GENRES, weights=[3, 1])
    assert weighted == pytest.approx({"drama": 0.75, "comedy": 0.25})
    assert genre_distribution([1], {1: "drama"}) == {"drama": 1.0}


def test_kl_matches_formula_and_is_zero_when_calibrated():
    p = {"drama": 0.7, "comedy": 0.3}
    assert kl_miscalibration(p, p) == pytest.approx(0.0, abs=1e-12)
    q = {"drama": 1.0}
    alpha = 0.01
    expected = 0.7 * math.log(0.7 / (0.99 * 1.0 + 0.01 * 0.7)) + 0.3 * math.log(0.3 / (0.01 * 0.3))
    assert kl_miscalibration(p, q, alpha) == pytest.approx(expected)
    assert kl_miscalibration({}, q) == 0.0


def test_miscalibration_mean_over_users():
    histories = [[1, 2, 5], [5, 6]]
    lists = [[1, 2, 3], [5, 6]]
    value = miscalibration(lists, histories, GENRES)
    first = kl_miscalibration(genre_distribution([1, 2, 5], GENRES), {"drama": 1.0})
    assert value == pytest.approx(first / 2)
    assert miscalibration([[1, 2, 5, 3]], [[1, 2, 5]], GENRES, k=3) == pytest.approx(0.0)


def test_calibrated_rerank_lambda_zero_is_score_order():
    cands = [1, 2, 3, 4, 5, 6]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert calibrated_rerank(cands, scores, [1, 5], GENRES, k=4, lam=0.0) == [1, 2, 3, 4]


def test_calibrated_rerank_improves_calibration():
    history = [1, 2, 3, 5, 6, 8] * 2  # drama 1/2, comedy 1/3, horror 1/6
    cands = [1, 2, 3, 4, 7, 5, 6, 8]
    scores = [1.0, 0.95, 0.9, 0.85, 0.8, 0.5, 0.45, 0.3]
    plain = calibrated_rerank(cands, scores, history, GENRES, k=6, lam=0.0)
    calib = calibrated_rerank(cands, scores, history, GENRES, k=6, lam=0.9)
    p = genre_distribution(history, GENRES)
    kl_plain = kl_miscalibration(p, genre_distribution(plain, GENRES))
    kl_calib = kl_miscalibration(p, genre_distribution(calib, GENRES))
    assert kl_calib < 0.5 * kl_plain
    assert 8 in calib and 8 not in plain
    assert len(calib) == 6 and len(set(calib)) == 6
    # accuracy cost is monotone in lambda
    sc = dict(zip(cands, scores))
    assert sum(sc[i] for i in calib) <= sum(sc[i] for i in plain)


def test_rerank_short_candidate_list_and_errors():
    assert calibrated_rerank([1, 5], [0.2, 0.1], [1], GENRES, k=5) == [1, 5]
    with pytest.raises(ValueError):
        calibrated_rerank([1, 1], [0.1, 0.2], [1], GENRES)
    with pytest.raises(ValueError):
        calibrated_rerank([1], [0.1, 0.2], [1], GENRES)
    with pytest.raises(ValueError):
        calibrated_rerank([1], [0.1], [1], GENRES, lam=1.5)
    with pytest.raises(ValueError):
        calibrated_rerank([1], [0.1], [1], GENRES, k=0)
    with pytest.raises(ValueError):
        kl_miscalibration({"a": 1.0}, {}, alpha=0.0)
    with pytest.raises(ValueError):
        miscalibration([[1]], [], GENRES)


def test_gini_index():
    assert gini_index([[1, 2], [3, 4]]) == pytest.approx(0.0)
    # all exposure on one item of a 4-item catalog -> (n-1)/n
    assert gini_index([[1], [1], [1]], catalog=[1, 2, 3, 4]) == pytest.approx(0.75)
    counts = np.array([1, 2, 3, 10], dtype=float)
    lists = [[i] for i, c in zip([1, 2, 3, 4], counts) for _ in range(int(c))]
    sorted_c = np.sort(counts)
    n = sorted_c.size
    expected = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_c) / (n * sorted_c.sum())
    assert gini_index(lists) == pytest.approx(expected)
    with pytest.raises(ValueError):
        gini_index([[]])


def test_arp_and_long_tail():
    pop = {1: 100, 2: 50, 3: 10, 4: 5, 5: 1}
    assert average_recommendation_popularity([[1, 2], [3]], pop) == pytest.approx((75 + 10) / 2)
    assert long_tail_items(pop, head_fraction=0.4) == {3, 4, 5}
    assert long_tail_items(pop, head_fraction=0.2) == {2, 3, 4, 5}
    share = average_long_tail_share([[1, 2], [3, 99], []], pop, head_fraction=0.4)
    assert share == pytest.approx((0.0 + 1.0) / 2)
    with pytest.raises(ValueError):
        long_tail_items(pop, head_fraction=1.0)
    with pytest.raises(ValueError):
        average_recommendation_popularity([[]], pop)


def test_popularity_recommender_is_more_biased_than_random():
    from reco_eval_kit.baselines import PopularityRecommender, RandomRecommender
    from reco_eval_kit.synthetic import generate_interactions

    interactions = generate_interactions(n_users=40, n_items=60, seed=0)
    pop_model = PopularityRecommender().fit(interactions)
    rnd_model = RandomRecommender(seed=0).fit(interactions)
    users = sorted(interactions["user_id"].unique())[:20]
    pop_lists = [pop_model.recommend(u, k=5) for u in users]
    rnd_lists = [rnd_model.recommend(u, k=5) for u in users]
    popularity = interactions["item_id"].value_counts().to_dict()
    catalog = sorted(interactions["item_id"].unique())
    assert gini_index(pop_lists, catalog) > gini_index(rnd_lists, catalog)
    assert average_recommendation_popularity(pop_lists, popularity) > average_recommendation_popularity(rnd_lists, popularity)
    assert average_long_tail_share(pop_lists, popularity) < average_long_tail_share(rnd_lists, popularity)
