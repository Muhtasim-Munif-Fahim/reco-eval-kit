import math

import pytest

from reco_eval_kit.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    inverse_popularity_at_k,
    mean_inverse_popularity_at_k,
    mean_metric,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_counts_hits_over_cutoff():
    assert precision_at_k([1, 2, 3, 4], {2, 4}, k=4) == pytest.approx(0.5)


def test_precision_keeps_k_denominator_beyond_list_length():
    assert precision_at_k([1, 2], {1}, k=5) == pytest.approx(0.2)


def test_recall_normalizes_by_all_relevant_items():
    assert recall_at_k([1], {1, 2}, k=3) == pytest.approx(0.5)
    assert recall_at_k([2, 1], {1, 2}, k=2) == pytest.approx(1.0)


def test_hit_rate_flags_any_overlap():
    assert hit_rate_at_k([7, 8, 9], {9}, k=3) == 1.0
    assert hit_rate_at_k([7, 8, 10], {9}, k=3) == 0.0


def test_average_precision_known_example():
    # hits at ranks 2 and 4: P@2=0.5, P@4=0.5; two relevant items
    assert average_precision_at_k([1, 2, 3, 4], {2, 4}, k=4) == pytest.approx(0.5)


def test_average_precision_caps_normalizer_at_k():
    # one relevant item found out of two total, but only k=1 positions used
    assert average_precision_at_k([1], {1, 9}, k=1) == pytest.approx(1.0)


def test_average_penalizes_late_first_hit():
    assert average_precision_at_k([5, 6, 7, 1], {1}, k=4) == pytest.approx(1 / 4)


def test_mrr_scores_mid_list_hit():
    assert mrr([10, 20, 30, 40], {30}) == pytest.approx(1 / 3)


def test_mrr_zero_when_no_relevant_recommended():
    assert mrr([10, 20, 30], {40}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)


def test_ndcg_binary_partial_ordering():
    rec = [1, 2, 3, 4]
    rel = {2, 4}
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1.0 + 1 / math.log2(3)
    assert ndcg_at_k(rec, rel, k=4) == pytest.approx(dcg / idcg)


def test_ndcg_supports_graded_gains():
    rec = [1, 2, 3]
    rel = {1: 3, 2: 2, 4: 1}
    dcg = 3.0 + 2 / math.log2(3)
    idcg = 3.0 + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(rec, rel, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_covers_relevant_items_outside_the_list():
    # only the weakest item is recommended, so the score is gain/ideal-gain
    assert ndcg_at_k([3], {1: 3, 2: 2, 3: 1}, k=1) == pytest.approx(1 / 3)


def test_ndcg_returns_zero_without_positive_gains():
    assert ndcg_at_k([1, 2], {}, k=2) == 0.0


def test_duplicate_recommendations_are_rejected():
    with pytest.raises(ValueError):
        precision_at_k([1, 1, 2], {2}, k=3)


def test_invalid_cutoff_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, k=0)


def test_mean_metric_averages_per_user_scores():
    recs = [[1, 2], [3, 4]]
    rels = [{1}, {9}]
    expected = (precision_at_k(recs[0], rels[0], k=2) + precision_at_k(recs[1], rels[1], k=2)) / 2
    assert mean_metric(precision_at_k, recs, rels, k=2) == pytest.approx(expected)


def test_mean_metric_handles_empty_evaluation_set():
    assert mean_metric(mrr, [], []) == 0.0


def test_recall_counts_graded_items_with_positive_gain():
    rel = {1: 0.0, 2: 2.0}
    assert recall_at_k([1], rel, k=1) == pytest.approx(0.0)
    assert recall_at_k([2], rel, k=1) == pytest.approx(1.0)


def test_inverse_popularity_rewards_rare_items_at_higher_ranks():
    popularity = {1: 0, 2: 3}
    rare_first = inverse_popularity_at_k([1, 2], popularity, k=2)
    popular_first = inverse_popularity_at_k([2, 1], popularity, k=2)
    disc1 = 1.0
    disc2 = 1.0 / math.log2(3)
    denom = disc1 + disc2
    assert rare_first == pytest.approx((1.0 * disc1 + 0.25 * disc2) / denom)
    assert popular_first == pytest.approx((0.25 * disc1 + 1.0 * disc2) / denom)
    assert rare_first > popular_first


def test_inverse_popularity_of_uniform_full_list_is_reciprocal_count():
    popularity = {1: 4, 2: 4, 3: 4}
    assert inverse_popularity_at_k([1, 2, 3], popularity, k=3) == pytest.approx(1 / 5)


def test_unseen_items_have_maximal_inverse_popularity():
    assert inverse_popularity_at_k([99], {1: 10}, k=1) == pytest.approx(1.0)
    assert inverse_popularity_at_k([1], {1: 0}, k=1) == pytest.approx(1.0)
    assert inverse_popularity_at_k([1], {1: 99}, k=1) == pytest.approx(0.01)


def test_inverse_popularity_penalizes_lists_shorter_than_k():
    score = inverse_popularity_at_k([7], {}, k=2)
    denom = 1.0 + 1.0 / math.log2(3)
    assert score == pytest.approx(1.0 / denom)


def test_inverse_popularity_rejects_bad_inputs():
    with pytest.raises(ValueError):
        inverse_popularity_at_k([1, 1], {1: 1}, k=2)
    with pytest.raises(ValueError):
        inverse_popularity_at_k([1], {1: 1}, k=0)
    with pytest.raises(ValueError):
        inverse_popularity_at_k([1], {1: -1}, k=1)
    with pytest.raises(ValueError):
        inverse_popularity_at_k([1], {1: math.nan}, k=1)
    with pytest.raises(TypeError):
        inverse_popularity_at_k([1], [1], k=1)


def test_mean_inverse_popularity_averages_users():
    popularity = {1: 1}
    lists = [[1], [9]]
    assert mean_inverse_popularity_at_k(lists, popularity, k=1) == pytest.approx(0.75)


def test_mean_inverse_popularity_empty_evaluation_is_zero():
    assert mean_inverse_popularity_at_k([], {1: 1}, k=1) == 0.0