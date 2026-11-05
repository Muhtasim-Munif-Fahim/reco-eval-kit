import math

import pytest

from reco_eval_kit.subtopic import (
    alpha_dcg_at_k,
    alpha_ndcg_at_k,
    ideal_alpha_dcg_at_k,
    mean_alpha_ndcg_at_k,
    mean_subtopic_recall_at_k,
    subtopic_recall_at_k,
)

# Two "action" sequels and one comedy.
J = {"a1": {"action"}, "a2": {"action"}, "c1": {"comedy"}}


def test_alpha_dcg_hand_computed():
    # ranks: a1 gain 1, a2 gain 0.5 (action seen once), c1 gain 1
    expected = 1 / math.log2(2) + 0.5 / math.log2(3) + 1 / math.log2(4)
    assert alpha_dcg_at_k(["a1", "a2", "c1"], J, 3, alpha=0.5) == pytest.approx(expected)


def test_diverse_ranking_beats_redundant_one():
    redundant = alpha_ndcg_at_k(["a1", "a2", "c1"], J, 3)
    diverse = alpha_ndcg_at_k(["a1", "c1", "a2"], J, 3)
    assert diverse == pytest.approx(1.0)
    assert redundant < diverse
    # plain NDCG-style (alpha=0) can't tell them apart
    assert alpha_ndcg_at_k(["a1", "a2", "c1"], J, 3, alpha=0.0) == pytest.approx(1.0)


def test_ideal_is_greedy_and_multi_subtopic_items():
    j = {"x": {"s1", "s2"}, "y": {"s1"}, "z": {"s3"}}
    # greedy: x (gain 2), then z (gain 1) > y (0.5)
    expected = 2 / math.log2(2) + 1 / math.log2(3) + 0.5 / math.log2(4)
    assert ideal_alpha_dcg_at_k(j, 3, 0.5) == pytest.approx(expected)
    assert alpha_ndcg_at_k(["x", "z", "y"], j, 3) == pytest.approx(1.0)


def test_irrelevant_and_unjudged_items_and_cutoff():
    assert alpha_ndcg_at_k(["n1", "n2"], J, 2) == 0.0
    assert alpha_ndcg_at_k(["n1", "a1", "c1"], J, 1) == 0.0
    assert 0 < alpha_ndcg_at_k(["n1", "a1", "c1"], J, 3) < 1
    assert math.isnan(alpha_ndcg_at_k(["a1"], {}, 3))
    assert math.isnan(alpha_ndcg_at_k(["a1"], {"a1": set()}, 3))


def test_string_subtopic_treated_as_single_label():
    assert alpha_dcg_at_k(["a"], {"a": "action"}, 1) == pytest.approx(1.0)


def test_subtopic_recall():
    assert subtopic_recall_at_k(["a1", "a2"], J, 2) == pytest.approx(0.5)
    assert subtopic_recall_at_k(["a1", "c1"], J, 2) == pytest.approx(1.0)
    assert subtopic_recall_at_k(["c1", "a1"], J, 1) == pytest.approx(0.5)
    assert math.isnan(subtopic_recall_at_k(["a1"], {}, 1))


def test_means_skip_users_without_judgments():
    lists = [["a1", "c1", "a2"], ["a1", "a2"], ["z"]]
    judgments = [J, J, {}]
    m = mean_alpha_ndcg_at_k(lists, judgments, 3)
    assert m == pytest.approx((1.0 + alpha_ndcg_at_k(["a1", "a2"], J, 3)) / 2)
    assert mean_subtopic_recall_at_k(lists, judgments, 2) == pytest.approx((1.0 + 0.5) / 2)
    assert math.isnan(mean_alpha_ndcg_at_k([["a"]], [{}], 3))
    with pytest.raises(ValueError, match="same length"):
        mean_alpha_ndcg_at_k(lists, judgments[:2], 3)


def test_validation():
    with pytest.raises(ValueError, match="k must"):
        alpha_ndcg_at_k(["a1"], J, 0)
    with pytest.raises(ValueError, match="alpha"):
        alpha_ndcg_at_k(["a1"], J, 3, alpha=1.0)
    with pytest.raises(ValueError, match="duplicate"):
        alpha_ndcg_at_k(["a1", "a1"], J, 3)
    with pytest.raises(TypeError):
        alpha_ndcg_at_k(["a1"], [("a1", "x")], 3)
