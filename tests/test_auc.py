import random

import pytest

from reco_eval_kit.auc import f1_at_k, gauc, list_auc, r_precision, user_auc
from reco_eval_kit.metrics import mean_metric

sklearn_metrics = pytest.importorskip("sklearn.metrics")


def test_user_auc_perfect_and_reversed():
    scores = {"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.0}
    assert user_auc(scores, {"a", "b"}) == pytest.approx(1.0)
    assert user_auc(scores, {"c", "d"}) == pytest.approx(0.0)


@pytest.mark.parametrize("seed", range(10))
def test_user_auc_matches_sklearn_with_ties(seed):
    rng = random.Random(seed)
    items = list(range(40))
    scores = {i: round(rng.random(), 1) for i in items}  # coarse -> many ties
    positives = {i for i in items if rng.random() < 0.3} or {0}
    y = [1 if i in positives else 0 for i in items]
    expected = sklearn_metrics.roc_auc_score(y, [scores[i] for i in items])
    assert user_auc(scores, positives) == pytest.approx(expected)


def test_user_auc_undefined_and_nan():
    with pytest.raises(ValueError):
        user_auc({"a": 1.0, "b": 0.5}, {"a", "b"})
    with pytest.raises(ValueError):
        user_auc({"a": 1.0, "b": 0.5}, set())
    with pytest.raises(ValueError):
        user_auc({"a": float("nan"), "b": 0.5}, {"a"})


def test_positives_outside_candidates_ignored():
    assert user_auc({"a": 1.0, "b": 0.0}, {"a", "zzz"}) == pytest.approx(1.0)


def test_list_auc_unrecommended_ties():
    catalog = range(6)
    # relevant 0 at rank 1, relevant 5 never recommended.
    auc = list_auc([0, 1, 2], {0, 5}, catalog)
    # negatives: 1,2 (ranked) and 3,4 (unranked). item 0 beats all 4,
    # item 5 loses to 1,2 and ties 3,4 -> (4 + 1) / 8.
    assert auc == pytest.approx(5 / 8)
    with pytest.raises(ValueError):
        list_auc([0, 99], {0}, catalog)


def test_gauc_weighting_and_skips():
    user_scores = {
        "u1": {"a": 0.9, "b": 0.1, "c": 0.2, "d": 0.3},  # AUC 1.0, 4 impressions
        "u2": {"a": 0.1, "b": 0.9},  # AUC 0.0, 2 impressions
        "u3": {"a": 0.5, "b": 0.4},  # only negatives -> skipped
    }
    positives = {"u1": {"a"}, "u2": {"a"}}
    res = gauc(user_scores, positives)
    assert res.gauc == pytest.approx(4 / 6)
    assert res.n_users == 2 and res.n_skipped == 1
    assert res.per_user == {"u1": 1.0, "u2": 0.0}
    assert gauc(user_scores, positives, weighting="uniform").gauc == pytest.approx(0.5)
    assert gauc(user_scores, positives, weighting="clicks").gauc == pytest.approx(0.5)
    with pytest.raises(ValueError):
        gauc(user_scores, positives, weighting="bogus")
    with pytest.raises(ValueError):
        gauc({"u3": user_scores["u3"]}, positives)


def test_gauc_removes_user_base_rate_bias():
    # Each user's items are ranked randomly, but the high-activity user gets
    # higher scores across the board: pooled AUC rewards that, GAUC does not.
    rng = random.Random(0)
    user_scores, positives, y, s = {}, {}, [], []
    for u, (offset, rate) in enumerate([(1.0, 0.8), (0.0, 0.1)] * 20):
        scores = {i: offset + rng.random() for i in range(30)}
        pos = {i for i in scores if rng.random() < rate}
        user_scores[u], positives[u] = scores, pos
        for i, v in scores.items():
            y.append(1 if i in pos else 0)
            s.append(v)
    pooled = sklearn_metrics.roc_auc_score(y, s)
    grouped = gauc(user_scores, positives).gauc
    assert pooled > 0.8
    assert abs(grouped - 0.5) < 0.06


def test_r_precision_and_f1():
    assert r_precision([1, 2, 3, 4], {1, 2}) == pytest.approx(1.0)
    assert r_precision([3, 1, 2], {1, 2}) == pytest.approx(0.5)
    assert r_precision([1, 9], {1: 2.0, 2: 0.0}) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        r_precision([1], set())
    # P@4 = 0.5, R@4 = 2/3 -> F1 = 4/7
    assert f1_at_k([1, 2, 3, 4], {1, 3, 9}, 4) == pytest.approx(4 / 7)
    assert f1_at_k([5, 6], {1}, 2) == 0.0
    assert mean_metric(r_precision, [[1, 2], [2, 1]], [{1}, {1}]) == pytest.approx(0.5)
