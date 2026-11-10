import random

import pytest

from reco_eval_kit.incomplete import (
    bpref,
    condense,
    condensed_ap_at_k,
    condensed_ndcg_at_k,
    judged_at_k,
)
from reco_eval_kit.metrics import average_precision_at_k, ndcg_at_k

QRELS = {"a": 1, "b": 0, "c": 1, "d": 0, "e": 0}


def test_bpref_hand_computed():
    # R=2, N=3 -> denominator min(R, N)=2.
    # a has 1 judged non-relevant above: 1 - 1/2; c has 2 above: 1 - 2/2.
    assert bpref(["x", "b", "a", "d", "c", "y"], QRELS) == pytest.approx(0.25)


def test_bpref_ignores_unjudged_items():
    base = bpref(["b", "a", "d", "c"], QRELS)
    noisy = bpref(["u1", "b", "u2", "a", "u3", "d", "c", "u4"], QRELS)
    assert base == pytest.approx(noisy)


def test_bpref_perfect_partial_and_no_nonrelevant():
    assert bpref(["a", "c", "b", "d"], QRELS) == pytest.approx(1.0)
    assert bpref(["a"], QRELS) == pytest.approx(0.5)  # c never retrieved
    assert bpref(["c", "z"], {"a": 1, "c": 1}) == pytest.approx(0.5)  # N=0


def test_bpref_caps_nonrelevant_count_at_R():
    qrels = {"r": 1, "n1": 0, "n2": 0, "n3": 0}
    # R=1, N=3: denominator 1, nonrel above capped at 1 -> contribution 0.
    assert bpref(["n1", "n2", "r"], qrels) == pytest.approx(0.0)


def test_bpref_errors():
    with pytest.raises(ValueError):
        bpref(["a"], {"a": 0})
    with pytest.raises(ValueError):
        bpref(["a", "a"], QRELS)
    with pytest.raises(ValueError):
        bpref(["a"], {"a": -1})
    with pytest.raises(TypeError):
        bpref(["a"], ["a", "c"])


def test_judged_at_k():
    assert judged_at_k(["a", "x", "b", "y"], QRELS, 4) == pytest.approx(0.5)
    assert judged_at_k(["a", "x"], {"a", "b"}, 4) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        judged_at_k(["a"], QRELS, 0)


def test_condensed_metrics_skip_unjudged():
    ranking = ["x", "a", "y", "c"]
    qrels = {"a": 1, "c": 1, "b": 0}
    assert condense(ranking, qrels) == ["a", "c"]
    assert condensed_ndcg_at_k(ranking, qrels, 2) == pytest.approx(1.0)
    assert condensed_ap_at_k(ranking, qrels, 2) == pytest.approx(1.0)
    assert ndcg_at_k(ranking, qrels, 2) < 1.0


def test_condensed_equals_standard_when_fully_judged():
    rng = random.Random(0)
    items = [f"i{j}" for j in range(20)]
    for _ in range(25):
        qrels = {item: rng.choice([0, 0, 1, 2]) for item in items}
        if not any(qrels.values()):
            continue
        ranking = rng.sample(items, 10)
        rel = {i: g for i, g in qrels.items() if g > 0}
        assert condensed_ndcg_at_k(ranking, qrels, 5) == pytest.approx(ndcg_at_k(ranking, qrels, 5))
        assert condensed_ap_at_k(ranking, qrels, 5) == pytest.approx(average_precision_at_k(ranking, rel, 5))


def test_condensed_ap_requires_relevant():
    with pytest.raises(ValueError):
        condensed_ap_at_k(["a"], {"a": 0}, 3)
