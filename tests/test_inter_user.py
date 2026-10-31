import math

import numpy as np
import pytest

from reco_eval_kit.inter_user import (
    personalization,
    recommendation_entropy,
    temporal_diversity,
)


def test_personalization_extremes():
    same = [[1, 2, 3]] * 4
    disjoint = [[1, 2], [3, 4], [5, 6]]
    assert personalization(same) == pytest.approx(0.0)
    assert personalization(disjoint) == pytest.approx(1.0)
    assert personalization(same, similarity="jaccard") == pytest.approx(0.0)


def test_personalization_hand_computed():
    lists = [[1, 2], [2, 3], [4, 5]]
    # cosine sims: (1/2, 0, 0) -> 1 - 1/6 ; jaccard: (1/3, 0, 0) -> 1 - 1/9
    assert personalization(lists) == pytest.approx(1 - 1 / 6)
    assert personalization(lists, similarity="jaccard") == pytest.approx(1 - 1 / 9)
    # truncation to top-1 makes lists disjoint
    assert personalization(lists, k=1) == pytest.approx(1.0)


def test_personalization_subsample_and_validation():
    rng = np.random.default_rng(0)
    lists = [list(rng.choice(50, 5, replace=False)) for _ in range(300)]
    full = personalization(lists)
    sub = personalization(lists, max_users=150, random_state=1)
    assert abs(full - sub) < 0.02
    with pytest.raises(ValueError):
        personalization([[1]])
    with pytest.raises(ValueError):
        personalization([[1], []])
    with pytest.raises(ValueError):
        personalization([[1], [2]], similarity="dice")
    with pytest.raises(ValueError):
        personalization([[1], [2]], k=0)


def test_recommendation_entropy():
    assert recommendation_entropy([[1], [1]]) == pytest.approx(0.0)
    assert recommendation_entropy([[1, 2], [3, 4]]) == pytest.approx(math.log(4))
    assert recommendation_entropy([[1, 2], [3, 4]], normalize=True) == pytest.approx(1.0)
    assert recommendation_entropy([[1, 2], [3, 4]], catalog=16, normalize=True) == pytest.approx(0.5)
    assert recommendation_entropy([[1, 1, 2]], k=1) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        recommendation_entropy([[]])
    with pytest.raises(ValueError):
        recommendation_entropy([[1, 2, 3]], catalog=2, normalize=True)


def test_temporal_diversity():
    prev = {"u": [1, 2, 3, 4], "v": [5, 6], "w": [9]}
    cur = {"u": [1, 2, 7, 8], "v": [5, 6], "x": [1]}
    assert temporal_diversity(prev, cur) == pytest.approx((0.5 + 0.0) / 2)
    assert temporal_diversity(prev, cur, k=2) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        temporal_diversity({"a": [1]}, {"b": [1]})
