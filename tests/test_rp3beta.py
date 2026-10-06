import numpy as np
import pandas as pd
import pytest

from reco_eval_kit.baselines import PopularityRecommender, RP3betaRecommender, RandomRecommender
from reco_eval_kit.metrics import hit_rate_at_k
from reco_eval_kit.splitting import leave_one_out
from reco_eval_kit.synthetic import generate_interactions


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


def _binary(frame):
    users = sorted(frame["user_id"].unique())
    items = sorted(frame["item_id"].unique())
    X = np.zeros((len(users), len(items)))
    for u, i in zip(frame["user_id"], frame["item_id"]):
        X[users.index(u), items.index(i)] = 1.0
    return X


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError):
        RP3betaRecommender().recommend("u1", k=2)


def test_weights_match_closed_form():
    model = RP3betaRecommender(alpha=0.8, beta=0.3, normalize=False).fit(INTERACTIONS)
    X = _binary(INTERACTIONS)
    du, di = X.sum(1), X.sum(0)
    expected = ((X.T / di[:, None]) ** 0.8) @ ((X / du[:, None]) ** 0.8)
    expected = expected / di[None, :] ** 0.3
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(model.item_weights_, expected)
    np.testing.assert_allclose(model.item_popularity_, di)


def test_p3alpha_rows_are_walk_probabilities():
    # alpha=1, beta=0: before removing the diagonal each row of W is a
    # 2-step item->user->item distribution, so rows sum to one minus the
    # self-return probability.
    model = RP3betaRecommender(alpha=1.0, beta=0.0, normalize=False).fit(INTERACTIONS)
    X = _binary(INTERACTIONS)
    du, di = X.sum(1), X.sum(0)
    full = (X.T / di[:, None]) @ (X / du[:, None])
    np.testing.assert_allclose(full.sum(axis=1), 1.0)
    np.testing.assert_allclose(model.item_weights_.sum(axis=1), 1.0 - np.diag(full))


def test_normalized_rows_and_zero_diagonal():
    model = RP3betaRecommender(beta=0.5).fit(INTERACTIONS)
    W = model.item_weights_
    assert np.allclose(np.diag(W), 0.0)
    sums = W.sum(axis=1)
    # item 14 has no co-occurring item -> empty row stays zero
    assert np.allclose(sums[sums > 0], 1.0)
    assert sums[model.catalog_.index(14)] == 0.0


def test_n_neighbors_truncates_rows():
    frame = generate_interactions(n_users=30, n_items=25, n_interactions=300, seed=3)
    model = RP3betaRecommender(n_neighbors=3, normalize=False).fit(frame)
    assert int((model.item_weights_ > 0).sum(axis=1).max()) <= 3
    full = RP3betaRecommender(n_neighbors=None, normalize=False).fit(frame)
    for row_t, row_f in zip(model.item_weights_, full.item_weights_):
        kept = row_t > 0
        np.testing.assert_allclose(row_t[kept], row_f[kept])
        if kept.any():
            assert row_t[kept].min() >= np.sort(row_f)[-3] - 1e-12


def test_beta_penalizes_popular_items():
    # u0 only saw hub item 0. Six users pair item 0 with popular item 1 and
    # one user pairs it with niche item 2: the plain walk prefers item 1,
    # a strong popularity penalty (beta > 1 here) flips the order.
    pairs = [("u0", 0)]
    pairs += [(f"p{i}", 0) for i in range(6)] + [(f"p{i}", 1) for i in range(6)]
    pairs += [("n0", 0), ("n0", 2)]
    frame = make_frame(pairs)
    plain = RP3betaRecommender(beta=0.0).fit(frame)
    strong = RP3betaRecommender(beta=1.5).fit(frame)
    assert plain.recommend("u0", k=2) == [1, 2]
    assert strong.recommend("u0", k=2) == [2, 1]


def test_recovers_held_out_affinity_and_excludes_seen():
    pairs = [
        ("a1", 1), ("a1", 2), ("a1", 3),
        ("a2", 1), ("a2", 2), ("a2", 3),
        ("a3", 1), ("a3", 2),
        ("b1", 10), ("b1", 11), ("b1", 12),
        ("b2", 10), ("b2", 11),
    ]
    model = RP3betaRecommender().fit(make_frame(pairs))
    recs = model.recommend("a3", k=3)
    assert recs[0] == 3
    assert {1, 2}.isdisjoint(recs)
    assert len(recs) == 3


def test_unknown_user_and_shortfall_fallback_to_popularity():
    model = RP3betaRecommender().fit(INTERACTIONS)
    assert model.recommend("nobody", k=2) == [10, 11]
    # u4 only saw an isolated item: everything comes from the fallback.
    assert model.recommend("u4", k=3) == PopularityRecommender().fit(INTERACTIONS).recommend(
        k=3, exclude={14}
    )
    assert model.recommend("u1", k=10, exclude={12}) == [14]


def test_deterministic():
    frame = generate_interactions(n_users=20, n_items=30, n_interactions=200, seed=1)
    a = RP3betaRecommender(alpha=0.7, beta=0.4, n_neighbors=5).fit(frame)
    b = RP3betaRecommender(alpha=0.7, beta=0.4, n_neighbors=5).fit(frame)
    for user in sorted(frame["user_id"].unique())[:5]:
        assert a.recommend(user, k=5) == b.recommend(user, k=5)


def test_beats_random_on_synthetic_data():
    frame = generate_interactions(n_users=80, n_items=120, n_interactions=3000, seed=11)
    frame = frame.assign(timestamp=np.arange(len(frame)))
    train, test = leave_one_out(frame)
    rp3 = RP3betaRecommender(alpha=1.0, beta=0.3).fit(train)
    rnd = RandomRecommender(seed=0).fit(train)
    held = test.groupby("user_id")["item_id"].apply(set).to_dict()
    hr_rp3 = np.mean([hit_rate_at_k(rp3.recommend(u, k=10), held[u], 10) for u in held])
    hr_rnd = np.mean([hit_rate_at_k(rnd.recommend(u, k=10, exclude=rp3.histories_.get(u, set())), held[u], 10) for u in held])
    assert hr_rp3 > hr_rnd


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0.0},
        {"alpha": -1.0},
        {"alpha": float("inf")},
        {"beta": -0.1},
        {"beta": float("nan")},
        {"n_neighbors": 0},
        {"n_neighbors": 2.5},
        {"n_neighbors": True},
    ],
)
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        RP3betaRecommender(**kwargs)
