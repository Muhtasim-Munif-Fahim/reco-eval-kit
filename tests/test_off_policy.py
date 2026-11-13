import numpy as np
import pytest

from reco_eval_kit.off_policy import (
    clipped_ips_estimate,
    direct_method_estimate,
    doubly_robust_estimate,
    evaluate_policy,
    importance_weights,
    ips_estimate,
    snips_estimate,
    switch_dr_estimate,
)


def _bandit(n=4000, k=5, seed=0):
    """Synthetic contextual bandit with known expected rewards."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    theta = rng.normal(size=(3, k))
    q = 1.0 / (1.0 + np.exp(-(x @ theta)))  # true click probabilities
    logits = 0.5 * (x @ theta)
    mu = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    mu = 0.8 * mu + 0.2 / k
    actions = np.array([rng.choice(k, p=row) for row in mu])
    rewards = rng.binomial(1, q[np.arange(n), actions]).astype(float)
    pscore = mu[np.arange(n), actions]
    target = np.zeros((n, k))
    target[np.arange(n), q.argmax(axis=1)] = 1.0  # greedy oracle policy
    true_value = float(np.mean(np.sum(target * q, axis=1)))
    return actions, rewards, pscore, mu, target, q, true_value


def test_on_policy_ips_equals_sample_mean():
    a, r, p, mu, _, _, _ = _bandit(n=500)
    est = ips_estimate(a, r, p, mu)
    assert est.value == pytest.approx(r.mean())
    assert snips_estimate(a, r, p, mu).value == pytest.approx(r.mean())
    np.testing.assert_allclose(importance_weights(a, p, mu), 1.0)
    assert est.effective_sample_size == pytest.approx(500)


def test_estimators_recover_true_value():
    a, r, p, mu, pi, q, v = _bandit(seed=1)
    noisy_q = np.clip(q + 0.15, 0, 1)  # biased reward model
    res = evaluate_policy(a, r, p, pi, noisy_q)
    assert set(res) == {"ips", "snips", "clipped_ips", "dm", "dr", "switch_dr"}
    for name in ("ips", "snips", "dr"):
        assert res[name].ci_low - 0.02 <= v <= res[name].ci_high + 0.02, name
    # DM inherits the reward-model bias; DR corrects it.
    assert abs(res["dm"].value - v) > 0.1
    assert abs(res["dr"].value - v) < 0.03


def test_ips_unbiased_over_replications():
    vals, truths = [], []
    for s in range(30):
        a, r, p, mu, pi, q, v = _bandit(n=800, seed=100 + s)
        vals.append(ips_estimate(a, r, p, pi).value)
        truths.append(v)
    assert abs(np.mean(vals) - np.mean(truths)) < 0.02


def test_dr_with_perfect_model_has_lower_variance_than_ips():
    a, r, p, mu, pi, q, v = _bandit(seed=2)
    ips = ips_estimate(a, r, p, pi)
    dr = doubly_robust_estimate(a, r, p, pi, q)
    assert dr.std_error < ips.std_error
    assert direct_method_estimate(pi, q).value == pytest.approx(v)


def test_switch_dr_limits_and_clipping():
    a, r, p, mu, pi, q, _ = _bandit(n=600, seed=3)
    qh = np.clip(q - 0.1, 0, 1)
    dr = doubly_robust_estimate(a, r, p, pi, qh).value
    dm = direct_method_estimate(pi, qh).value
    assert switch_dr_estimate(a, r, p, pi, qh, tau=np.inf).value == pytest.approx(dr)
    assert switch_dr_estimate(a, r, p, pi, qh, tau=0.0).value == pytest.approx(dm)
    w = importance_weights(a, p, pi)
    big = clipped_ips_estimate(a, r, p, pi, max_weight=w.max() + 1)
    assert big.value == pytest.approx(ips_estimate(a, r, p, pi).value)
    small = clipped_ips_estimate(a, r, p, pi, max_weight=1.0)
    assert small.value <= big.value + 1e-12


def test_hand_computed_example():
    actions = [0, 1, 1, 0]
    rewards = [1.0, 0.0, 1.0, 1.0]
    pscore = [0.5, 0.5, 0.25, 0.8]
    pi = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 1.0]])
    # w = [2, 2, 2, 0]; IPS = (2 + 0 + 2 + 0)/4 = 1; SNIPS = 4/6
    assert ips_estimate(actions, rewards, pscore, pi).value == pytest.approx(1.0)
    est = snips_estimate(actions, rewards, pscore, pi)
    assert est.value == pytest.approx(4 / 6)
    assert est.effective_sample_size == pytest.approx(36 / 12)
    d = est.as_dict()
    assert d["estimator"] == "snips" and d["n"] == 4
    q = np.full((4, 2), 0.5)
    # DR_i = 0.5 + w_i (r_i - 0.5) -> [1.5, -0.5, 1.5, 0.5]
    assert doubly_robust_estimate(actions, rewards, pscore, pi, q).value == pytest.approx(0.75)


def test_validation():
    pi = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        ips_estimate([0, 2], [1, 0], [0.5, 0.5], pi)
    with pytest.raises(ValueError):
        ips_estimate([0, 1], [1, 0], [0.0, 0.5], pi)
    with pytest.raises(ValueError):
        ips_estimate([0, 1], [1, 0], [0.5, 0.5], np.array([[0.5, 0.2], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        ips_estimate([0], [1, 0], [0.5, 0.5], pi)
    with pytest.raises(ValueError):
        snips_estimate([1, 0], [1, 0], [0.5, 0.5], pi)
    with pytest.raises(ValueError):
        doubly_robust_estimate([0, 1], [1, 0], [0.5, 0.5], pi, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        clipped_ips_estimate([0, 1], [1, 0], [0.5, 0.5], pi, max_weight=0)
    with pytest.raises(ValueError):
        switch_dr_estimate([0, 1], [1, 0], [0.5, 0.5], pi, np.zeros((2, 2)), tau=-1)
    with pytest.raises(ValueError):
        ips_estimate([0, 1], [1, 0], [0.5, 0.5], pi, confidence=1.5)
