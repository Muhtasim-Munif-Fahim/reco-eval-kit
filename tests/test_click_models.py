import numpy as np
import pytest

from reco_eval_kit.click_models import (
    click_log_likelihood,
    click_perplexity,
    fit_cascade_model,
    fit_position_based_model,
    fit_power_law,
    randomization_propensities,
)

K = 5
TRUE_THETA = np.arange(1, K + 1) ** -1.0


def _simulate(n_sessions=4000, seed=0, shuffle=True):
    rng = np.random.default_rng(seed)
    n_q, n_d = 20, K
    gamma = {(q, d): rng.uniform(0.1, 0.9) for q in range(n_q) for d in range(n_d)}
    recs = []
    for _ in range(n_sessions):
        q = int(rng.integers(n_q))
        order = rng.permutation(n_d) if shuffle else np.arange(n_d)
        for k, d in enumerate(order, start=1):
            c = int(rng.random() < TRUE_THETA[k - 1] * gamma[(q, int(d))])
            recs.append((q, int(d), k, c))
    return recs, gamma


def test_pbm_recovers_examination_and_attractiveness():
    recs, gamma = _simulate()
    res = fit_position_based_model(recs, max_iter=500)
    assert res.examination[0] == pytest.approx(1.0)
    np.testing.assert_allclose(res.examination, TRUE_THETA, atol=0.08)
    est = np.array([res.attractiveness[p] * res.raw_examination[0] for p in gamma])
    true = np.array(list(gamma.values()))
    assert np.corrcoef(est, true)[0, 1] > 0.9
    assert res.propensity(3) == pytest.approx(res.examination[2])


def test_pbm_likelihood_monotone_and_beats_constant():
    recs, _ = _simulate(n_sessions=800, seed=1)
    a = fit_position_based_model(recs, max_iter=2, tol=0)
    b = fit_position_based_model(recs, max_iter=50, tol=0)
    assert b.log_likelihood >= a.log_likelihood - 1e-9
    ctr = np.mean([c for *_, c in recs])
    const = np.mean([np.log(ctr) if c else np.log(1 - ctr) for *_, c in recs])
    assert click_log_likelihood(b, recs) > const


def test_randomization_propensities_match_truth():
    recs, _ = _simulate(n_sessions=6000, seed=2)
    p = randomization_propensities(recs)
    np.testing.assert_allclose(p, TRUE_THETA, atol=0.06)
    assert fit_power_law(p) == pytest.approx(1.0, abs=0.12)


def test_fit_power_law_exact():
    assert fit_power_law(np.arange(1, 8) ** -0.7) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        fit_power_law([1.0, 0.0])


def test_perplexity_bounds():
    recs, _ = _simulate(n_sessions=500, seed=3)
    res = fit_position_based_model(recs)
    ppl = click_perplexity(res, recs)
    assert ppl.shape == (K,)
    assert np.all((ppl >= 1.0) & (ppl <= 2.0))


def test_cascade_mle():
    sessions = [
        ("q", ["a", "b", "c"], [0, 1, 0]),
        ("q", ["a", "b", "c"], [1, 0, 0]),
        ("q", ["a", "b", "c"], [0, 0, 0]),
    ]
    res = fit_cascade_model(sessions, prior=(0, 0))
    assert res.examined[("q", "a")] == 3
    assert res.attractiveness[("q", "a")] == pytest.approx(1 / 3)
    assert res.examined[("q", "b")] == 2
    assert res.attractiveness[("q", "b")] == pytest.approx(1 / 2)
    assert res.attractiveness[("q", "c")] == pytest.approx(0.0)
    assert np.isfinite(res.log_likelihood)


def test_validation():
    with pytest.raises(ValueError):
        fit_position_based_model([])
    with pytest.raises(ValueError):
        fit_position_based_model([("q", "a", 0, 1)])
    with pytest.raises(ValueError):
        fit_position_based_model([("q", "a", 1, 2)])
    with pytest.raises(ValueError):
        randomization_propensities([("q", "a", 1, 0), ("q", "b", 2, 1)])
    with pytest.raises(ValueError):
        fit_cascade_model([("q", ["a"], [0, 1])])
