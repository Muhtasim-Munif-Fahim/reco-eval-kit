import itertools
import math

import pytest

from reco_eval_kit.cli import main
from reco_eval_kit.exposure import (
    expected_exposure,
    mean_expected_exposure,
    plackett_luce_rankings,
    policy_exposure,
    rbp_exposure,
    target_exposure,
)


def test_rbp_exposure_geometric_and_cutoff():
    assert rbp_exposure(4, 0.5).tolist() == [1.0, 0.5, 0.25, 0.125]
    assert rbp_exposure(4, 0.5, k=2).tolist() == [1.0, 0.5, 0.0, 0.0]
    with pytest.raises(ValueError):
        rbp_exposure(3, 1.0)


def test_target_exposure_splits_slots_evenly():
    rel, non = target_exposure(2, 4, gamma=0.5)
    assert rel == pytest.approx(0.75)
    assert non == pytest.approx((0.25 + 0.125) / 2)
    with pytest.raises(ValueError):
        target_exposure(5, 4)


def test_policy_exposure_averages_rankings():
    eps = policy_exposure([["a", "b"], ["b", "a"]], gamma=0.5)
    assert eps == {"a": 0.75, "b": 0.75}
    with pytest.raises(ValueError):
        policy_exposure([["a", "a"]])


def test_ideal_policy_has_zero_loss():
    rel = {"a", "b"}
    rankings = [list(p) + list(q) for p in itertools.permutations("ab") for q in itertools.permutations("cde")]
    ee = expected_exposure(rankings, rel, gamma=0.7)
    assert ee.loss == pytest.approx(0.0, abs=1e-12)
    assert ee.normalized_relevance == pytest.approx(1.0)
    assert ee.disparity == pytest.approx(ee.target_norm)


def test_decomposition_identity_and_deterministic_disparity():
    ranking = ["c", "a", "d", "b"]
    ee = expected_exposure([ranking], {"a", "b"}, gamma=0.5)
    assert ee.loss == pytest.approx(ee.disparity - ee.relevance + ee.target_norm)
    assert ee.disparity == pytest.approx(sum(g * g for g in [1, 0.5, 0.25, 0.125]))
    # Direct vector computation of ||eps - eps*||^2.
    t_rel, t_non = target_exposure(2, 4, 0.5)
    eps = {"c": 1.0, "a": 0.5, "d": 0.25, "b": 0.125}
    direct = sum((eps[i] - (t_rel if i in "ab" else t_non)) ** 2 for i in eps)
    assert ee.loss == pytest.approx(direct)


def test_unshown_candidates_count_in_target():
    ee_small = expected_exposure([["a", "b"]], {"a"}, gamma=0.5)
    ee_big = expected_exposure([["a", "b"]], {"a"}, gamma=0.5, n_candidates=10)
    assert ee_big.n_candidates == 10
    assert ee_big.target_norm != ee_small.target_norm
    with pytest.raises(ValueError):
        expected_exposure([["a", "b"]], {"c"}, n_candidates=2)


def test_mapping_relevance_and_unretrieved_relevant():
    ee = expected_exposure([["x", "y"]], {"a": 1, "x": 0})
    assert ee.n_relevant == 1 and ee.n_candidates == 3
    assert ee.relevance == pytest.approx(2 * (1.0 * 0.375 + 0.5 * 0.375))  # only non-relevant shown


def test_stochastic_policy_reduces_disparity_among_equals():
    cands = list("abcdef")
    det = plackett_luce_rankings(cands, [1.0] * 6, n_samples=5, temperature=0)
    assert all(r == cands for r in det)
    sto = plackett_luce_rankings(cands, [1.0] * 6, n_samples=3000, temperature=1.0, seed=0)
    rel = set(cands)
    ee_det = expected_exposure(det, rel)
    ee_sto = expected_exposure(sto, rel)
    assert ee_sto.disparity < ee_det.disparity
    assert ee_sto.loss < 0.01 < ee_det.loss


def test_plackett_luce_temperature_and_validation():
    cands = ["a", "b", "c"]
    cold = plackett_luce_rankings(cands, [3.0, 2.0, 1.0], n_samples=200, temperature=0.05, seed=1)
    assert all(r == cands for r in cold)
    hot = plackett_luce_rankings(cands, [3.0, 2.0, 1.0], n_samples=400, temperature=5.0, seed=1, k=2)
    assert all(len(r) == 2 for r in hot)
    assert len({tuple(r) for r in hot}) > 3
    first = sum(r[0] == "a" for r in hot) / len(hot)
    expected = math.exp(3 / 5) / sum(math.exp(s / 5) for s in (3, 2, 1))
    assert first == pytest.approx(expected, abs=0.08)
    for bad in (dict(scores=[1.0]), dict(scores=[1, 2, float("nan")]), dict(scores=[1, 2, 3], temperature=-1)):
        kwargs = dict(candidates=cands, **bad)
        with pytest.raises(ValueError):
            plackett_luce_rankings(**kwargs)


def test_mean_expected_exposure_skips_users_without_relevance():
    out = mean_expected_exposure([[["a", "b"]], [["c"]]], [{"a"}, set()], n_candidates=[2, None])
    assert out["users"] == 1
    single = expected_exposure([["a", "b"]], {"a"})
    assert out["loss"] == pytest.approx(single.loss)
    assert mean_expected_exposure([[["a"]]], [set()])["users"] == 0
    with pytest.raises(ValueError):
        mean_expected_exposure([[["a"]]], [])


def test_cli_expected_exposure_columns(tmp_path):
    output = tmp_path / "report.md"
    argv = ["--n-users", "8", "--n-items", "12", "--n-interactions", "40", "--k", "3",
            "--seed", "0", "--output", str(output), "--expected-exposure", "0.5"]
    assert main(argv) == 0
    text = output.read_text(encoding="utf-8")
    for col in ("| EE-L |", "| EE-D |", "| EE-R |", "| EE-R/ideal |", "| Expected exposure |"):
        assert col in text


def test_cli_expected_exposure_rejects_bad_args(tmp_path):
    with pytest.raises(SystemExit):
        main(["--expected-exposure", "-1", "--output", str(tmp_path / "r.md")])
    with pytest.raises(SystemExit):
        main(["--expected-exposure", "1", "--exposure-gamma", "1.5", "--output", str(tmp_path / "r.md")])
