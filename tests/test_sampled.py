"""Tests for sampled-negative metrics and the Krichene & Rendle correction."""

import numpy as np
import pytest
from scipy.stats import hypergeom

from reco_eval_kit.cli import main
from reco_eval_kit.sampled import (
    correction_weights,
    corrected_sampled_metric,
    expected_sampled_metric,
    full_rank,
    metric_by_rank,
    rank_transition_matrix,
    sample_negative_ranks,
    sampled_rank,
)


def test_transition_matrix_is_hypergeometric():
    P = rank_transition_matrix(40, 9)
    assert P.shape == (41, 10)
    assert np.allclose(P.sum(axis=1), 1.0)
    for R in (0, 3, 20, 40):
        assert np.allclose(P[R], hypergeom(40, R, 9).pmf(np.arange(10)))
    assert P[0, 0] == 1.0 and P[40, 9] == 1.0
    # expected sampled rank is R * m / n
    assert np.allclose(P @ np.arange(10), np.arange(41) * 9 / 40)


def test_metric_by_rank_definitions():
    assert metric_by_rank("hit", 3, 5).tolist() == [1, 1, 1, 0, 0]
    ndcg = metric_by_rank("ndcg", 2, 4)
    assert ndcg == pytest.approx([1.0, 1 / np.log2(3), 0.0, 0.0])
    assert metric_by_rank("rr", None, 3) == pytest.approx([1.0, 0.5, 1 / 3])
    with pytest.raises(ValueError, match="metric"):
        metric_by_rank("map", 3, 5)
    with pytest.raises(ValueError, match="k is required"):
        metric_by_rank("hit", None, 5)


def test_full_and_sampled_rank_helpers():
    ranking = ["a", "b", "c", "d"]
    assert full_rank(ranking, "c", ["a", "b", "c", "d", "e"]) == 2
    assert full_rank(ranking, "a", ["c", "e", "a"]) == 0
    assert full_rank(ranking, "z", ["a", "b", "z"]) == 2  # missing positive = worst
    assert full_rank(ranking, "d", ["e", "f", "d"]) == 0  # unlisted negatives rank last
    assert sampled_rank(0.5, [0.9, 0.5, 0.1]) == 2
    assert sampled_rank(0.5, [0.9, 0.5, 0.1], ties="optimistic") == 1
    assert sampled_rank(0.5, [0.9, 0.5, 0.1], ties="average") == 1.5
    with pytest.raises(ValueError, match="ties"):
        sampled_rank(0.5, [0.1], ties="random")


def test_no_sampling_reproduces_the_full_metric():
    for metric in ("hit", "ndcg"):
        w = correction_weights(30, 30, metric, 5, method="ls")
        assert np.allclose(w, metric_by_rank(metric, 5, 31))
    ranks = np.array([0, 4, 7, 29])
    est = corrected_sampled_metric(ranks, 30, 30, "hit", 5, method="ls")
    assert est.naive == pytest.approx(0.5) and est.corrected == pytest.approx(0.5, abs=1e-8)


def test_naive_sampled_metric_is_biased_upwards_and_bv_corrects_it():
    rng = np.random.default_rng(0)
    n, m = 999, 100
    full = np.minimum(rng.geometric(1 / 80, size=6000) - 1, n)
    sampled = sample_negative_ranks(full, n, m, seed=1)
    for metric in ("hit", "ndcg"):
        truth = metric_by_rank(metric, 10, n + 1)[full].mean()
        expected = expected_sampled_metric(full, n, m, metric, 10)
        assert expected["full"] == pytest.approx(truth)
        est = corrected_sampled_metric(sampled, n, m, metric, 10, method="bv", gamma=0.01)
        assert est.naive == pytest.approx(expected["expected_sampled"], rel=0.05)
        assert est.naive > 4 * truth  # sampled metrics hugely overstate HR/NDCG@10
        assert abs(est.corrected - truth) < 0.15 * truth
        assert abs(est.corrected - truth) < 0.05 * abs(est.naive - truth)


def test_correction_preserves_model_order_that_naive_sampling_can_flip():
    # model A: many items at rank 0 but a long tail; model B: consistently good ranks
    rng = np.random.default_rng(5)
    n, m, k = 499, 50, 5
    a = np.where(rng.random(4000) < 0.3, 0, rng.integers(30, n + 1, 4000))
    b = rng.integers(5, 41, 4000)  # never in the full top 5, but always near the top
    full_hit = {name: metric_by_rank("hit", k, n + 1)[r].mean() for name, r in (("a", a), ("b", b))}
    assert full_hit["a"] > full_hit["b"]
    est = {}
    for name, r in (("a", a), ("b", b)):
        s = sample_negative_ranks(r, n, m, seed=2)
        est[name] = corrected_sampled_metric(s, n, m, "hit", k, method="bv", gamma=0.001)
    assert est["b"].naive > est["a"].naive  # naive sampled HR@5 flips the order
    assert est["a"].corrected > est["b"].corrected  # the correction restores it


def test_per_user_candidate_counts_and_prior():
    rng = np.random.default_rng(3)
    ns = rng.integers(200, 260, size=3000)
    full = np.minimum(rng.geometric(1 / 30, size=3000) - 1, ns)
    sampled = np.array([rng.hypergeometric(R, n - R, 40) for R, n in zip(full, ns)])
    truth = np.mean([metric_by_rank("hit", 10, n + 1)[R] for R, n in zip(full, ns)])
    est = corrected_sampled_metric(sampled, ns, 40, "hit", 10)
    assert est.n_negatives == -1 and est.weights.size == 0
    assert abs(est.corrected - truth) < 0.2 * truth
    prior = np.ones(201)
    prior[50:] = 0.1
    w = correction_weights(200, 40, "hit", 10, prior=prior)
    assert w.shape == (41,)
    with pytest.raises(ValueError, match="shared"):
        corrected_sampled_metric(sampled, ns, 40, prior=np.ones(201))


def test_bv_gamma_trades_bias_for_variance():
    n, m = 300, 30
    P = rank_transition_matrix(n, m)
    target = metric_by_rank("hit", 10, n + 1)
    small = correction_weights(n, m, "hit", 10, gamma=1e-4)
    large = correction_weights(n, m, "hit", 10, gamma=1.0)
    bias = lambda w: np.mean((P @ w - target) ** 2)  # noqa: E731
    var = lambda w: np.mean(P @ (w**2) - (P @ w) ** 2)  # noqa: E731
    assert bias(small) < bias(large)
    assert var(large) < var(small)


def test_fractional_ranks_interpolate():
    est = corrected_sampled_metric([1.5], 20, 5, "hit", 2, method="naive")
    assert est.naive == pytest.approx(0.5)


def test_validation():
    with pytest.raises(ValueError, match="n_sampled"):
        rank_transition_matrix(10, 11)
    with pytest.raises(ValueError, match="n_negatives"):
        rank_transition_matrix(0, 1)
    with pytest.raises(ValueError, match="method"):
        correction_weights(10, 5, method="em")
    with pytest.raises(ValueError, match="gamma"):
        correction_weights(10, 5, gamma=-1)
    with pytest.raises(ValueError, match="prior"):
        correction_weights(10, 5, prior=np.ones(3))
    with pytest.raises(ValueError, match="empty"):
        corrected_sampled_metric([], 10, 5)
    with pytest.raises(ValueError, match="lie in"):
        corrected_sampled_metric([6], 10, 5)
    with pytest.raises(ValueError, match="full ranks"):
        sample_negative_ranks([11], 10, 5)
    with pytest.raises(ValueError, match="full ranks"):
        expected_sampled_metric([-1], 10, 5)


def test_cli_sampled_protocol(tmp_path):
    output = tmp_path / "report.md"
    args = ["--n-users", "10", "--n-items", "40", "--n-interactions", "120", "--k", "3",
            "--seed", "0", "--output", str(output), "--sampled-negatives", "10"]
    assert main(args) == 0
    text = output.read_text(encoding="utf-8")
    for label in ("Sampled-HR@K", "Corrected-HR@K", "Sampled-NDCG@K", "Corrected-NDCG@K"):
        assert f"| {label} |" in text
    assert "Sampled protocol" in text
    with pytest.raises(SystemExit):
        main(args[:-1] + ["0"])
