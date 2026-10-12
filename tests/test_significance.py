import numpy as np
import pytest
from scipy import stats

from reco_eval_kit.metrics import mean_metric, ndcg_at_k, recall_at_k
from reco_eval_kit.significance import (
    compare_recommenders,
    holm_adjust,
    paired_comparison,
    per_user_scores,
)


def test_per_user_scores_average_to_mean_metric():
    recs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rels = [{1}, {6, 10}, {11}]
    scores = per_user_scores(ndcg_at_k, recs, rels, k=3)
    assert scores.shape == (3,)
    assert scores.mean() == pytest.approx(mean_metric(ndcg_at_k, recs, rels, k=3))


def test_clear_improvement_is_significant():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.5, 300)
    b = a + 0.05 + rng.normal(0, 0.05, 300)
    res = paired_comparison(a, b, n_resamples=4000, seed=1)
    assert res.mean_diff == pytest.approx(np.mean(b - a))
    assert res.significant(0.01)
    assert res.randomization_p == pytest.approx(1 / 4001)
    assert res.ci_low > 0 and res.ci_high > res.ci_low
    assert res.effect_size_dz > 0.5


def test_t_and_wilcoxon_match_scipy():
    rng = np.random.default_rng(2)
    a = rng.normal(size=50)
    b = a + rng.normal(0.1, 1.0, 50)
    res = paired_comparison(a, b, n_resamples=500)
    assert res.t_p == pytest.approx(stats.ttest_rel(b, a).pvalue)
    assert res.wilcoxon_p == pytest.approx(stats.wilcoxon(b - a).pvalue)


def test_null_pvalues_are_roughly_uniform():
    rng = np.random.default_rng(3)
    pvals = []
    for i in range(200):
        a = rng.uniform(size=40)
        b = rng.uniform(size=40)
        pvals.append(paired_comparison(a, b, n_resamples=400, seed=i).randomization_p)
    pvals = np.array(pvals)
    assert 0.02 <= np.mean(pvals < 0.05) <= 0.09
    assert 0.4 <= np.mean(pvals) <= 0.6


def test_randomization_matches_exact_enumeration_for_small_n():
    d = np.array([0.3, -0.1, 0.2, 0.25, 0.05, -0.02])
    signs = np.array(np.meshgrid(*[[-1, 1]] * d.size)).T.reshape(-1, d.size)
    exact = np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12)
    res = paired_comparison(np.zeros_like(d), d, n_resamples=40_000, seed=5)
    assert res.randomization_p == pytest.approx(exact, abs=0.01)


def test_identical_systems():
    a = np.array([0.1, 0.4, 0.0, 1.0])
    res = paired_comparison(a, a.copy(), n_resamples=200)
    assert res.mean_diff == 0.0
    assert res.randomization_p == 1.0
    assert res.t_p == 1.0 and res.wilcoxon_p == 1.0
    assert res.ci_low == res.ci_high == 0.0
    assert not res.significant()


def test_seed_reproducibility_and_to_dict():
    rng = np.random.default_rng(4)
    a, b = rng.uniform(size=30), rng.uniform(size=30)
    r1 = paired_comparison(a, b, n_resamples=300, seed=9)
    r2 = paired_comparison(a, b, n_resamples=300, seed=9)
    assert r1 == r2
    assert r1.to_dict()["n_users"] == 30


def test_compare_recommenders_end_to_end():
    rels = [{u} for u in range(100)]
    good = [[u, 1000 + u] for u in range(100)]
    bad = [[2000 + u, 3000 + u] for u in range(100)]
    res = compare_recommenders(recall_at_k, bad, good, rels, k=2, n_resamples=500)
    assert res.mean_a == 0.0 and res.mean_b == 1.0
    assert res.significant()
    with pytest.raises(ValueError, match="same users"):
        compare_recommenders(recall_at_k, bad[:5], good, rels, k=2)


def test_holm_adjust():
    adj = holm_adjust([0.01, 0.04, 0.03, 0.5])
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08->0.09 (monotone), 0.5*1=0.5
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError):
        holm_adjust([1.5])


def test_validation():
    with pytest.raises(ValueError, match="same length"):
        paired_comparison([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="two paired"):
        paired_comparison([1], [2])
    with pytest.raises(ValueError, match="n_resamples"):
        paired_comparison([1, 2], [2, 3], n_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        paired_comparison([1, 2], [2, 3], confidence=1.0)


def test_cli_significance_section(tmp_path):
    from reco_eval_kit.cli import main

    output = tmp_path / "report.md"
    args = [
        "--n-users", "30", "--n-items", "40", "--n-interactions", "400",
        "--k", "5", "--seed", "0", "--output", str(output),
        "--significance-baseline", "popularity", "--significance-resamples", "200",
    ]
    assert main(args) == 0
    text = output.read_text(encoding="utf-8")
    assert "## Paired significance vs popularity (NDCG@5)" in text
    assert "| ease |" in text
    assert "| popularity |" not in text.split("## Paired significance")[1]


def test_cli_rejects_unknown_baseline(tmp_path):
    from reco_eval_kit.cli import main

    with pytest.raises(SystemExit, match="unknown --significance-baseline"):
        main(["--n-users", "5", "--output", str(tmp_path / "r.md"), "--significance-baseline", "nope"])
