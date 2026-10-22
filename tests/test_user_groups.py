"""Tests for user-group fairness auditing."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from reco_eval_kit.cli import main
from reco_eval_kit.user_groups import (
    activity_groups,
    cold_start_groups,
    compare_group_fairness,
    group_fairness_report,
    render_group_fairness_section,
)


def _interactions():
    rows = []
    for user, count in zip(range(20), [50, 40, 30] + [5] * 10 + [1] * 7):
        rows += [(user, item) for item in range(count)]
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def test_advantaged_split_takes_top_fraction():
    groups = activity_groups(_interactions(), top_fraction=0.1)
    adv = sorted(u for u, g in groups.items() if g == "advantaged")
    assert adv == [0, 1]  # ceil(0.1 * 20) = 2 most active
    assert sum(g == "disadvantaged" for g in groups.values()) == 18
    tiny = activity_groups(_interactions(), top_fraction=0.01)
    assert sum(g == "advantaged" for g in tiny.values()) == 1


def test_quantile_buckets_are_ordered_by_activity():
    inter = _interactions()
    groups = activity_groups(inter, method="quantile", n_groups=4)
    counts = inter.groupby("user_id").size()
    by_group = {}
    for u, g in groups.items():
        by_group.setdefault(g, []).append(counts[u])
    assert sorted(by_group) == ["Q1", "Q2", "Q3", "Q4"]
    assert [len(by_group[q]) for q in ["Q1", "Q2", "Q3", "Q4"]] == [5, 5, 5, 5]
    assert max(by_group["Q1"]) <= min(by_group["Q4"])
    assert 0 in [u for u, g in groups.items() if g == "Q4"]


def test_users_argument_adds_zero_activity_users_and_cold_start():
    inter = _interactions()
    groups = activity_groups(inter, users=[0, 5, 99], top_fraction=0.3)
    assert groups == {0: "advantaged", 5: "disadvantaged", 99: "disadvantaged"}
    cold = cold_start_groups(inter, users=[0, 5, 15, 99], max_interactions=5)
    assert cold == {0: "warm", 5: "cold", 15: "cold", 99: "cold"}
    with pytest.raises(ValueError):
        activity_groups(inter, method="kmeans")
    with pytest.raises(ValueError):
        activity_groups(inter, top_fraction=1.5)
    with pytest.raises(ValueError):
        activity_groups(inter, method="quantile", n_groups=1)
    with pytest.raises(ValueError):
        cold_start_groups(inter, max_interactions=-1)
    with pytest.raises(ValueError):
        activity_groups(pd.DataFrame({"user": [1]}))


def test_report_statistics_match_hand_computation():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.2, 0.0])
    groups = ["a", "a", "a", "b", "b", "b", "b", "b"]
    rep = group_fairness_report(scores, groups, n_permutations=500, seed=0)
    assert rep.groups["a"]["mean"] == pytest.approx(0.8)
    assert rep.groups["b"]["mean"] == pytest.approx(0.16)
    assert rep.gap == pytest.approx(0.64)
    assert rep.best_group == "a" and rep.worst_group == "b"
    assert rep.ratio == pytest.approx(0.16 / 0.8)
    means = np.array([0.8, 0.16])
    assert rep.coefficient_of_variation == pytest.approx(means.std() / means.mean())
    assert rep.gini == pytest.approx(abs(0.8 - 0.16) / (2 * 2 * means.mean()))
    a = scores[:3]
    half = stats.t.ppf(0.975, 2) * a.std(ddof=1) / np.sqrt(3)
    assert rep.groups["a"]["ci_low"] == pytest.approx(0.8 - half)
    assert rep.parametric_test == "welch_t"
    assert rep.parametric_p == pytest.approx(stats.ttest_ind(a, scores[3:], equal_var=False).pvalue)
    assert rep.permutation_p < 0.05
    assert rep.n_users == 8
    assert set(rep.to_frame().columns) >= {"mean", "ci_low", "ci_high", "n_users"}


def test_permutation_test_is_calibrated_under_null():
    rng = np.random.default_rng(1)
    pvals = []
    for rep_seed in range(60):
        scores = rng.random(60)
        groups = rng.choice(["x", "y", "z"], size=60)
        pvals.append(group_fairness_report(scores, groups, n_permutations=199, seed=rep_seed).permutation_p)
    pvals = np.array(pvals)
    assert 0.0 < np.mean(pvals < 0.1) < 0.25
    rep = group_fairness_report(rng.random(60), rng.choice(["x", "y", "z"], size=60), n_permutations=99)
    assert rep.parametric_test == "kruskal_wallis"


def test_compare_and_render():
    rng = np.random.default_rng(2)
    groups = ["adv"] * 10 + ["dis"] * 30
    fair = rng.random(40) * 0.2
    unfair = np.r_[0.5 + rng.random(10) * 0.1, rng.random(30) * 0.1]
    reps = compare_group_fairness({"fair": fair, "unfair": unfair}, groups, n_permutations=300, seed=0)
    assert reps["unfair"].gap > reps["fair"].gap
    assert reps["unfair"].permutation_p < 0.01
    md = render_group_fairness_section(reps, "NDCG@10", "activity")
    assert "## User-group fairness (NDCG@10 by activity)" in md
    assert "| unfair |" in md and "adv: 10" in md
    assert render_group_fairness_section({}) == ""


def test_report_validation():
    with pytest.raises(ValueError, match="one entry"):
        group_fairness_report([0.1, 0.2], ["a"])
    with pytest.raises(ValueError, match="two groups"):
        group_fairness_report([0.1, 0.2], ["a", "a"])
    with pytest.raises(ValueError, match="finite"):
        group_fairness_report([0.1, np.nan], ["a", "b"])
    with pytest.raises(ValueError, match="ci"):
        group_fairness_report([0.1, 0.2], ["a", "b"], ci=1.0)
    single = group_fairness_report([0.1, 0.2, 0.3], ["a", "b", "b"], n_permutations=10)
    assert np.isnan(single.groups["a"]["se"])


def test_cli_user_groups_section(tmp_path, capsys):
    out = tmp_path / "report.md"
    argv = ["--n-users", "40", "--n-items", "60", "--n-interactions", "900", "--k", "5",
            "--user-groups", "quantile", "--user-groups-n", "2", "--output", str(out)]
    assert main(argv) == 0
    text = out.read_text()
    assert "## User-group fairness (NDCG@5 by quantile)" in text
    assert "| Q1 | Q2 |" in text.replace("Overall | ", "")
    assert "largest user-group NDCG gap" in capsys.readouterr().out
