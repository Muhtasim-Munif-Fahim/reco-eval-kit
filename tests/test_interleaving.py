import itertools

import numpy as np
import pytest

from reco_eval_kit.cli import main
from reco_eval_kit.interleaving import (
    METHODS,
    balanced_interleave,
    balanced_outcome,
    cascade_clicks,
    position_based_clicks,
    probabilistic_interleave,
    probabilistic_outcome,
    simulate_interleaving,
    team_draft_interleave,
    team_draft_outcome,
)


def test_balanced_interleave_matches_joachims_example():
    a = ["a1", "s", "a3", "a4"]
    b = ["b1", "a1", "s", "b4"]
    merged, first = balanced_interleave(a, b, 6, a_first=True)
    assert first is True
    assert merged == ["a1", "b1", "s", "a3", "a4", "b4"]
    merged_b, _ = balanced_interleave(a, b, 6, a_first=False)
    assert merged_b == ["b1", "a1", "s", "a3", "b4", "a4"]
    # every prefix of length m contains A's top ceil and B's top floor (or vice versa)
    for m in range(1, 5):
        ka = sum(1 for x in a[:m] if x in merged[: 2 * m])
        assert ka == m


def test_balanced_credit_rule():
    a = ["x", "y", "z"]
    b = ["y", "w", "x"]
    merged, _ = balanced_interleave(a, b, 4, a_first=True)  # x, y, w, z
    assert merged == ["x", "y", "w", "z"]
    # click on x only: lowest click x has rank 1 in A, 3 in B -> cut 1 -> A gets 1, B 0
    assert balanced_outcome(a, b, merged, [0]) == 1
    # click on w: rank 2 in B, absent in A -> cut 2: A top-2 {x,y} 0 clicks, B top-2 {y,w} 1
    assert balanced_outcome(a, b, merged, [2]) == -1
    assert balanced_outcome(a, b, merged, []) == 0
    assert balanced_outcome(a, a, balanced_interleave(a, a, 3, a_first=False)[0], [0, 2]) == 0


def test_team_draft_properties():
    rng = np.random.default_rng(0)
    a = list(range(10))
    b = list(range(9, -1, -1))
    for _ in range(50):
        merged, teams = team_draft_interleave(a, b, 7, rng)
        assert len(merged) == len(set(merged)) == 7
        assert abs(teams.count("A") - teams.count("B")) <= 1
        # each team contributes its own best untaken item
        taken = set()
        for item, team in zip(merged, teams):
            src = a if team == "A" else b
            assert item == next(x for x in src if x not in taken)
            taken.add(item)
    assert team_draft_outcome(["A", "B", "A"], [0, 2]) == 1
    assert team_draft_outcome(["A", "B", "A"], [1]) == -1
    assert team_draft_outcome(["A", "B"], [0, 1]) == 0


def test_interleave_handles_short_and_disjoint_lists():
    merged, teams = team_draft_interleave(["a"], ["b", "c", "d"], 10, rng=1)
    assert sorted(merged) == ["a", "b", "c", "d"] and len(teams) == 4
    merged, _ = balanced_interleave(["a", "a"], [], 5, a_first=False)
    assert merged == ["a"]
    merged, teams, probs = probabilistic_interleave(["a", "b"], ["c"], 10, rng=2)
    assert sorted(merged) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        team_draft_interleave(["a"], ["b"], 0)
    with pytest.raises(ValueError):
        probabilistic_interleave(["a"], ["b"], 2, tau=0)


def test_probabilistic_marginal_outcome_matches_enumeration():
    rng = np.random.default_rng(3)
    a = list("abcdef")
    b = list("fbdace")
    merged, teams, probs = probabilistic_interleave(a, b, 5, tau=2.0, rng=rng)
    clicks = [0, 2, 3]
    # brute force: P(assignment | list) ∝ prod P_team(l_i | remaining)
    total = expected = 0.0
    for assign in itertools.product("AB", repeat=len(merged)):
        w = np.prod([probs[i][0] if t == "A" else probs[i][1] for i, t in enumerate(assign)])
        total += w
        expected += w * team_draft_outcome(assign, clicks)
    assert probabilistic_outcome(probs, clicks) == pytest.approx(expected / total)
    assert probabilistic_outcome(probs, clicks, teams, marginalize=False) == team_draft_outcome(teams, clicks)
    # identical rankers -> each slot is a fair coin and the outcome is exactly 0 for one click
    _, _, same = probabilistic_interleave(a, a, 4, rng=rng)
    assert probabilistic_outcome(same, [1]) == pytest.approx(0.0)
    assert probabilistic_outcome(same, []) == 0.0


def test_click_models():
    rng = np.random.default_rng(0)
    ranking = list(range(10))
    assert cascade_clicks(ranking, {3}, p_rel=1.0, p_nonrel=0.0, p_stop=1.0, rng=rng) == [3]
    assert cascade_clicks(ranking, {1, 5}, p_rel=1.0, p_nonrel=0.0, p_stop=0.0, rng=rng) == [1, 5]
    assert position_based_clicks(ranking, {0, 9}, eta=0.0, p_rel=1.0, p_nonrel=0.0, rng=rng) == [0, 9]
    deep = sum(9 in position_based_clicks(ranking, {9}, eta=1.0, p_rel=1.0, rng=rng) for _ in range(2000))
    assert 140 < deep < 270  # examination prob 1/10
    with pytest.raises(ValueError):
        cascade_clicks(ranking, set(), p_rel=1.5)
    with pytest.raises(ValueError):
        position_based_clicks(ranking, set(), eta=-1)


def _synthetic(n_users=120, n_items=40, seed=0):
    rng = np.random.default_rng(seed)
    rel = [set(rng.choice(n_items, 4, replace=False).tolist()) for _ in range(n_users)]
    good = [sorted(range(n_items), key=lambda i: (i not in r, i)) for r in rel]
    rand = [rng.permutation(n_items).tolist() for _ in range(n_users)]
    return rel, good, rand


@pytest.mark.parametrize("method", METHODS)
def test_simulation_detects_better_ranker(method):
    rel, good, rand = _synthetic()
    res = simulate_interleaving(good, rand, rel, k=10, method=method, n_impressions=400, seed=1)
    assert res.wins_a + res.wins_b + res.ties == 400
    assert res.delta > 0.25 and res.p_value < 1e-10 and res.preferred == "A"
    assert res.ci_low < res.delta < res.ci_high
    flipped = simulate_interleaving(rand, good, rel, k=10, method=method, n_impressions=400, seed=1)
    assert flipped.preferred == "B"


@pytest.mark.parametrize("method", METHODS)
def test_simulation_unbiased_under_random_clicks(method):
    rel, good, rand = _synthetic(seed=2)

    def random_clicks(ranking, relevant, rng):
        return [i for i in range(len(ranking)) if rng.random() < 0.2]

    res = simulate_interleaving(good, rand, rel, k=10, method=method, n_impressions=3000,
                                click_model=random_clicks, seed=5)
    assert abs(res.delta) < 0.03


def test_simulation_validation():
    rel, good, rand = _synthetic(n_users=3)
    with pytest.raises(ValueError):
        simulate_interleaving(good, rand[:2], rel)
    with pytest.raises(ValueError):
        simulate_interleaving(good, rand, rel, method="optimized")
    with pytest.raises(ValueError):
        simulate_interleaving(good, rand, rel, click_model="ubm")
    with pytest.raises(ValueError):
        simulate_interleaving(good, rand, rel, n_impressions=0)


def test_cli_interleave_section(tmp_path, capsys):
    out = tmp_path / "report.md"
    args = ["--n-users", "10", "--n-items", "20", "--n-interactions", "80", "--k", "5", "--seed", "0",
            "--output", str(out), "--interleave", "popularity,item-knn(cosine)",
            "--interleave-impressions", "50", "--interleave-method", "team-draft"]
    assert main(args) == 0
    text = out.read_text()
    assert "## Online interleaving: popularity (A) vs item-knn(cosine) (B)" in text
    assert "| team-draft | 50 |" in text
    assert "interleaving[team-draft]" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(args[:-6] + ["--interleave", "popularity,nope"])
