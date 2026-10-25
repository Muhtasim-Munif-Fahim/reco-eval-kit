import math

import pytest

from reco_eval_kit.cli import main
from reco_eval_kit.rank_similarity import (
    compare_rankings,
    jaccard_at_k,
    kendall_tau_topk_distance,
    overlap_at_k,
    pairwise_rank_similarity,
    rank_biased_overlap,
)


def _full_rbo(s, t, p, depth=4000):
    """RBO of two (long) explicit rankings by direct summation."""
    total, seen_s, seen_t = 0.0, set(), set()
    for d in range(1, depth + 1):
        seen_s.add(s[d - 1])
        seen_t.add(t[d - 1])
        total += (1 - p) * p ** (d - 1) * len(seen_s & seen_t) / d
    return total


def test_rbo_identical_and_disjoint():
    same = rank_biased_overlap(list("abcdefg"), list("abcdefg"), p=0.9)
    assert same.ext == pytest.approx(1.0)
    assert same.min < same.ext <= same.max + 1e-12
    disjoint = rank_biased_overlap(list("abcde"), list("fghij"), p=0.9)
    assert disjoint.min == pytest.approx(0.0)
    assert disjoint.ext == pytest.approx(0.0)
    assert rank_biased_overlap([], []).ext == 1.0


def test_rbo_ext_hand_computed():
    # A_1 = 0, A_2..5 = 1  ->  (1-p) * sum_{d=2}^5 p^(d-1) + p^5 = 0.9 for p = 0.9
    r = rank_biased_overlap(list("abcde"), list("bacde"), p=0.9)
    assert r.ext == pytest.approx(0.9)
    # min matches Webber eq. 11 directly
    p, xs = 0.9, [0, 2, 3, 4, 5]
    expected = (1 - p) / p * (
        sum((x - 5) / d * p ** d for d, x in enumerate(xs, start=1)) - 5 * math.log(1 - p)
    )
    assert r.min == pytest.approx(expected)


@pytest.mark.parametrize(
    "s, t, best_s, best_t, worst_s, worst_t",
    [
        ("abcde", "abxyz", "abcdexyz", "abxyzcde", None, None),
        ("abcde", "fghij", "abcdefghij", "fghijabcde", None, None),
        ("abcd", "abcdefg", "abcdefg", "abcdefg", None, None),
    ],
)
def test_rbo_bounds_are_tight(s, t, best_s, best_t, worst_s, worst_t):
    p = 0.85
    r = rank_biased_overlap(list(s), list(t), p=p)
    shared = [f"z{i}" for i in range(5000)]
    best = _full_rbo(list(best_s) + shared, list(best_t) + shared, p)
    worst = _full_rbo(
        list(s) + [f"s{i}" for i in range(5000)], list(t) + [f"t{i}" for i in range(5000)], p
    )
    assert r.max == pytest.approx(best, abs=1e-9)
    assert r.min == pytest.approx(worst, abs=1e-9)
    assert r.min - 1e-12 <= r.ext <= r.max + 1e-12


def test_rbo_symmetric_truncation_and_validation():
    a, b = list("abcdefgh"), list("bdacxyeg")
    assert rank_biased_overlap(a, b, p=0.8).ext == pytest.approx(rank_biased_overlap(b, a, p=0.8).ext)
    assert rank_biased_overlap(a, b, k=3) == rank_biased_overlap(a[:3], b[:3])
    with pytest.raises(ValueError):
        rank_biased_overlap(a, b, p=1.0)
    with pytest.raises(ValueError):
        rank_biased_overlap(["a", "a"], ["a"])
    # higher p looks deeper: a list that agrees only in the tail scores higher
    tail_agree = (list("abcdef"), list("xyzdef"))
    assert rank_biased_overlap(*tail_agree, p=0.95).ext > rank_biased_overlap(*tail_agree, p=0.5).ext


def test_set_overlaps():
    assert overlap_at_k("abcd", "abxy", 4) == 0.5
    assert jaccard_at_k("abcd", "abxy", 4) == pytest.approx(2 / 6)
    assert jaccard_at_k([], [], 3) == 1.0
    assert jaccard_at_k("abcz", "abcy", 3) == 1.0


def test_kendall_topk_distance_cases():
    assert kendall_tau_topk_distance("abc", "abc") == 0.0
    assert kendall_tau_topk_distance("abc", "xyz") == 1.0
    # full reversal of 3 shared items: 3 discordant of the worst 9 + 0.5 * 6
    assert kendall_tau_topk_distance("abc", "cba") == pytest.approx(3 / 12)
    # case 2: a,b in s; only b in t -> t implicitly ranks b ahead of a: cost 1
    # case 3: c (s only) vs x (t only): cost 1; case 4 pairs with penalty
    d0 = kendall_tau_topk_distance("ab", "bx", penalty=0.0)
    d1 = kendall_tau_topk_distance("ab", "bx", penalty=1.0)
    # pairs: (a,b) case 2 -> 1; (a,x) case 3 -> 1; (b,x) both in t, b only in s, s ranks b, t ranks b ahead -> 0
    assert d0 == pytest.approx(2 / 4)
    assert d1 == pytest.approx(2 / 6)
    with pytest.raises(ValueError):
        kendall_tau_topk_distance("ab", "ab", penalty=2)


def test_compare_and_pairwise_matrix():
    a = [list("abcde"), list("vwxyz")]
    b = [list("abcde"), list("zyxwv")]
    c = [list("fghij"), list("klmno")]
    out = compare_rankings(a, b, k=5)
    assert out["Jaccard@K"] == 1.0
    assert 0.5 < out["RBO_ext"] < 1.0
    assert out["RBO_min"] <= out["RBO_ext"]
    mat = pairwise_rank_similarity({"a": a, "b": b, "c": c}, k=5)
    assert (mat.values.diagonal() == 1.0).all()
    assert mat.loc["a", "b"] == mat.loc["b", "a"] > mat.loc["a", "c"] == 0.0
    kd = pairwise_rank_similarity({"a": a, "c": c}, k=5, metric="KendallTopK")
    assert kd.loc["a", "c"] == 1.0 and kd.loc["a", "a"] == 0.0
    with pytest.raises(ValueError):
        compare_rankings(a, b[:1])
    with pytest.raises(ValueError):
        pairwise_rank_similarity({"a": a}, metric="nope")


def test_cli_rank_similarity_section(tmp_path, capsys):
    output = tmp_path / "report.md"
    args = ["--n-users", "8", "--n-items", "12", "--n-interactions", "40", "--k", "3",
            "--seed", "0", "--output", str(output), "--rank-similarity", "0.8"]
    assert main(args) == 0
    text = output.read_text(encoding="utf-8")
    assert "## Ranking similarity between models (RBO@3, p=0.8)" in text
    assert "| popularity |" in text and "Most similar pairs:" in text
    with pytest.raises(SystemExit):
        main(args[:-1] + ["1.5"])
