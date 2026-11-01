"""Online interleaving comparisons of two rankers, with simulated clicks.

Interleaving merges two rankings A and B into one list for each user,
records which items the user clicks, and credits each click to one of
the rankers. In online A/B testing it reaches a confident preference
with one to two orders of magnitude fewer impressions than comparing
absolute click metrics (Chapelle, Joachims, Radlinski & Yue, TOIS 2012).
This module offers three methods:

* :func:`balanced_interleave` (Joachims 2002). Two pointers advance
  through A and B so that every prefix of the merged list holds the
  top-``ka`` of A and the top-``kb`` of B with ``|ka - kb| <= 1``. The
  credit cut-off is the shallower of the two ranks of the lowest click.
* :func:`team_draft_interleave` (Radlinski, Kurup & Joachims 2008). As
  in a schoolyard team draft, the team with fewer picks (a coin flip
  breaks ties) adds its highest-ranked item that is not taken yet.
  Each click goes to the team that contributed the item.
* :func:`probabilistic_interleave` (Hofmann, Whiteson & de Rijke 2011).
  Each ranker becomes a softmax ``P(d) ∝ rank(d)^-tau``. A coin picks
  the ranker for every slot, and an item is sampled from it and removed
  from both. With ``marginalize=True`` the outcome is the exact
  expectation over every team assignment that could have produced the
  observed list. This has lower variance, and the remaining set after
  each slot depends only on the list prefix, so the posterior factorises
  per position and is computed exactly without enumeration.

Clicks come from :func:`cascade_clicks` (a dependent click model: scan
top-down, click relevant items with ``p_rel`` and others with
``p_nonrel``, stop after a click with ``p_stop``) or from
:func:`position_based_clicks` (the examination probability decays with
rank). :func:`simulate_interleaving` runs many impressions and reports
wins, losses and ties, plus ``delta = (wins + ties/2)/n - 0.5`` (> 0
means A is preferred), a confidence interval and a two-sided sign-test
p-value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Hashable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import binomtest

Item = Hashable
METHODS = ("team-draft", "balanced", "probabilistic")


def _rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _check_k(k: int) -> int:
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    return int(k)


def _dedupe(ranking: Sequence[Item]) -> List[Item]:
    seen: Set[Item] = set()
    out = []
    for item in ranking:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# --------------------------------------------------------------------------- merging


def balanced_interleave(
    ranking_a: Sequence[Item], ranking_b: Sequence[Item], k: int, rng=None, a_first: Optional[bool] = None
) -> Tuple[List[Item], bool]:
    """Balanced interleaving. Returns ``(merged, a_first)``."""
    k = _check_k(k)
    a, b = _dedupe(ranking_a), _dedupe(ranking_b)
    if a_first is None:
        a_first = bool(_rng(rng).random() < 0.5)
    merged: List[Item] = []
    taken: Set[Item] = set()
    ka = kb = 0
    while len(merged) < k and (ka < len(a) or kb < len(b)):
        use_a = (ka < kb or (ka == kb and a_first)) if (ka < len(a) and kb < len(b)) else ka < len(a)
        if use_a:
            item = a[ka]
            ka += 1
        else:
            item = b[kb]
            kb += 1
        if item not in taken:
            taken.add(item)
            merged.append(item)
    return merged, a_first


def team_draft_interleave(
    ranking_a: Sequence[Item], ranking_b: Sequence[Item], k: int, rng=None
) -> Tuple[List[Item], List[str]]:
    """Team-draft interleaving. Returns ``(merged, teams)``, where ``teams[i]`` is ``"A"`` or ``"B"``."""
    k = _check_k(k)
    gen = _rng(rng)
    lists = {"A": _dedupe(ranking_a), "B": _dedupe(ranking_b)}
    ptr = {"A": 0, "B": 0}
    size = {"A": 0, "B": 0}
    merged: List[Item] = []
    teams: List[str] = []
    taken: Set[Item] = set()

    def next_free(team):
        lst = lists[team]
        while ptr[team] < len(lst) and lst[ptr[team]] in taken:
            ptr[team] += 1
        return lst[ptr[team]] if ptr[team] < len(lst) else None

    while len(merged) < k:
        if size["A"] < size["B"] or (size["A"] == size["B"] and gen.random() < 0.5):
            order = ("A", "B")
        else:
            order = ("B", "A")
        for team in order:
            item = next_free(team)
            if item is not None:
                break
        else:
            break
        taken.add(item)
        merged.append(item)
        teams.append(team)
        size[team] += 1
    return merged, teams


def _softmax_weights(ranking: List[Item], pool: List[Item], tau: float) -> dict:
    pos = {item: i + 1 for i, item in enumerate(ranking)}
    missing_rank = len(ranking) + 1
    return {item: float(pos.get(item, missing_rank)) ** (-tau) for item in pool}


def probabilistic_interleave(
    ranking_a: Sequence[Item], ranking_b: Sequence[Item], k: int, tau: float = 3.0, rng=None
) -> Tuple[List[Item], List[str], List[Tuple[float, float]]]:
    """Probabilistic interleaving. Returns ``(merged, teams, slot_probs)``.

    ``slot_probs[i] = (P_A(l_i | remaining), P_B(l_i | remaining))`` are
    the probabilities that each ranker's softmax would draw the item
    shown in slot ``i``. :func:`interleaving_outcome` uses them to
    marginalise over team assignments. Items missing from a ranking
    share rank ``len(ranking) + 1``.
    """
    k = _check_k(k)
    if tau <= 0:
        raise ValueError("tau must be positive")
    gen = _rng(rng)
    a, b = _dedupe(ranking_a), _dedupe(ranking_b)
    pool = _dedupe(a + b)
    wa, wb = _softmax_weights(a, pool, tau), _softmax_weights(b, pool, tau)
    remaining = list(pool)
    merged, teams, probs = [], [], []
    while remaining and len(merged) < k:
        za = sum(wa[d] for d in remaining)
        zb = sum(wb[d] for d in remaining)
        team = "A" if gen.random() < 0.5 else "B"
        weights = np.array([(wa if team == "A" else wb)[d] for d in remaining])
        item = remaining[int(gen.choice(len(remaining), p=weights / weights.sum()))]
        merged.append(item)
        teams.append(team)
        probs.append((wa[item] / za, wb[item] / zb))
        remaining.remove(item)
    return merged, teams, probs


# --------------------------------------------------------------------------- credit


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def balanced_outcome(ranking_a, ranking_b, merged, clicked_positions) -> int:
    """+1 if A wins, -1 if B wins, 0 for a tie (Joachims' balanced credit rule)."""
    clicks = sorted(set(clicked_positions))
    if not clicks:
        return 0
    a, b = _dedupe(ranking_a), _dedupe(ranking_b)
    clicked = {merged[i] for i in clicks}
    lowest = merged[clicks[-1]]
    ra = a.index(lowest) + 1 if lowest in a else math.inf
    rb = b.index(lowest) + 1 if lowest in b else math.inf
    cut = min(ra, rb)
    ha = sum(1 for item in a[: int(min(cut, len(a)))] if item in clicked)
    hb = sum(1 for item in b[: int(min(cut, len(b)))] if item in clicked)
    return _sign(ha - hb)


def team_draft_outcome(teams: Sequence[str], clicked_positions) -> int:
    """+1 / -1 / 0 by comparing the clicks credited to each team."""
    ca = sum(1 for i in set(clicked_positions) if teams[i] == "A")
    cb = sum(1 for i in set(clicked_positions) if teams[i] == "B")
    return _sign(ca - cb)


def probabilistic_outcome(slot_probs, clicked_positions, teams=None, marginalize: bool = True) -> float:
    """Outcome of a probabilistic interleave, in [-1, 1].

    With ``marginalize=True`` this is ``E[sign(c_A - c_B) | list, clicks]``
    over team assignments. Given the list, each slot belongs to A
    independently with probability ``P_A / (P_A + P_B)``, so the
    distribution of ``c_A - c_B`` is an exact convolution over the
    clicked slots. Otherwise the sampled ``teams`` are used, as in
    team-draft.
    """
    clicks = sorted(set(clicked_positions))
    if not marginalize:
        if teams is None:
            raise ValueError("teams are required when marginalize=False")
        return float(team_draft_outcome(teams, clicks))
    dist = {0: 1.0}
    for i in clicks:
        pa, pb = slot_probs[i]
        qa = pa / (pa + pb)
        new: dict = {}
        for diff, prob in dist.items():
            new[diff + 1] = new.get(diff + 1, 0.0) + prob * qa
            new[diff - 1] = new.get(diff - 1, 0.0) + prob * (1.0 - qa)
        dist = new
    return float(sum(prob * _sign(diff) for diff, prob in dist.items()))


# --------------------------------------------------------------------------- click models


def cascade_clicks(
    ranking: Sequence[Item],
    relevant: Set[Item],
    p_rel: float = 0.9,
    p_nonrel: float = 0.05,
    p_stop: float = 0.6,
    rng=None,
) -> List[int]:
    """Clicked positions under a dependent cascade click model."""
    for name, value in (("p_rel", p_rel), ("p_nonrel", p_nonrel), ("p_stop", p_stop)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    gen = _rng(rng)
    clicks = []
    for i, item in enumerate(ranking):
        if gen.random() < (p_rel if item in relevant else p_nonrel):
            clicks.append(i)
            if gen.random() < p_stop:
                break
    return clicks


def position_based_clicks(
    ranking: Sequence[Item],
    relevant: Set[Item],
    eta: float = 1.0,
    p_rel: float = 0.9,
    p_nonrel: float = 0.05,
    rng=None,
) -> List[int]:
    """Clicked positions under a position-based model: P(examine rank r) = r^-eta."""
    if eta < 0:
        raise ValueError("eta must be non-negative")
    gen = _rng(rng)
    clicks = []
    for i, item in enumerate(ranking):
        if gen.random() < (i + 1) ** (-eta) and gen.random() < (p_rel if item in relevant else p_nonrel):
            clicks.append(i)
    return clicks


# --------------------------------------------------------------------------- simulation


@dataclass(frozen=True)
class InterleavingResult:
    method: str
    impressions: int
    wins_a: int
    wins_b: int
    ties: int
    delta: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def preferred(self) -> str:
        if self.p_value >= 0.05 or self.delta == 0:
            return "none"
        return "A" if self.delta > 0 else "B"


def interleave_once(ranking_a, ranking_b, relevant, k, method, click_fn, rng, tau=3.0, marginalize=True) -> float:
    """Run one impression and return its outcome in [-1, 1] (> 0 favours A)."""
    if method == "team-draft":
        merged, teams = team_draft_interleave(ranking_a, ranking_b, k, rng)
        return float(team_draft_outcome(teams, click_fn(merged, relevant, rng)))
    if method == "balanced":
        merged, _ = balanced_interleave(ranking_a, ranking_b, k, rng)
        return float(balanced_outcome(ranking_a, ranking_b, merged, click_fn(merged, relevant, rng)))
    if method == "probabilistic":
        merged, teams, probs = probabilistic_interleave(ranking_a, ranking_b, k, tau, rng)
        return probabilistic_outcome(probs, click_fn(merged, relevant, rng), teams, marginalize)
    raise ValueError(f"method must be one of {METHODS}")


def simulate_interleaving(
    lists_a: Sequence[Sequence[Item]],
    lists_b: Sequence[Sequence[Item]],
    relevant_lists: Sequence[Set[Item]],
    k: int = 10,
    method: str = "team-draft",
    n_impressions: int = 2000,
    click_model: str | Callable = "cascade",
    tau: float = 3.0,
    marginalize: bool = True,
    seed: int | None = 0,
) -> InterleavingResult:
    """Simulate ``n_impressions`` interleaved impressions for randomly drawn users.

    ``click_model`` is ``"cascade"``, ``"pbm"`` or a callable
    ``(ranking, relevant, rng) -> clicked positions``.
    """
    k = _check_k(k)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not (len(lists_a) == len(lists_b) == len(relevant_lists)) or not lists_a:
        raise ValueError("lists_a, lists_b and relevant_lists must be non-empty and aligned")
    if n_impressions < 1:
        raise ValueError("n_impressions must be positive")
    if click_model == "cascade":
        click_fn = lambda r, rel, g: cascade_clicks(r, rel, rng=g)  # noqa: E731
    elif click_model == "pbm":
        click_fn = lambda r, rel, g: position_based_clicks(r, rel, rng=g)  # noqa: E731
    elif callable(click_model):
        click_fn = click_model
    else:
        raise ValueError("click_model must be 'cascade', 'pbm' or a callable")
    gen = np.random.default_rng(seed)
    users = gen.integers(0, len(lists_a), size=n_impressions)
    outcomes = np.array(
        [
            interleave_once(
                lists_a[u], lists_b[u], set(relevant_lists[u]), k, method, click_fn, gen, tau, marginalize
            )
            for u in users
        ]
    )
    wins_a = int(np.sum(outcomes > 0))
    wins_b = int(np.sum(outcomes < 0))
    ties = n_impressions - wins_a - wins_b
    delta = float(outcomes.mean() / 2.0)
    half = 1.96 * float(outcomes.std(ddof=1)) / math.sqrt(n_impressions) / 2.0 if n_impressions > 1 else math.inf
    p_value = binomtest(wins_a, wins_a + wins_b, 0.5).pvalue if wins_a + wins_b else 1.0
    return InterleavingResult(
        method, n_impressions, wins_a, wins_b, ties, delta, delta - half, delta + half, float(p_value)
    )


def render_interleaving_section(results: dict, name_a: str, name_b: str, k: int, click_model: str) -> str:
    """Markdown section for ``{method: InterleavingResult}``."""
    lines = [
        f"## Online interleaving: {name_a} (A) vs {name_b} (B)",
        "",
        f"Simulated {click_model} clicks on interleaved top-{k} lists. "
        "delta = (wins_A + ties/2)/n - 0.5; > 0 favours A. p-value: two-sided sign test.",
        "",
        "| method | impressions | A wins | B wins | ties | delta | 95% CI | p-value | preferred |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for method, r in results.items():
        pref = {"A": name_a, "B": name_b}.get(r.preferred, "none")
        lines.append(
            f"| {method} | {r.impressions} | {r.wins_a} | {r.wins_b} | {r.ties} | {r.delta:+.4f} | "
            f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.p_value:.3g} | {pref} |"
        )
    return "\n".join(lines) + "\n"
