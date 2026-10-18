"""Tests for MMR and greedy-DPP diversity re-ranking."""

import numpy as np
import pandas as pd
import pytest

from reco_eval_kit.beyond_accuracy import intra_list_diversity
from reco_eval_kit.cli import main
from reco_eval_kit.diversify import cosine_similarity_matrix, dpp_rerank, mmr_rerank, rank_scores


def _clusters():
    # items a1..a3 nearly identical (cluster A), b1 (cluster B), c1 (cluster C)
    feats = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.99, 0.05, 0.0], [0.98, 0.0, 0.05], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        index=["a1", "a2", "a3", "b1", "c1"],
    )
    scores = [1.0, 0.95, 0.9, 0.6, 0.5]
    return list(feats.index), scores, feats


def test_rank_scores_and_cosine_matrix():
    assert rank_scores(4) == [1.0, 0.75, 0.5, 0.25]
    assert rank_scores(0) == []
    feats = pd.DataFrame([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], index=[10, 11, 12])
    S = cosine_similarity_matrix([10, 11, 12], feats)
    assert np.allclose(S, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="missing"):
        cosine_similarity_matrix([10, 99], feats)


def test_mmr_extremes_and_diversification():
    items, scores, feats = _clusters()
    assert mmr_rerank(items, scores, 3, lam=1.0, item_features=feats) == ["a1", "a2", "a3"]
    diverse = mmr_rerank(items, scores, 3, lam=0.5, item_features=feats)
    assert diverse == ["a1", "b1", "c1"]
    assert intra_list_diversity([diverse], feats) > intra_list_diversity([["a1", "a2", "a3"]], feats)


def test_mmr_matches_bruteforce_objective_each_step():
    rng = np.random.default_rng(0)
    feats = pd.DataFrame(rng.normal(size=(15, 5)), index=range(15))
    items, scores = list(range(15)), rng.uniform(size=15)
    lam = 0.6
    out = mmr_rerank(items, scores, 6, lam=lam, item_features=feats)
    S = cosine_similarity_matrix(items, feats)
    rel = (scores - scores.min()) / (scores.max() - scores.min())
    chosen = []
    for _ in range(6):
        def value(i):
            pen = max(S[i, j] for j in chosen) if chosen else 0.0
            return lam * rel[i] - (1 - lam) * pen
        chosen.append(max((i for i in items if i not in chosen), key=value))
    assert out == chosen


def test_dpp_matches_bruteforce_greedy_logdet():
    rng = np.random.default_rng(1)
    feats = pd.DataFrame(rng.normal(size=(12, 8)), index=[f"i{j}" for j in range(12)])
    items, scores = list(feats.index), rng.uniform(size=12)
    theta = 0.6
    out = dpp_rerank(items, scores, 5, theta=theta, item_features=feats)
    S = cosine_similarity_matrix(items, feats)
    rel = (scores - scores.min()) / (scores.max() - scores.min())
    q = np.exp(theta / (2 * (1 - theta)) * rel)
    L = q[:, None] * S * q[None, :]
    sel = []
    for _ in range(5):
        sel.append(max((i for i in range(12) if i not in sel),
                       key=lambda i: np.linalg.slogdet(L[np.ix_(sel + [i], sel + [i])])[1]))
    assert out == [items[i] for i in sel]


def test_dpp_diversifies_and_fills_after_duplicates():
    items, scores, feats = _clusters()
    assert dpp_rerank(items, scores, 3, theta=0.3, item_features=feats) == ["a1", "b1", "c1"]
    # a high theta leans toward relevance
    assert dpp_rerank(items, scores, 2, theta=0.95, item_features=feats)[0] == "a1"
    # exact duplicates add no volume -> completed in relevance order
    S = np.ones((3, 3))
    assert dpp_rerank(["x", "y", "z"], [0.2, 0.9, 0.5], 3, similarity=S) == ["y", "z", "x"]


def test_precomputed_similarity_and_validation():
    S = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert mmr_rerank(["p", "q", "r"], [1.0, 0.9, 0.8], 2, lam=0.5, similarity=S) == ["p", "r"]
    assert mmr_rerank(["p"], [1.0], 5, similarity=np.ones((1, 1))) == ["p"]  # k capped
    assert dpp_rerank([], [], 3, similarity=np.zeros((0, 0))) == []
    with pytest.raises(ValueError, match="exactly one"):
        mmr_rerank(["p"], [1.0], 1)
    with pytest.raises(ValueError, match="same length"):
        mmr_rerank(["p", "q"], [1.0], 1, similarity=S[:2, :2])
    with pytest.raises(ValueError, match="unique"):
        mmr_rerank(["p", "p"], [1.0, 0.5], 1, similarity=np.eye(2))
    with pytest.raises(ValueError, match="lam"):
        mmr_rerank(["p"], [1.0], 1, lam=1.5, similarity=np.eye(1))
    with pytest.raises(ValueError, match="theta"):
        dpp_rerank(["p"], [1.0], 1, theta=1.0, similarity=np.eye(1))
    with pytest.raises(ValueError, match="symmetric"):
        dpp_rerank(["p", "q"], [1.0, 0.5], 1, similarity=np.array([[1.0, 0.2], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="k must"):
        dpp_rerank(["p"], [1.0], 0, similarity=np.eye(1))


@pytest.mark.parametrize("method", ["mmr", "dpp"])
def test_cli_diversify_adds_reranked_models(tmp_path, method):
    output = tmp_path / "report.md"
    args = ["--n-users", "10", "--n-items", "40", "--n-interactions", "200", "--k", "4",
            "--seed", "0", "--output", str(output), "--diversify", method, "--diversify-lambda", "0.5"]
    assert main(args) == 0
    text = output.read_text(encoding="utf-8")
    assert f"popularity+{method}" in text
    assert f"{method}, lambda=0.5, pool=3xk" in text


def test_cli_diversify_rejects_bad_lambda():
    with pytest.raises(SystemExit):
        main(["--diversify", "dpp", "--diversify-lambda", "1.0"])
