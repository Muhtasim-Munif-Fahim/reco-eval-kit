import math

import numpy as np
import pytest

from reco_eval_kit.cli import main
from reco_eval_kit.metrics import mean_metric, ndcg_at_k, recall_at_k
from reco_eval_kit.unbiased import (
    mean_unbiased_metric_at_k,
    power_law_propensities,
    unbiased_user_metric,
)


def test_power_law_propensities_shape_and_clipping():
    props = power_law_propensities({"a": 100, "b": 25, "c": 0, "d": 1}, gamma=1.0, min_propensity=0.05)
    assert props["a"] == pytest.approx(1.0)
    assert props["b"] == pytest.approx(0.25)  # exponent (1+1)/2 = 1
    assert props["c"] == pytest.approx(0.05)
    assert props["d"] == pytest.approx(0.05)  # 0.01 clipped up
    sqrt_props = power_law_propensities({"a": 100, "b": 25}, gamma=0.0)
    assert sqrt_props["b"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        power_law_propensities({"a": 1}, max_propensity=0.0)
    with pytest.raises(ValueError):
        power_law_propensities({"a": -1})


def test_user_metric_values():
    props = {1: 0.5, 2: 0.25, 3: 1.0}
    est = unbiased_user_metric([1, 9, 2], [1, 2, 3], props, k=3, gain="hit")
    # hits: 1 (w=2), 2 (w=4); miss: 3 (w=1)
    assert est.naive == pytest.approx(2 / 3)
    assert est.ips == pytest.approx((2 + 4) / 3)
    assert est.snips == pytest.approx(6 / 7)
    dcg = unbiased_user_metric([1, 9, 2], [1, 2, 3], props, k=3, gain="dcg")
    assert dcg.naive == pytest.approx((1 + 1 / math.log2(4)) / 3)
    rr = unbiased_user_metric([1, 9, 2], [2], props, k=2, gain="rr")
    assert rr.naive == 0.0  # rank 3 beyond k
    assert unbiased_user_metric([1], [], props, k=1) is None


def test_uniform_propensities_reduce_to_recall():
    rng = np.random.default_rng(0)
    recs = [list(rng.permutation(50)[:10]) for _ in range(30)]
    rels = [set(rng.choice(50, 4, replace=False).tolist()) for _ in range(30)]
    props = {i: 0.3 for i in range(50)}
    est = mean_unbiased_metric_at_k(recs, rels, props, k=10)
    assert est.snips == pytest.approx(mean_metric(recall_at_k, recs, rels, k=10))
    assert est.naive == pytest.approx(est.snips)
    assert est.ips == pytest.approx(est.naive / 0.3)
    assert est.effective_sample_size == pytest.approx(est.n_observed)


def _mnar_world(seed=0, n_users=3000, n_items=200):
    """True relevance is popularity-neutral; observation is popularity-biased."""
    rng = np.random.default_rng(seed)
    exposure_count = {i: float(n_items - i) ** 2 for i in range(n_items)}
    props = power_law_propensities(exposure_count, gamma=1.0, min_propensity=0.02)
    true_rel, observed = [], []
    for _ in range(n_users):
        rel = set(rng.choice(n_items, 10, replace=False).tolist())
        obs = {i for i in rel if rng.uniform() < props[i]}
        true_rel.append(rel)
        observed.append(obs)
    popular_recs = [list(range(20)) for _ in range(n_users)]  # always top-popular
    return props, true_rel, observed, popular_recs


def test_snips_corrects_popularity_bias_toward_truth():
    props, true_rel, observed, recs = _mnar_world()
    truth = mean_metric(recall_at_k, recs, true_rel, k=20)
    est = mean_unbiased_metric_at_k(recs, observed, props, k=20)
    assert truth == pytest.approx(0.1, abs=0.01)  # 20 of 200 items
    assert est.naive > 2 * truth  # naive evaluation flatters the popular recommender
    assert abs(est.snips - truth) < 0.03
    assert est.effective_sample_size < est.n_observed


def test_leave_one_out_snips_is_not_naive():
    props = {"pop": 1.0, "rare": 0.1}
    recs = [["pop"], ["pop"]]
    obs = [["pop"], ["rare"]]
    est = mean_unbiased_metric_at_k(recs, obs, props, k=1)
    assert est.naive == pytest.approx(0.5)
    assert est.snips == pytest.approx(1.0 / (1.0 + 10.0))


def test_errors_and_defaults():
    with pytest.raises(KeyError):
        unbiased_user_metric([1], [2], {}, k=1)
    assert unbiased_user_metric([1], [2], {}, k=1, default_propensity=0.5).naive == 0.0
    with pytest.raises(ValueError):
        unbiased_user_metric([1], [1], {1: 0.0}, k=1)
    with pytest.raises(ValueError):
        unbiased_user_metric([1], [1], {1: 1.0}, k=1, gain="map")
    with pytest.raises(ValueError):
        mean_unbiased_metric_at_k([[1]], [], {}, k=1)
    empty = mean_unbiased_metric_at_k([[1]], [[]], {}, k=1)
    assert empty.n_users == 0 and empty.snips == 0.0


def test_cli_adds_snips_metrics(tmp_path, capsys):
    out = tmp_path / "report.md"
    code = main([
        "--n-users", "30", "--n-items", "60", "--n-interactions", "600",
        "--k", "5", "--output", str(out), "--propensity-gamma", "0.5",
    ])
    assert code == 0
    text = out.read_text()
    assert "SNIPS-Recall@K" in text and "SNIPS-NDCG@K" in text
    assert "power law, gamma=0.5" in text
