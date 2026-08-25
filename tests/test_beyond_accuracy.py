import math

import pandas as pd
import pytest

from reco_eval_kit.beyond_accuracy import (
    catalog_coverage,
    intra_list_diversity,
    novelty,
)


FEATURES = pd.DataFrame(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    index=[10, 20, 30],
    columns=["f0", "f1"],
)


def test_coverage_counts_distinct_recommended_items_once():
    assert catalog_coverage([[1, 2], [2, 3]], catalog=[1, 2, 3, 4]) == pytest.approx(0.75)


def test_coverage_accepts_integer_catalog_size():
    assert catalog_coverage([[1, 2], [2, 3]], catalog=4) == pytest.approx(0.75)


def test_coverage_is_zero_without_recommendations():
    assert catalog_coverage([[], []], catalog=[1, 2]) == 0.0


def test_coverage_rejects_empty_catalog():
    with pytest.raises(ValueError):
        catalog_coverage([[1]], catalog=[])


def test_rare_items_carry_more_novelty_than_blockbusters():
    popularity = {1: 90, 2: 10}
    rare = novelty([[2]], popularity)
    common = novelty([[1]], popularity)
    assert rare == pytest.approx(-math.log2(0.1))
    assert common == pytest.approx(-math.log2(0.9))
    assert rare > common


def test_unseen_items_receive_floor_probability_novelty():
    popularity = {1: 100}
    assert novelty([[99]], popularity) == pytest.approx(-math.log2(1e-6))


def test_novelty_requires_at_least_one_item():
    with pytest.raises(ValueError):
        novelty([], {1: 1})


def test_intra_list_diversity_of_orthogonal_items_is_one():
    assert intra_list_diversity([[10, 20]], FEATURES) == pytest.approx(1.0)


def test_intra_list_diversity_of_collinear_items_is_zero():
    assert intra_list_diversity([[10, 30]], pd.DataFrame(
        [[1.0, 0.0], [2.0, 0.0]], index=[10, 30], columns=["f0", "f1"]
    )) == pytest.approx(0.0)


def test_intra_list_diversity_pools_pairs_across_lists():
    # pair (10, 20): distance 1; pair (30, 30-copy): distance 0
    features = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
        index=[10, 20, 30, 31],
        columns=["f0", "f1"],
    )
    assert intra_list_diversity([[10, 20], [30, 31]], features) == pytest.approx(0.5)


def test_intra_list_diversity_of_singletons_is_zero():
    assert intra_list_diversity([[10], [20]], FEATURES) == 0.0


def test_intra_list_diversity_reports_missing_feature_rows():
    with pytest.raises(ValueError):
        intra_list_diversity([[10, 999]], FEATURES)