import math

import pandas as pd
import pytest

from reco_eval_kit.beyond_accuracy import (
    catalog_coverage,
    intra_list_diversity,
    novelty,
    profile_unexpectedness,
    serendipity,
    unexpectedness,
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


def test_rare_items_are_more_unexpected_than_blockbusters():
    popularity = {1: 90, 2: 10}
    rare = unexpectedness([[2]], popularity)
    common = unexpectedness([[1]], popularity)
    assert rare == pytest.approx(0.9)
    assert common == pytest.approx(0.1)
    assert rare > common


def test_unseen_items_are_fully_unexpected():
    assert unexpectedness([[99]], {1: 100}) == pytest.approx(1.0)


def test_unexpectedness_pools_entries_across_lists():
    popularity = {1: 80, 2: 20}
    assert unexpectedness([[1], [2]], popularity) == pytest.approx(0.5)


def test_unexpectedness_requires_at_least_one_item():
    with pytest.raises(ValueError):
        unexpectedness([], {1: 1})


def test_profile_unexpectedness_of_orthogonal_item_is_one():
    # history item 10 = [1, 0]; recommended 20 = [0, 1]
    assert profile_unexpectedness([[20]], [[10]], FEATURES) == pytest.approx(1.0)


def test_profile_unexpectedness_of_collinear_item_is_zero():
    features = pd.DataFrame(
        [[1.0, 0.0], [2.0, 0.0]],
        index=[10, 30],
        columns=["f0", "f1"],
    )
    assert profile_unexpectedness([[30]], [[10]], features) == pytest.approx(0.0)


def test_empty_history_is_fully_unexpected():
    assert profile_unexpectedness([[10]], [[]], FEATURES) == pytest.approx(1.0)


def test_profile_unexpectedness_requires_at_least_one_item():
    with pytest.raises(ValueError):
        profile_unexpectedness([[]], [[]], FEATURES)


def test_profile_unexpectedness_requires_aligned_histories():
    with pytest.raises(ValueError):
        profile_unexpectedness([[10]], [], FEATURES)


def test_profile_unexpectedness_reports_missing_feature_rows():
    with pytest.raises(ValueError):
        profile_unexpectedness([[999]], [[10]], FEATURES)
    with pytest.raises(ValueError):
        profile_unexpectedness([[10]], [[999]], FEATURES)


def test_serendipity_is_unexpectedness_of_relevant_items_only():
    popularity = {1: 75, 2: 25}
    # item 2 is relevant and unexpectedness 0.75; item 1 is irrelevant
    score = serendipity([[1, 2]], [{2}], item_popularity=popularity)
    assert score == pytest.approx(0.375)


def test_serendipity_is_zero_when_hits_are_fully_expected():
    popularity = {1: 100}
    assert serendipity([[1]], [{1}], item_popularity=popularity) == pytest.approx(0.0)


def test_serendipity_is_zero_for_unexpected_but_irrelevant_items():
    popularity = {1: 100}
    assert serendipity([[99]], [{1}], item_popularity=popularity) == pytest.approx(0.0)


def test_serendipity_uses_k_as_denominator():
    popularity = {1: 0, 2: 100}
    # one relevant unexpected item in two slots
    assert serendipity([[1]], [{1}], item_popularity=popularity, k=2) == pytest.approx(0.5)


def test_serendipity_averages_per_user():
    popularity = {1: 100}
    # user A: relevant unseen item (unexp 1); user B: empty overlap
    score = serendipity([[99], [1]], [{99}, {2}], item_popularity=popularity)
    assert score == pytest.approx(0.5)


def test_profile_serendipity_rewards_relevant_items_far_from_history():
    score = serendipity(
        [[20]],
        [{20}],
        user_histories=[[10]],
        item_features=FEATURES,
    )
    assert score == pytest.approx(1.0)


def test_serendipity_respects_graded_zero_gain():
    popularity = {1: 0}
    assert serendipity([[1]], [{1: 0.0}], item_popularity=popularity) == pytest.approx(0.0)
    assert serendipity([[1]], [{1: 2.0}], item_popularity=popularity) == pytest.approx(1.0)


def test_serendipity_requires_exactly_one_unexpectedness_source():
    with pytest.raises(ValueError):
        serendipity([[1]], [{1}])
    with pytest.raises(ValueError):
        serendipity(
            [[10]],
            [{10}],
            item_popularity={10: 1},
            user_histories=[[10]],
            item_features=FEATURES,
        )
    with pytest.raises(ValueError):
        serendipity([[10]], [{10}], user_histories=[[10]])
    with pytest.raises(ValueError):
        serendipity([[10]], [{10}], item_features=FEATURES)


def test_serendipity_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        serendipity([[1]], [], item_popularity={1: 1})
    with pytest.raises(ValueError):
        serendipity([[10]], [{10}], user_histories=[], item_features=FEATURES)


def test_serendipity_empty_evaluation_is_zero():
    assert serendipity([], [], item_popularity={1: 1}) == 0.0


def test_serendipity_rejects_invalid_cutoff():
    with pytest.raises(ValueError):
        serendipity([[1]], [{1}], item_popularity={1: 1}, k=0)