"""Metric correctness, checked against hand-computed values."""

from __future__ import annotations

import math

import pytest

from ragqa.metrics import (
    dcg_at_k,
    hit_at_k,
    mean,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    split_sentences,
    term_support,
)


def test_recall_at_k():
    assert recall_at_k([1, 2, 3], {2, 4}, 3) == pytest.approx(0.5)  # got 1 of 2
    assert recall_at_k([1, 2, 3], {2, 4}, 10) == pytest.approx(0.5)
    assert recall_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert recall_at_k([], {1}, 5) == 0.0


def test_recall_with_no_relevant_items_is_zero_not_undefined():
    assert recall_at_k([1, 2], set(), 2) == 0.0


def test_precision_at_k():
    assert precision_at_k([1, 2, 3], {2, 4}, 3) == pytest.approx(1 / 3)
    assert precision_at_k([2, 1], {1, 2}, 2) == pytest.approx(1.0)
    assert precision_at_k([], {1}, 5) == 0.0


def test_hit_at_k():
    assert hit_at_k([9, 8, 1], {1}, 2) == 0.0  # relevant item is at rank 3
    assert hit_at_k([9, 8, 1], {1}, 3) == 1.0


def test_mrr_uses_the_first_relevant_rank():
    assert mrr([9, 8, 1], {1}) == pytest.approx(1 / 3)
    assert mrr([1, 8, 9], {1}) == pytest.approx(1.0)
    assert mrr([2, 3], {1}) == 0.0


def test_dcg_matches_hand_computation():
    # gains [1, 0, 1] -> 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0.5
    assert dcg_at_k([1, 0, 1], 3) == pytest.approx(1.5)


def test_ndcg_matches_hand_computation():
    # retrieved [1,2,3], relevant {1,3}: DCG = 1.5
    # IDCG (ideal: [1,1,0]) = 1/log2(2) + 1/log2(3) = 1 + 0.63093
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k([1, 2, 3], {1, 3}, 3) == pytest.approx(expected)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)


def test_ndcg_rewards_putting_relevant_items_first():
    early = ndcg_at_k([1, 2, 3, 4], {1}, 4)
    late = ndcg_at_k([2, 3, 4, 1], {1}, 4)
    assert early > late


def test_ndcg_with_more_relevant_than_k_can_still_reach_one():
    # 5 relevant items, k=3: a perfect top-3 should score 1.0.
    assert ndcg_at_k([1, 2, 3], {1, 2, 3, 4, 5}, 3) == pytest.approx(1.0)


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
    assert mean([1, 2, 3]) == pytest.approx(2.0)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 0) == pytest.approx(1.0)
    assert percentile([1, 2, 3, 4], 100) == pytest.approx(4.0)
    assert percentile([5], 95) == pytest.approx(5.0)
    assert percentile([], 50) == 0.0


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_split_sentences_basic():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("A\nB\n\nC") == ["A", "B", "C"]
    assert split_sentences("") == []


def test_term_support_counts_terms_present_in_context():
    assert term_support("alpha beta gamma", "alpha and beta here", ["alpha", "beta", "gamma"]) == pytest.approx(2 / 3)
    assert term_support("x", "y", []) == 0.0
