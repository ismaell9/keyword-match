"""Reciprocal Rank Fusion: ordering semantics, robustness, determinism."""

from __future__ import annotations

import pytest

from ragqa.fusion import reciprocal_rank_fusion


def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion({"bm25": [(10, -5.0), (20, -3.0), (30, -1.0)]})
    assert [h.id for h in fused] == [10, 20, 30]


def test_item_ranked_by_both_lists_beats_item_ranked_by_one():
    """The core RRF property: agreement across legs is rewarded."""
    fused = reciprocal_rank_fusion(
        {
            "bm25": [(1, 0.0), (2, 0.0)],  # 1 first, 2 second
            "vector": [(2, 0.0), (1, 0.0)],  # 2 first, 1 second
        },
        k=60,
    )
    assert {h.id for h in fused} == {1, 2}
    # Both appear in both lists, so both get 1/(60+1) + 1/(60+2).
    assert fused[0].score == pytest.approx(fused[1].score)


def test_agreement_beats_a_single_first_place():
    fused = reciprocal_rank_fusion(
        {
            "bm25": [(1, 0.0), (2, 0.0), (3, 0.0)],
            "vector": [(2, 0.0), (3, 0.0)],
        },
        k=60,
    )
    # Item 1 is #1 in one list only; item 2 is #1 and #1 -> should win.
    assert fused[0].id == 2


def test_ranks_are_recorded_for_explainability():
    fused = reciprocal_rank_fusion({"bm25": [(7, -2.0)], "vector": [(7, 0.5), (8, 0.4)]})
    top = fused[0]
    assert top.id == 7
    assert top.ranks == {"bm25": 1, "vector": 1}
    assert top.raw_scores == {"bm25": -2.0, "vector": 0.5}


def test_weights_shift_the_ordering():
    rankings = {"bm25": [(1, 0.0)], "vector": [(2, 0.0)]}
    even = reciprocal_rank_fusion(rankings, k=60)
    assert [h.id for h in even][0] == 1  # tie broken by first-seen order

    weighted = reciprocal_rank_fusion(rankings, k=60, weights={"vector": 3.0, "bm25": 1.0})
    assert weighted[0].id == 2


def test_zero_weight_drops_a_leg():
    fused = reciprocal_rank_fusion({"bm25": [(1, 0.0)], "vector": [(2, 0.0)]}, weights={"vector": 0.0})
    assert [h.id for h in fused] == [1]


def test_top_k_truncates():
    fused = reciprocal_rank_fusion({"bm25": [(i, 0.0) for i in range(20)]}, top_k=5)
    assert len(fused) == 5


def test_duplicate_ids_within_one_list_are_counted_once():
    """A list containing the same id twice must not double-count it."""
    fused = reciprocal_rank_fusion({"bm25": [(1, 0.0), (1, 0.0)]})
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"bm25": [], "vector": []}) == []


def test_deterministic_tie_break_uses_first_seen_order():
    """Equal scores must not reorder between runs (reproducible evaluations)."""
    rankings = {"bm25": [(5, 0.0), (6, 0.0)], "vector": [(5, 0.0), (6, 0.0)]}
    first = [h.id for h in reciprocal_rank_fusion(rankings)]
    second = [h.id for h in reciprocal_rank_fusion(rankings)]
    assert first == second == [5, 6]


def test_larger_k_flattens_rank_differences():
    rankings = {"bm25": [(1, 0.0), (2, 0.0)]}
    small = reciprocal_rank_fusion(rankings, k=1)
    large = reciprocal_rank_fusion(rankings, k=1000)
    assert (small[0].score - small[1].score) > (large[0].score - large[1].score)


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"bm25": [(1, 0.0)]}, k=0)
