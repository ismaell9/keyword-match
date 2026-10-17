"""Reciprocal Rank Fusion (RRF) for combining ranked lists.

RRF scores a document as ``sum_lists weight_l / (k + rank_list(document))``,
where rank is 1-based (Cormack et al., 2009). Two properties make it the right
choice here over score normalisation:

1. It only uses *ranks*, so BM25 scores (unbounded, negative) and vector
   distances (L2) never have to be put on a common scale -- a normalisation you
   would have to re-tune per corpus.
2. It is robust: a document ranked moderately by both legs beats a document that
   is #1 in one leg and absent from the other, which is exactly the behaviour you
   want when one leg is lexical and the other semantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FusedHit:
    """One fused result, carrying the per-list ranks so the result is explainable."""

    id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[int, float]]],
    *,
    k: int = 60,
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[FusedHit]:
    """Fuse several ranked lists into one.

    Args:
        rankings: ``{list_name: [(item_id, score), ...]}``, each already sorted
            best-first. Scores are kept for diagnostics but NOT used for fusion.
        k: RRF constant. Larger k flattens the rank weighting.
        top_k: truncate the fused output.
        weights: optional per-list multipliers (e.g. trust the vector leg more).

    Returns:
        Fused hits sorted by descending fused score. Ties break deterministically
        by first-seen order so results are reproducible run to run.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    weights = weights or {}
    fused: dict[int, FusedHit] = {}
    first_seen: dict[int, int] = {}
    counter = 0

    for name, hits in rankings.items():
        weight = float(weights.get(name, 1.0))
        if weight == 0.0:
            continue
        seen_in_list: set[int] = set()
        for rank, (item_id, raw) in enumerate(hits, start=1):
            # Defensive: duplicates inside a single list would double-count.
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            hit = fused.get(item_id)
            if hit is None:
                hit = FusedHit(id=item_id, score=0.0)
                fused[item_id] = hit
                first_seen[item_id] = counter
                counter += 1
            hit.score += weight / (k + rank)
            hit.ranks[name] = rank
            hit.raw_scores[name] = float(raw)

    ordered = sorted(fused.values(), key=lambda h: (-h.score, first_seen[h.id]))
    return ordered[:top_k] if top_k else ordered
