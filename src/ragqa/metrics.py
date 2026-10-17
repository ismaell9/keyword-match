"""Retrieval and answer-quality metrics.

All functions are pure and take plain sequences so they are trivially unit
testable and reusable by both the offline evaluator and any future online
monitoring. Relevance is binary unless stated otherwise.

Implemented:
  * recall@k        -- did we get *all* the relevant chunks in the top k?
  * precision@k     -- how much of the top k was relevant?
  * ndcg@k          -- binary-gain nDCG; rewards putting relevant items early.
  * mrr             -- reciprocal rank of the first relevant item (0 if none).
  * hit@k           -- at least one relevant item in the top k?

nDCG is the headline retrieval metric because, unlike recall, it is sensitive to
*ordering*, which is exactly what re-ranking changes.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# --- averaging helpers -----------------------------------------------------


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile. `p` in [0, 100]."""
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


# --- ranking metrics -------------------------------------------------------


def recall_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """|retrieved[:k] ∩ relevant| / |relevant|. 0.0 when there is nothing relevant."""
    rel = set(relevant)
    if not rel:
        return 0.0
    top = set(list(retrieved)[:k])
    return len(top & rel) / len(rel)


def precision_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """|retrieved[:k] ∩ relevant| / min(k, len(retrieved)). 0.0 if nothing retrieved."""
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for x in top if x in rel) / len(top)


def hit_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return 1.0 if any(x in rel for x in list(retrieved)[:k]) else 0.0


def mrr(retrieved: Sequence[int], relevant: Iterable[int]) -> float:
    """Reciprocal rank of the first relevant item across the whole list."""
    rel = set(relevant)
    if not rel:
        return 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with the standard log2(i+1) discount."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(retrieved: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Binary-gain nDCG@k.

    Normalising by the ideal ranking makes the value comparable across questions
    that have different numbers of relevant chunks -- without that, a question
    with 5 relevant chunks could never score 1.0 with k=3.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    gains = [1.0 if doc in rel else 0.0 for doc in list(retrieved)[:k]]
    dcg = dcg_at_k(gains, k)
    # Ideal: all the relevant items, as early as possible.
    ideal = [1.0] * min(len(rel), k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


# --- text / groundedness helpers -------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter: on ., !, ? and newlines.

    Deliberately not an NLP sentence tokenizer -- we only need enough structure to
    ask "how many sentences carry a citation?" and a regex is auditable. It will
    over-split on abbreviations; the metric is a floor, not a precise count.
    """
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def term_support(answer_text: str, context_text: str, terms: Sequence[str]) -> float:
    """Fraction of the answer's content terms that appear in the supplied context.

    A *proxy* for groundedness, not entailment. It catches the common failure
    mode where a model sprinkles in plausible-but-absent entities/numbers; it does
    NOT catch an answer that reuses context vocabulary in a false relation. This
    limitation is stated in docs/evaluation.md and must not be presented as more
    than it is.
    """
    if not terms:
        return 0.0
    ctx = context_text.lower()
    present = sum(1 for t in terms if t in ctx)
    return present / len(terms)
