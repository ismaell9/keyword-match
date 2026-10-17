"""Retrieval internals: lexical coverage, evidence confidence, FTS query safety."""

from __future__ import annotations

import pytest

from ragqa.retrieval import (
    content_terms,
    evidence_confidence,
    lexical_coverage,
    sigmoid,
)
from ragqa.store import TokenizeError, build_fts_query
from ragqa.types import Chunk, ScoredChunk


def _hit(text: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(1, "docA", 0, text, None, 0, len(text), len(text.split()))
    return ScoredChunk(chunk=chunk, score=score, source="fused", ranks={})


def test_content_terms_drops_stopwords_and_single_chars():
    assert content_terms("What is the password policy?") == ["password", "policy"]
    assert content_terms("a i x to be") == []


def test_lexical_coverage_is_fraction_of_content_terms_present():
    assert lexical_coverage("password rotation policy", "the password rotation policy is 90 days") == pytest.approx(1.0)
    assert lexical_coverage("password rotation policy", "the password is here") == pytest.approx(1 / 3)
    assert lexical_coverage("password", "nothing relevant") == 0.0


def test_lexical_coverage_with_no_content_terms_is_zero():
    assert lexical_coverage("the is of", "anything") == 0.0


def test_sigmoid_is_bounded_and_monotonic():
    assert sigmoid(0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-20) < 0.01
    assert 0.99 < sigmoid(20) < 1.0
    assert sigmoid(-1) < sigmoid(1)


def test_evidence_confidence_is_zero_without_hits():
    assert evidence_confidence("q", [], reranker_active=True) == 0.0
    assert evidence_confidence("q", [], reranker_active=False) == 0.0


def test_evidence_confidence_uses_sigmoid_when_reranker_active():
    hits = [_hit("irrelevant text", score=0.0)]
    assert evidence_confidence("q", hits, reranker_active=True) == pytest.approx(0.5)
    high = [_hit("x", score=8.0)]
    assert evidence_confidence("q", high, reranker_active=True) > 0.99


def test_evidence_confidence_uses_coverage_without_reranker():
    # Query content terms: [often, passwords, rotated] -- 3 terms.
    # Top chunk contains "passwords" and "rotated" but not "often" -> 2/3.
    hits = [_hit("passwords must be rotated every 90 days")]
    conf = evidence_confidence("how often are passwords rotated", hits, reranker_active=False)
    assert conf == pytest.approx(2 / 3, abs=1e-6)

    weak = [_hit("the cat sat on the mat")]
    assert evidence_confidence("how often are passwords rotated", weak, reranker_active=False) == 0.0


def test_evidence_confidence_blends_top_two_chunks():
    hits = [_hit("passwords rotated", score=2.0), _hit("unrelated filler words entirely", score=1.0)]
    conf = evidence_confidence("passwords rotated", hits, reranker_active=False)
    # 0.8 * coverage(top) + 0.2 * coverage(second); second covers "passwords"+"rotated"? no -> 0
    assert 0.0 < conf < 1.0


def test_fts_query_quotes_every_term_and_joins_with_or():
    assert build_fts_query("password rotation") == '"password" OR "rotation"'


def test_fts_query_supports_and_mode():
    assert build_fts_query("password rotation", mode="and") == '"password" AND "rotation"'


def test_fts_query_is_injection_safe():
    """FTS5 operators and quotes must not leak into the MATCH expression."""
    q = build_fts_query('password" OR 1=1 -- * NEAR(foo)')
    # Every token is either the joiner or a double-quoted word; nothing raw leaks.
    for token in q.split():
        if token not in ("OR", "AND"):
            assert token.startswith('"') and token.endswith('"'), token
            assert all(ch.isalnum() or ch in "_" for ch in token[1:-1]), token


def test_fts_query_strips_quotes_by_tokenising_them_away():
    """Quotes (and other FTS syntax) are removed during tokenisation, so a term
    can never terminate the quoting early and inject an operator."""
    assert build_fts_query('pass"word') == '"pass" OR "word"'
    assert build_fts_query('ab"(cd)*ef') == '"ab" OR "cd" OR "ef"'


def test_fts_query_raises_on_terms_free_query():
    with pytest.raises(TokenizeError):
        build_fts_query("!!! ??? ---")
    with pytest.raises(TokenizeError):
        build_fts_query("a")  # single-character tokens are dropped


def test_fts_query_handles_unicode():
    q = build_fts_query("café naïve")
    assert '"café"' in q and '"naïve"' in q
