"""Hybrid retrieval: BM25 + vector -> RRF -> cross-encoder re-rank.

The pipeline is intentionally legible and each stage is separable, because the
evaluation harness needs to attribute a metric to a stage:

    query
      |-- BM25 (FTS5, porter) -----> ranked list A (lexical recall)
      |-- dense vector (sqlite-vec) -> ranked list B (semantic recall)
                    |
                    v
              RRF fusion (rank-based)
                    |
                    v
          cross-encoder re-rank (precision)
                    |
                    v
               top-k chunks + a scalar evidence-confidence score

`evidence_confidence` is the *abstention signal*. It is deliberately a distinct
notion from the retrieval score: retrieval scores always produce a "best" chunk,
even when nothing relevant exists, so asking "is the best chunk good?" needs its
own calibration. Two definitions are used depending on what is available:

* With a real cross-encoder: the cross-encoder's score is already a query-document
  relevance judgement, so we squash the top logit to (0, 1) with a sigmoid.
* With only lexical/fused signals: we fall back to *query-term coverage* of the
  best chunk -- the fraction of query content words that actually appear. It is a
  crude signal, but it is honest, deterministic, and testable, and the eval
  harness sweeps its threshold rather than us pretending one value is optimal.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from .config import Settings
from .embeddings import Embedder, EmbeddingError
from .fusion import reciprocal_rank_fusion
from .rerank import Reranker
from .store import SqliteStore, TokenizeError
from .types import Chunk, Retrieved, ScoredChunk

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9_]+")

# A deliberately small stop list. Its only job is to stop "what is the" from
# inflating query-term coverage in the confidence signal. Aggressive stop lists
# hurt queries like "to be or not to be", so we keep it minimal and obvious.
STOPWORDS: frozenset[str] = frozenset(
    """a an the is are was were be been being do does did doing have has had having
    of in on at to for from by with about into over after before under above
    and or but if then else than that this these those it its as not no nor
    what which who whom whose when where why how many much any some all
    can could should would may might will shall i you he she they we me my your""".split()
)


def content_terms(text: str) -> list[str]:
    """Lower-cased content words (stopwords and 1-char tokens removed)."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def lexical_coverage(query: str, text: str) -> float:
    """Fraction of query content terms present in `text`. In [0, 1]."""
    q = content_terms(query)
    if not q:
        return 0.0
    t = set(_WORD_RE.findall(text.lower()))
    return sum(1 for w in q if w in t) / len(q)


def sigmoid(x: float) -> float:
    # Numerically stable two-branch form.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(slots=True)
class RetrievalResult:
    hits: list[ScoredChunk]
    diagnostics: Retrieved
    bm25: list[tuple[int, float]] = field(default_factory=list)
    vector: list[tuple[int, float]] = field(default_factory=list)
    # Fused ordering *before* re-ranking. Exposed so the evaluation harness can
    # ablate retrieval stages (bm25 only / vector only / hybrid / +rerank) from a
    # single retrieval call instead of re-running the pipeline four times.
    fused_ids: list[int] = field(default_factory=list)
    # Document metadata for the selected hits, resolved once by the retriever so
    # the generation layer never has to reach back into the store.
    titles: dict[str, str] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)


class Retriever:
    """Stateless-per-call hybrid retriever over a :class:`SqliteStore`."""

    def __init__(
        self,
        store: SqliteStore,
        embedder: Embedder,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings

    # ------------------------------------------------------------------ public

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.retrieve_top_k
        diag = Retrieved(
            vector_backend=self.store.vector_backend,
            embed_backend=self.embedder.name,
        )

        # -- leg 1: BM25 ------------------------------------------------------
        try:
            bm25 = self.store.search_bm25(query, s.bm25_top_k)
        except TokenizeError as exc:
            # Queries made only of punctuation produce no MATCH expression.
            # That is not an error condition for the service: the vector leg can
            # still answer. We record it and continue with one leg.
            log.info("BM25 leg skipped: %s", exc)
            bm25 = []
        diag.candidates_bm25 = len(bm25)

        # -- leg 2: dense vector ---------------------------------------------
        vector: list[tuple[int, float]] = []
        try:
            qvec = self.embedder.embed_query(query)
            vector = self.store.search_vector(qvec, s.vector_top_k)
        except EmbeddingError as exc:
            # Degraded mode: lexical-only. Better a warning and a working
            # keyword search than a 500.
            log.warning("vector leg skipped: %s", exc)
        diag.candidates_vector = len(vector)

        # -- fuse -------------------------------------------------------------
        fused = reciprocal_rank_fusion(
            {"bm25": bm25, "vector": vector},
            k=s.rrf_k,
            top_k=s.fusion_top_k,
        )
        diag.fused = len(fused)
        if not fused:
            diag.evidence_confidence = 0.0
            return RetrievalResult(hits=[], diagnostics=diag, bm25=bm25, vector=vector, fused_ids=[])

        chunk_map = self.store.get_chunks([h.id for h in fused])
        ordered: list[ScoredChunk] = []
        for hit in fused:
            chunk = chunk_map.get(hit.id)
            if chunk is None:  # vector row without a chunk row: data drift, skip
                continue
            ordered.append(
                ScoredChunk(chunk=chunk, score=hit.score, source="fused", ranks=dict(hit.ranks))
            )

        # -- re-rank ----------------------------------------------------------
        ordered = self._rerank(query, ordered)
        diag.reranked = not self.reranker.is_stub
        selected = ordered[:top_k]

        diag.evidence_confidence = evidence_confidence(
            query, selected, reranker_active=not self.reranker.is_stub
        )
        doc_ids = [sc.chunk.doc_id for sc in selected]
        return RetrievalResult(
            hits=selected,
            diagnostics=diag,
            bm25=bm25,
            vector=vector,
            fused_ids=[sc.chunk.chunk_id for sc in ordered],
            titles=self.store.document_titles(doc_ids),
            paths=self.store.document_paths(doc_ids),
        )

    # ----------------------------------------------------------------- private

    def _rerank(self, query: str, ordered: list[ScoredChunk]) -> list[ScoredChunk]:
        if self.reranker.is_stub or len(ordered) < 2:
            return ordered
        window = ordered[: self.settings.rerank_top_n]
        tail = ordered[self.settings.rerank_top_n :]
        documents = [sc.chunk.text for sc in window]
        try:
            scores = self.reranker.rerank(query, documents)
        except Exception as exc:  # never let re-ranking break a query
            log.warning("re-ranking failed, keeping fused order: %s", exc)
            return ordered
        reranked = [
            ScoredChunk(
                chunk=sc.chunk,
                score=float(score),
                source="rerank",
                ranks=dict(sc.ranks),
            )
            for sc, score in zip(window, scores, strict=True)
        ]
        # Stable sort so equal cross-encoder scores keep their fused order.
        reranked.sort(key=lambda sc: -sc.score)
        return reranked + tail


def evidence_confidence(
    query: str,
    hits: list[ScoredChunk],
    *,
    reranker_active: bool,
) -> float:
    """Scalar in [0, 1]: how much evidence the top chunks actually contain.

    See the module docstring for the two definitions. Returning 0.0 for an empty
    hit list is what makes the empty-retrieval case unambiguous downstream.
    """
    if not hits:
        return 0.0
    if reranker_active:
        # Cross-encoder logits for ms-marco models are typically in [-12, +6];
        # sigmoid maps that to (0, 1) without needing a tuned offset.
        return sigmoid(hits[0].score)
    # No cross-encoder: coverage of the best chunk, plus a small credit for the
    # second-best, to avoid a knife-edge dependence on one chunk's phrasing.
    best = lexical_coverage(query, hits[0].chunk.text)
    if len(hits) > 1:
        second = lexical_coverage(query, hits[1].chunk.text)
        return min(1.0, 0.8 * best + 0.2 * second)
    return best
