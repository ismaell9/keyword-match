"""Cross-encoder re-ranking.

A bi-encoder embeds query and document *independently*, so it cannot model their
interaction; it is fast but coarse. A cross-encoder reads (query, document) as a
single sequence and scores relevance directly, which is markedly more precise but
too slow to run over a whole corpus. The standard two-stage design -- retrieve
broad with bi-encoder/BM25, then re-rank a shortlist with a cross-encoder -- is
what this module implements, and it is the layer that turns "looks relevant" into
"is relevant".

The reranker is optional by design: if fastembed or the model is unavailable we
fall back to :class:`NoopReranker`, which preserves the fused order. That keeps
the service honest (it still works) and makes the eval harness able to measure
"with rerank" vs "without rerank" as an ablation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

log = logging.getLogger(__name__)


class RerankError(Exception):
    pass


class Reranker(ABC):
    name: str = "unknown"
    is_stub: bool = False

    @abstractmethod
    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document, higher == more relevant.

        Scores are only required to be *comparable within a single call*; they are
        not required to be probabilities or on any fixed scale.
        """


class NoopReranker(Reranker):
    """Identity reranker: preserves input order by returning a constant score.

    Keeping it a real Reranker (rather than `None` checks scattered through the
    pipeline) means the retrieval code has exactly one code path, and evaluations
    can toggle re-ranking with a single config flag.
    """

    name = "none"
    is_stub = True

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        # Descending scores preserve the incoming (fused) order.
        n = len(documents)
        return [float(n - i) for i in range(n)]


class CrossEncoderReranker(Reranker):
    """ONNX cross-encoder via fastembed (default: ms-marco MiniLM L6)."""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RerankError("fastembed is not installed") from exc
        self.name = model_name
        try:
            self._model = TextCrossEncoder(model_name=model_name)
        except Exception as exc:
            raise RerankError(f"failed to load cross-encoder {model_name!r}: {exc}") from exc
        self.is_stub = False

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        try:
            scores = list(self._model.rerank(query, list(documents)))
        except Exception as exc:
            raise RerankError(f"cross-encoder inference failed: {exc}") from exc
        if len(scores) != len(documents):
            raise RerankError(f"cross-encoder returned {len(scores)} scores for {len(documents)} docs")
        return [float(s) for s in scores]


def build_reranker(*, enabled: bool, model_name: str) -> tuple[Reranker, list[str]]:
    """Build a reranker, degrading to Noop with an explanatory note."""
    notes: list[str] = []
    if not enabled:
        notes.append("rerank: disabled by config (fused order preserved)")
        return NoopReranker(), notes
    try:
        reranker = CrossEncoderReranker(model_name)
        notes.append(f"rerank: cross-encoder {model_name}")
        return reranker, notes
    except RerankError as exc:
        notes.append(f"rerank: cross-encoder unavailable ({exc}); using no-op reranker")
        log.warning("cross-encoder unavailable: %s", exc)
        return NoopReranker(), notes
