"""Core data types shared across the pipeline.

These are plain dataclasses (not pydantic models) for the internal pipeline so the
hot path stays allocation-cheap and dependency-light. The API layer converts them
to pydantic response models explicitly in `api.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# A chunk's provenance in the retrieval pipeline. Kept as a Literal so mypy and
# readers both see the legal states.
RetrievalSource = Literal["bm25", "vector", "fused", "rerank"]


@dataclass(slots=True)
class Segment:
    """A unit of source text with an optional page number.

    For PDFs each page becomes one Segment so that citations can point at a page.
    For Markdown/plain text the whole document is a single Segment (page=None).
    """

    text: str
    page: Optional[int] = None


@dataclass(slots=True)
class Document:
    """A source document after loading, before chunking."""

    doc_id: str  # stable content-addressed id (see make_doc_id)
    path: str  # path relative to the corpus root
    title: str
    segments: list[Segment]
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments if s.text)

    @property
    def page_count(self) -> int:
        pages = [s.page for s in self.segments if s.page is not None]
        return max(pages) if pages else 0


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text. `chunk_id` is assigned by the store on insert."""

    chunk_id: int
    doc_id: str
    ordinal: int  # 0-based position of the chunk within its document
    text: str
    page: Optional[int]
    char_start: int
    char_end: int
    n_words: int

    @property
    def key(self) -> str:
        """Stable, human-readable identifier (doc-independent of DB rowids)."""
        return f"{self.doc_id}:{self.ordinal}"


@dataclass(slots=True)
class ScoredChunk:
    """A chunk plus retrieval bookkeeping, used to explain *why* it was returned."""

    chunk: Chunk
    score: float
    source: RetrievalSource = "fused"
    ranks: dict[str, int] = field(default_factory=dict)  # {"bm25": 3, "vector": 7}


@dataclass(slots=True)
class Citation:
    """A single citation surfaced to the user and used for groundedness scoring."""

    marker: str  # e.g. "[S1]"
    index: int  # 1-based source number
    chunk_id: int
    doc_id: str
    doc_title: str
    path: str
    ordinal: int
    page: Optional[int]
    char_start: int
    char_end: int
    snippet: str  # short, display-safe excerpt
    score: float
    # FULL chunk text. This is what the LLM is actually given as context -- a
    # truncated snippet would silently hide evidence that sits beyond the cut,
    # which looks exactly like a model hallucinating "not stated". Display code
    # and API responses use `snippet`, never this.
    context: str = ""


@dataclass(slots=True)
class Retrieved:
    """Diagnostics for the retrieval stage, returned alongside the answer."""

    candidates_bm25: int = 0
    candidates_vector: int = 0
    fused: int = 0
    reranked: bool = False
    vector_backend: str = "unknown"
    embed_backend: str = "unknown"
    evidence_confidence: float = 0.0


AnswerStatus = Literal[
    "answered",  # LLM produced an answer grounded in citations
    "abstained_insufficient_evidence",  # pre-gate: retrieval too weak, LLM not called
    "abstained_model_refused",  # LLM emitted the abstention phrase
    "abstained_no_llm",  # generation disabled by config
    "abstained_llm_unavailable",  # generation enabled but the endpoint is unreachable
    "error",  # generation or orchestration failure (message in `reason`)
]


@dataclass(slots=True)
class Answer:
    question: str
    status: AnswerStatus
    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    diagnostics: Retrieved = field(default_factory=Retrieved)
    model: Optional[str] = None
    reason: Optional[str] = None
    latency_ms: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.status.startswith("abstained")


def make_doc_id(rel_path: str, sha256: str) -> str:
    """Content-addressed document id.

    Including the path (not just the hash) means two byte-identical files in
    different locations stay distinct, which is what a user expects. The hash
    keeps the id stable across re-ingestion of unchanged content.
    """
    h = hashlib.sha256(f"{rel_path}\x00{sha256}".encode("utf-8")).hexdigest()
    return h[:16]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
