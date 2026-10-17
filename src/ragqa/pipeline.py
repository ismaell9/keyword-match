"""Pipeline wiring: one object that owns store + embedder + reranker + generator.

`build_pipeline()` is the single place where configuration turns into concrete
objects, and it returns human-readable `notes` describing every degradation that
happened (real vs stub embedder, sqlite-vec vs numpy, cross-encoder vs no-op,
LLM endpoint vs offline). Those notes are what the API's /health, the CLI banner,
and the evaluation report all print, so the running configuration is never a
mystery.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .chunking import chunk_document
from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .generation import Generator, LLMClient, NullLLM, build_llm
from .ingest import IngestError, IngestReport, iter_files, load_document
from .rerank import Reranker, build_reranker
from .retrieval import Retriever, RetrievalResult
from .store import SqliteStore
from .types import Answer

log = logging.getLogger(__name__)


@dataclass
class Pipeline:
    settings: Settings
    store: SqliteStore
    embedder: Embedder
    reranker: Reranker
    llm: LLMClient
    retriever: Retriever
    generator: Generator
    notes: list[str] = field(default_factory=list)

    # --------------------------------------------------------------- ingestion

    def ingest_paths(
        self,
        paths: Iterable[Path],
        *,
        root: Path | None = None,
        force: bool = False,
        incremental: bool | None = None,
    ) -> IngestReport:
        """Chunk, embed and upsert documents.

        `force=True` re-embeds even files whose content hash is unchanged.
        Content-hash skipping is what makes re-running ingestion cheap: on a
        stable corpus it does no embedding work at all.
        """
        s = self.settings
        incremental = s.incremental_ingest if incremental is None else incremental
        indexed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for path in paths:
            try:
                doc = load_document(path, root=root or path.parent)
            except IngestError as exc:
                errors.append({"path": str(path), "reason": str(exc)})
                continue
            except Exception as exc:  # defensive
                errors.append({"path": str(path), "reason": f"unexpected: {exc}"})
                continue

            if incremental and not force:
                existing = self.store.document_unchanged(doc.path, doc.sha256)
                if existing is not None:
                    skipped.append({"path": doc.path, "doc_id": existing, "reason": "unchanged"})
                    continue

            chunks = chunk_document(
                doc, size=s.chunk_size_words, overlap=s.chunk_overlap_words, min_size=s.chunk_min_words
            )
            if not chunks:
                skipped.append({"path": doc.path, "reason": "no extractable text"})
                continue

            started = time.perf_counter()
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            embed_ms = (time.perf_counter() - started) * 1000.0
            result = self.store.upsert_document(doc, chunks, vectors)
            indexed.append(
                {
                    "path": doc.path,
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "chunks": result["chunks"],
                    "pages": doc.page_count,
                    "embed_ms": round(embed_ms, 1),
                    "replaced_documents": result["replaced_documents"],
                }
            )

        if indexed:
            self.store.finalize_ingest(
                embed_name=self.embedder.name,
                embed_stub=self.embedder.is_stub,
                dim=self.embedder.dim,
                chunk_size=s.chunk_size_words,
                chunk_overlap=s.chunk_overlap_words,
                notes=self.notes,
            )
        return IngestReport(indexed=indexed, skipped=skipped, errors=errors)

    def ingest_corpus(self, *, force: bool = False) -> IngestReport:
        """Ingest every supported file under CORPUS_DIR."""
        root = self.settings.corpus_dir
        if not root.exists():
            return IngestReport(
                indexed=[], skipped=[], errors=[{"path": str(root), "reason": "corpus dir does not exist"}]
            )
        root = root if root.is_dir() else root.parent
        return self.ingest_paths(iter_files(root), root=root, force=force)

    def rebuild(self) -> None:
        """Drop all indexed content and re-create empty structures."""
        self.store.clear()

    # ----------------------------------------------------------------- queries

    def retrieve(self, question: str, *, top_k: int | None = None) -> RetrievalResult:
        return self.retriever.retrieve(question, top_k=top_k)

    def query(self, question: str, *, top_k: int | None = None) -> Answer:
        """Full RAG path: retrieve then generate (or abstain)."""
        started = time.perf_counter()
        result = self.retrieve(question, top_k=top_k)
        answer = self.generator.answer(question, result)
        answer.latency_ms = (time.perf_counter() - started) * 1000.0
        return answer

    # ------------------------------------------------------------------ health

    def health(self, *, probe_llm: bool = False) -> dict[str, Any]:
        stats = self.store.stats()
        llm_info: dict[str, Any] = {
            "configured": self.llm.is_configured(),
            "client": self.llm.name,
            "base_url": self.settings.llm_base_url,
            "model": self.settings.llm_model,
        }
        if probe_llm and self.llm.is_configured():
            llm_info.update(_probe_llm(self))
        return {
            "status": "ok",
            "index": stats,
            "capabilities": {
                "embeddings": {
                    "backend": self.embedder.name,
                    "dim": self.embedder.dim,
                    "is_stub": self.embedder.is_stub,
                },
                "vectors": {
                    "backend": self.store.vector_backend,
                    "sqlite_vec_error": self.store.sqlite_vec_error,
                },
                "reranker": {"backend": self.reranker.name, "is_stub": self.reranker.is_stub},
                "generation": llm_info,
            },
            "notes": self.notes,
            "config": {
                "chunk_size_words": self.settings.chunk_size_words,
                "chunk_overlap_words": self.settings.chunk_overlap_words,
                "bm25_top_k": self.settings.bm25_top_k,
                "vector_top_k": self.settings.vector_top_k,
                "rrf_k": self.settings.rrf_k,
                "retrieve_top_k": self.settings.retrieve_top_k,
                "min_evidence_confidence": self.settings.min_evidence_confidence,
            },
        }

    def close(self) -> None:
        self.store.close()


def _probe_llm(pipeline: Pipeline) -> dict[str, Any]:
    """Cheap reachability check for /health?probe_llm=1."""
    started = time.perf_counter()
    try:
        # A 1-token completion exercises auth + model name + network in one shot.
        pipeline.llm.chat([{"role": "user", "content": "ok"}])  # type: ignore[arg-type]
        return {"reachable": True, "probe_ms": round((time.perf_counter() - started) * 1000.0, 1)}
    except Exception as exc:
        return {"reachable": False, "probe_error": str(exc)[:300]}


def build_pipeline(
    settings: Settings | None = None,
    *,
    embed_backend: str | None = None,
    rerank_enabled: bool | None = None,
    load_llm: bool = True,
    create: bool = True,
) -> Pipeline:
    """Construct the whole pipeline from settings.

    Keyword overrides exist so the evaluation harness and tests can build
    alternative configurations (e.g. hash stub, rerank off) without mutating
    environment variables.
    """
    s = (settings or get_settings()).resolved()
    notes: list[str] = []

    store = SqliteStore(s.db_path, vector_backend=s.vector_backend, create=create)
    store.init_schema()
    notes.append(f"storage: {s.db_path} (vector backend: {store.vector_backend})")
    if store.sqlite_vec_error:
        notes.append(f"storage: sqlite-vec unavailable -> numpy fallback ({store.sqlite_vec_error})")

    embedder, embed_notes = build_embedder(
        backend=embed_backend or s.embed_backend,
        model_name=s.embed_model,
        hash_dim=s.embed_dim,
        batch_size=s.embed_batch_size,
    )
    notes.extend(embed_notes)

    reranker, rerank_notes = build_reranker(
        enabled=s.rerank_enabled if rerank_enabled is None else rerank_enabled,
        model_name=s.rerank_model,
    )
    notes.extend(rerank_notes)

    if load_llm:
        llm, llm_notes = build_llm(s)
    else:
        llm, llm_notes = NullLLM("LLM not loaded for this run"), ["generation: not loaded"]
    notes.extend(llm_notes)

    store.assert_compatible(
        embed_name=embedder.name,
        dim=embedder.dim,
        chunk_size=s.chunk_size_words,
        chunk_overlap=s.chunk_overlap_words,
    )

    retriever = Retriever(store, embedder, reranker, s)
    generator = Generator(llm, s)
    return Pipeline(
        settings=s,
        store=store,
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        retriever=retriever,
        generator=generator,
        notes=notes,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Small utility used by tests/diagnostics."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def iter_supported_paths(paths: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(iter_files(p))
        elif p.is_file():
            out.append(p)
    return out
