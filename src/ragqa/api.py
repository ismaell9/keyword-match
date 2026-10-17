"""FastAPI service: /health, /ingest, /query (+ /documents and upload ingest).

The app owns exactly one :class:`~ragqa.pipeline.Pipeline` for its lifetime,
built at startup (model loading is the expensive part) and torn down at shutdown.

Security note, stated plainly because it belongs in the code and not only in a
wiki: this service has **no authentication**. `/ingest` writes files the caller
supplies and reads paths the caller names. That is acceptable for a local demo
and for the portfolio story ("here is the service"), but deploying it as-is on a
shared network would let anyone enumerate the filesystem and fill the disk. The
Dockerfile therefore binds to loopback by default and the README says so. Adding
an API-key dependency is a two-line change and is listed in SPEC.md as v3 work.
"""

# NOTE: this module deliberately does NOT use `from __future__ import annotations`.
# FastAPI resolves endpoint annotations at runtime, and postponed annotations turn
# names like the function-local `PipelineDep` alias into strings it cannot resolve
# -- which silently degrades a dependency into a required query parameter.

import logging
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .ingest import SUPPORTED_EXTS
from .pipeline import Pipeline, build_pipeline

log = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------- schemas


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    mode: str = Field(
        default="answer",
        pattern="^(answer|retrieval_only)$",
        description="'answer' runs the LLM; 'retrieval_only' returns just the ranked chunks.",
    )


class CitationModel(BaseModel):
    marker: str
    index: int
    chunk_id: int
    doc_id: str
    doc_title: str
    path: str
    ordinal: int
    page: Optional[int]
    char_start: int
    char_end: int
    snippet: str
    score: float


class RetrievedChunkModel(BaseModel):
    chunk_id: int
    doc_id: str
    ordinal: int
    page: Optional[int]
    score: float
    source: str
    ranks: dict[str, int]
    text: str


class QueryResponse(BaseModel):
    question: str
    status: str
    abstained: bool
    answer: str
    reason: Optional[str]
    citations: list[CitationModel]
    retrieved: list[RetrievedChunkModel]
    evidence_confidence: float
    model: Optional[str]
    latency_ms: float
    diagnostics: dict[str, Any]
    usage: dict[str, Any]


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=list, description="Files or dirs under the server's filesystem.")
    force: bool = Field(default=False, description="Re-embed even if the content hash is unchanged.")
    corpus: bool = Field(default=False, description="Ingest the configured CORPUS_DIR instead of `paths`.")


# --------------------------------------------------------------------- app


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("building pipeline (this loads the embedding/reranker models) ...")
    pipeline = build_pipeline(settings)
    for note in pipeline.notes:
        log.info("config: %s", note)
    app.state.pipeline = pipeline
    app.state.settings = settings
    try:
        yield
    finally:
        pipeline.close()
        log.info("pipeline closed")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version="0.1.0",
        description=(
            "Document Q&A with citations: BM25 + dense vectors fused with RRF, "
            "cross-encoder re-ranking, grounded generation with an abstention gate."
        ),
        lifespan=lifespan,
    )

    def get_pipeline(request: Request) -> Pipeline:
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:  # pragma: no cover - lifespan always sets it
            raise HTTPException(status_code=503, detail="pipeline not initialised")
        return pipeline

    PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]

    # ------------------------------------------------------------------ routes

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {
            "service": settings.api_title,
            "endpoints": ["/health", "/ingest", "/ingest/upload", "/query", "/documents"],
            "docs": "/docs",
        }

    @app.get("/health")
    def health(pipeline: PipelineDep, probe_llm: bool = Query(default=False)) -> dict[str, Any]:
        """Liveness + a full description of the *actual* running configuration.

        `probe_llm=true` additionally makes a 1-token call to the LLM endpoint so
        you can tell "configured" from "reachable".
        """
        return pipeline.health(probe_llm=probe_llm)

    @app.post("/ingest")
    def ingest(pipeline: PipelineDep, body: IngestRequest) -> dict[str, Any]:
        """Ingest server-side paths, or the whole corpus directory."""
        if body.corpus or not body.paths:
            report = pipeline.ingest_corpus(force=body.force)
            return report.as_dict()

        resolved: list[Path] = []
        for raw in body.paths:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (settings.corpus_dir.parent / p).resolve()
            if not p.exists():
                raise HTTPException(status_code=400, detail=f"path does not exist: {raw}")
            resolved.append(p)
        from .pipeline import iter_supported_paths

        report = pipeline.ingest_paths(iter_supported_paths(resolved), force=body.force)
        return report.as_dict()

    @app.post("/ingest/upload")
    async def ingest_upload(
        pipeline: PipelineDep,
        files: Annotated[list[UploadFile], File(description="Documents to ingest (.pdf/.md/.txt/...)")],
        force: bool = Query(default=False),
    ) -> dict[str, Any]:
        """Ingest uploaded files via a sandboxed temp directory.

        Filenames are reduced to a safe basename and the temp dir is removed
        afterwards, so an upload cannot escape into the corpus directory or
        overwrite an existing file. Size is capped by MAX_UPLOAD_BYTES.
        """
        if not files:
            raise HTTPException(status_code=400, detail="no files supplied")
        max_bytes = settings.max_upload_bytes
        written: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="ragqa-upload-") as tmp:
            tmpdir = Path(tmp)
            for uf in files:
                suffix = Path(uf.filename or "").suffix.lower()
                if suffix not in SUPPORTED_EXTS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_EXTS)}",
                    )
                safe = _SAFE_NAME_RE.sub("_", Path(uf.filename or "upload").name) or "upload"
                dest = (tmpdir / safe).resolve()
                if tmpdir.resolve() not in dest.parents:  # belt and braces
                    raise HTTPException(status_code=400, detail="invalid filename")
                data = await uf.read()
                if len(data) > max_bytes:
                    raise HTTPException(status_code=413, detail=f"{safe} exceeds {max_bytes} bytes")
                dest.write_bytes(data)
                written.append(dest)

            report = pipeline.ingest_paths(written, root=tmpdir, force=force)
            out = report.as_dict()
            # The paths in the report point at a temp dir that is about to vanish;
            # report the original filenames instead so the response is meaningful.
            for entry in out["indexed"]:
                entry["path"] = Path(entry["path"]).name
            return out

    @app.post("/query", response_model=QueryResponse)
    def query(pipeline: PipelineDep, body: QueryRequest) -> QueryResponse:
        if body.mode == "retrieval_only":
            result = pipeline.retrieve(body.question, top_k=body.top_k)
            return QueryResponse(
                question=body.question,
                status="retrieval_only",
                abstained=False,
                answer="",
                reason=None,
                citations=[],
                retrieved=[
                    RetrievedChunkModel(
                        chunk_id=sc.chunk.chunk_id,
                        doc_id=sc.chunk.doc_id,
                        ordinal=sc.chunk.ordinal,
                        page=sc.chunk.page,
                        score=round(sc.score, 4),
                        source=sc.source,
                        ranks=sc.ranks,
                        text=sc.chunk.text,
                    )
                    for sc in result.hits
                ],
                evidence_confidence=result.diagnostics.evidence_confidence,
                model=None,
                latency_ms=0.0,
                diagnostics=_diag_dict(result.diagnostics),
                usage={},
            )

        answer = pipeline.query(body.question, top_k=body.top_k)
        return QueryResponse(
            question=answer.question,
            status=answer.status,
            abstained=answer.abstained,
            answer=answer.text,
            reason=answer.reason,
            citations=[CitationModel(**_citation_dict(c)) for c in answer.citations],
            retrieved=[
                RetrievedChunkModel(
                    chunk_id=sc.chunk.chunk_id,
                    doc_id=sc.chunk.doc_id,
                    ordinal=sc.chunk.ordinal,
                    page=sc.chunk.page,
                    score=round(sc.score, 4),
                    source=sc.source,
                    ranks=sc.ranks,
                    text=sc.chunk.text,
                )
                for sc in answer.retrieved
            ],
            evidence_confidence=answer.diagnostics.evidence_confidence,
            model=answer.model,
            latency_ms=round(answer.latency_ms, 1),
            diagnostics=_diag_dict(answer.diagnostics),
            usage=answer.usage,
        )

    @app.get("/documents")
    def documents(pipeline: PipelineDep) -> dict[str, Any]:
        return {"documents": pipeline.store.list_documents()}

    return app


def _citation_dict(c: Any) -> dict[str, Any]:
    # dataclasses with slots=True have no __dict__; build the mapping explicitly.
    return {
        "marker": c.marker,
        "index": c.index,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "doc_title": c.doc_title,
        "path": c.path,
        "ordinal": c.ordinal,
        "page": c.page,
        "char_start": c.char_start,
        "char_end": c.char_end,
        "snippet": c.snippet,
        "score": c.score,
    }


def _diag_dict(d: Any) -> dict[str, Any]:
    return {
        "candidates_bm25": d.candidates_bm25,
        "candidates_vector": d.candidates_vector,
        "fused": d.fused,
        "reranked": d.reranked,
        "vector_backend": d.vector_backend,
        "embed_backend": d.embed_backend,
        "evidence_confidence": d.evidence_confidence,
    }


# Module-level app for `uvicorn ragqa.api:app`. The pipeline is built in lifespan,
# not at import, so importing this module stays cheap (and testable).
app = create_app()


def main() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    settings: Settings = get_settings()
    uvicorn.run(
        "ragqa.api:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
    )
    _ = settings


if __name__ == "__main__":  # pragma: no cover
    main()
