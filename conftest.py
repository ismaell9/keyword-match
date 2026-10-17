"""Shared pytest fixtures.

The suite is designed to run with **no network and no model downloads**: every
test that needs an embedder uses the deterministic hash stub and the no-op
reranker. The real fastembed/ONNX path is exercised by the smoke test in the
Makefile and documented in docs/SETUP-NOTES.md, not here -- a test suite that
downloads a 130MB model is a test suite nobody runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragqa.chunking import chunk_document  # noqa: E402
from ragqa.config import Settings  # noqa: E402
from ragqa.ingest import load_document  # noqa: E402


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointing at a throwaway DB, with the stub embedder and no LLM."""
    return Settings(
        db_path=tmp_path / "index.sqlite3",
        corpus_dir=tmp_path / "corpus",
        embed_backend="hash",
        embed_dim=64,
        rerank_enabled=False,
        llm_enabled=False,
        vector_backend="auto",
        chunk_size_words=60,
        chunk_overlap_words=12,
        chunk_min_words=15,
        retrieve_top_k=5,
    )


@pytest.fixture
def sample_corpus(tmp_path: Path) -> Path:
    """A tiny two-document corpus with facts that support the test queries."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "security.md").write_text(
        "# Security Policy\n\n"
        "All production logins must use multi-factor authentication.\n\n"
        "Passwords must be rotated every 90 days. The minimum length is 14 characters.\n\n"
        "Authentication logs are retained for 400 days.\n",
        encoding="utf-8",
    )
    (corpus / "runbook.md").write_text(
        "# Deployment Runbook\n\n"
        "A rollback redeploys the previous image and keeps it for 14 days.\n\n"
        "The availability objective is 99.9 percent over a rolling 30 days.\n",
        encoding="utf-8",
    )
    return corpus


@pytest.fixture
def indexed_pipeline(tmp_settings: Settings, sample_corpus: Path):
    """A Pipeline over the sample corpus, using the stub embedder."""
    from ragqa.pipeline import build_pipeline

    settings = tmp_settings.model_copy(update={"corpus_dir": sample_corpus})
    pipeline = build_pipeline(settings, load_llm=False)
    report = pipeline.ingest_corpus()
    assert not report.errors, report.errors
    yield pipeline
    pipeline.close()
