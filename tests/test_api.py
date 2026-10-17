"""API tests, run against the stub embedder with generation disabled.

These exercise the real request/response path (including FastAPI's lifespan, which
builds the pipeline) without downloading any model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragqa import config as config_module


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "policy.md").write_text(
        "# Security Policy\n\n"
        "Passwords must be rotated every 90 days for all accounts.\n\n"
        "All production logins require multi-factor authentication.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.setenv("CORPUS_DIR", str(corpus))
    monkeypatch.setenv("EMBED_BACKEND", "hash")
    monkeypatch.setenv("EMBED_DIM", "64")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("RETRIEVE_TOP_K", "4")
    config_module.get_settings.cache_clear()

    from ragqa.api import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    config_module.get_settings.cache_clear()


def test_root_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "/query" in r.json()["endpoints"]


def test_health_reports_the_real_configuration(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    caps = body["capabilities"]
    # The stub must be labelled as a stub, loudly.
    assert caps["embeddings"]["is_stub"] is True
    assert caps["embeddings"]["backend"].startswith("hash-stub")
    assert caps["reranker"]["is_stub"] is True
    assert caps["generation"]["configured"] is False
    assert isinstance(caps["vectors"]["backend"], str)


def test_health_can_probe_llm_without_crashing_when_disabled(client):
    r = client.get("/health", params={"probe_llm": True})
    assert r.status_code == 200


def test_ingest_corpus_then_query_retrieval_only(client):
    r = client.post("/ingest", json={"corpus": True})
    assert r.status_code == 200
    report = r.json()
    assert report["counts"]["indexed"] == 1
    assert report["counts"]["errors"] == 0

    r = client.post("/query", json={"question": "how often must passwords be rotated", "mode": "retrieval_only"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "retrieval_only"
    assert body["retrieved"], "expected at least one retrieved chunk"
    assert body["retrieved"][0]["text"]
    assert body["evidence_confidence"] >= 0.0


def test_query_in_answer_mode_without_an_llm_returns_a_clear_status(client):
    client.post("/ingest", json={"corpus": True})
    r = client.post("/query", json={"question": "how often must passwords be rotated"})
    assert r.status_code == 200
    body = r.json()
    # Either the evidence gate fired or generation is off; both must be explicit
    # and must still return the retrieved evidence.
    assert body["status"] in {"abstained_no_llm", "abstained_insufficient_evidence"}
    assert body["abstained"] is True
    assert body["retrieved"]


def test_query_rejects_empty_question(client):
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 422


def test_query_rejects_bad_mode(client):
    r = client.post("/query", json={"question": "x", "mode": "nonsense"})
    assert r.status_code == 422


def test_ingest_unknown_path_returns_400(client):
    r = client.post("/ingest", json={"paths": ["/definitely/not/here.md"]})
    assert r.status_code == 400


def test_ingest_is_incremental_on_second_call(client):
    client.post("/ingest", json={"corpus": True})
    r = client.post("/ingest", json={"corpus": True})
    body = r.json()
    assert body["counts"]["indexed"] == 0
    assert body["counts"]["skipped"] == 1


def test_force_reingests(client):
    client.post("/ingest", json={"corpus": True})
    r = client.post("/ingest", json={"corpus": True, "force": True})
    assert r.json()["counts"]["indexed"] == 1


def test_documents_endpoint_lists_ingested_docs(client):
    client.post("/ingest", json={"corpus": True})
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["title"] == "Security Policy"


def test_upload_rejects_unsupported_extension(client):
    r = client.post(
        "/ingest/upload",
        files={"files": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_upload_ingests_a_markdown_file(client):
    r = client.post(
        "/ingest/upload",
        files={"files": ("note.md", b"# Uploaded Note\n\nThe secret code is 42.\n", "text/markdown")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["indexed"] == 1
    # The temp-dir path is rewritten to the original filename in the response.
    assert body["indexed"][0]["path"] == "note.md"

    hits = client.post("/query", json={"question": "what is the secret code", "mode": "retrieval_only"}).json()
    assert any("42" in c["text"] for c in hits["retrieved"])


def test_upload_path_traversal_filename_is_neutralised(client):
    r = client.post(
        "/ingest/upload",
        files={"files": ("../../escape.md", b"# Escaped\n\ncontent\n", "text/markdown")},
    )
    assert r.status_code == 200
    assert r.json()["indexed"][0]["path"] == "escape.md"


def test_upload_empty_request_is_400(client):
    r = client.post("/ingest/upload", files=[])
    assert r.status_code in (400, 422)
