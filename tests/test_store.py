"""Store behaviour: ingest, keyword + vector search, deletion, incremental skip."""

from __future__ import annotations

import numpy as np
import pytest

from ragqa.chunking import chunk_document
from ragqa.embeddings import HashEmbedder
from ragqa.ingest import load_document
from ragqa.store import SchemaMismatch, SqliteStore


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "idx.sqlite3", vector_backend="auto")
    s.init_schema()
    yield s
    s.close()


def _ingest(store: SqliteStore, path, embedder: HashEmbedder, *, root=None):
    doc = load_document(path, root=root)
    chunks = chunk_document(doc, size=60, overlap=10, min_size=15)
    vectors = embedder.embed_documents([c.text for c in chunks])
    store.upsert_document(doc, chunks, vectors)
    return doc, chunks


def test_init_schema_is_idempotent(tmp_path):
    s = SqliteStore(tmp_path / "i.sqlite3")
    s.init_schema()
    s.init_schema()
    assert s.stats()["documents"] == 0
    s.close()


def test_vector_backend_is_reported(tmp_path):
    s = SqliteStore(tmp_path / "i.sqlite3")
    s.init_schema()
    assert s.vector_backend in ("sqlite-vec", "numpy-brute-force")
    s.close()


def test_forcing_sqlite_backend_when_unavailable_raises(tmp_path, monkeypatch):
    """If an operator explicitly asks for sqlite-vec, a silent fallback is wrong."""
    from ragqa import store as store_module

    monkeypatch.setattr(store_module, "load_sqlite_vec", lambda conn: "simulated missing extension")
    with pytest.raises(store_module.StoreError, match="sqlite-vec unavailable"):
        SqliteStore(tmp_path / "i.sqlite3", vector_backend="sqlite")


def test_auto_backend_falls_back_to_numpy_when_sqlite_vec_is_missing(tmp_path, monkeypatch):
    """The documented fallback must happen -- and be recorded, not hidden."""
    from ragqa import store as store_module

    monkeypatch.setattr(store_module, "load_sqlite_vec", lambda conn: "simulated missing extension")
    s = SqliteStore(tmp_path / "i.sqlite3", vector_backend="auto")
    s.init_schema()
    assert s.vector_backend == "numpy-brute-force"
    assert s.sqlite_vec_error == "simulated missing extension"
    s.close()


def test_bm25_finds_the_document_containing_a_term(store, tmp_path):
    embedder = HashEmbedder(dim=64)
    p = tmp_path / "a.md"
    p.write_text("passwords must be rotated every ninety days", encoding="utf-8")
    q = tmp_path / "b.md"
    q.write_text("the availability objective is measured over thirty days", encoding="utf-8")
    da, ca = _ingest(store, p, embedder, root=tmp_path)
    _ingest(store, q, embedder, root=tmp_path)

    hits = store.search_bm25("passwords rotated", 10)
    assert hits
    top_id, score = hits[0]
    assert top_id in {c.chunk_id for c in ca}
    assert score < 0  # FTS5 bm25 is negative; more negative == better


def test_bm25_respects_limit(store, tmp_path):
    embedder = HashEmbedder(dim=64)
    for i in range(5):
        p = tmp_path / f"d{i}.md"
        p.write_text("alpha beta gamma " * 20, encoding="utf-8")
        _ingest(store, p, embedder, root=tmp_path)
    assert len(store.search_bm25("alpha", 3)) == 3


def test_bm25_on_term_free_query_raises_tokenize_error(store):
    from ragqa.store import TokenizeError

    with pytest.raises(TokenizeError):
        store.search_bm25("!!!", 5)


def test_vector_search_returns_nearest_first(store, tmp_path):
    embedder = HashEmbedder(dim=64)
    p = tmp_path / "v.md"
    p.write_text("deployment rollback restores the previous image quickly", encoding="utf-8")
    _, chunks = _ingest(store, p, embedder, root=tmp_path)

    query_vec = embedder.embed_query("deployment rollback previous image")
    results = store.search_vector(query_vec, 5)
    assert results
    assert results[0][0] in {c.chunk_id for c in chunks}
    assert results[0][1] <= results[-1][1]  # ascending distance


def test_get_chunks_round_trips_offsets_and_page(store, tmp_path):
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "page.md"
    p.write_text("line one\n\n" + "word " * 80, encoding="utf-8")
    _, chunks = _ingest(store, p, embedder, root=tmp_path)
    fetched = store.get_chunks([c.chunk_id for c in chunks])
    assert len(fetched) == len(chunks)
    for c in chunks:
        got = fetched[c.chunk_id]
        assert got.text == c.text
        assert got.char_start == c.char_start
        assert got.char_end == c.char_end


def test_reingesting_a_changed_file_replaces_the_old_document(store, tmp_path):
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "doc.md"
    p.write_text("original content about apples", encoding="utf-8")
    doc1, _ = _ingest(store, p, embedder, root=tmp_path)
    assert store.stats()["documents"] == 1

    p.write_text("completely different content about oranges", encoding="utf-8")
    doc2, _ = _ingest(store, p, embedder, root=tmp_path)
    assert doc1.doc_id != doc2.doc_id  # content-addressed
    # The path is unique, so the stale row must be gone (no duplicate documents).
    assert store.stats()["documents"] == 1
    assert store.get_document(doc1.doc_id) is None
    assert store.get_document(doc2.doc_id) is not None


def test_unchanged_file_is_detected(store, tmp_path):
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "doc.md"
    p.write_text("stable content", encoding="utf-8")
    doc, _ = _ingest(store, p, embedder, root=tmp_path)
    assert store.document_unchanged(doc.path, doc.sha256) == doc.doc_id
    assert store.document_unchanged(doc.path, "different-hash") is None


def test_delete_document_removes_chunks_and_vectors(store, tmp_path):
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "doc.md"
    p.write_text("content to be deleted entirely", encoding="utf-8")
    doc, _ = _ingest(store, p, embedder, root=tmp_path)
    assert store.stats()["vectors"] > 0
    assert store.delete_document(doc.doc_id) is True
    assert store.stats()["documents"] == 0
    assert store.stats()["chunks"] == 0
    assert store.stats()["vectors"] == 0
    assert store.delete_document(doc.doc_id) is False


def test_stats_counts_add_up(store, tmp_path):
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "doc.md"
    p.write_text("word " * 300, encoding="utf-8")
    _, chunks = _ingest(store, p, embedder, root=tmp_path)
    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == len(chunks)
    assert stats["vectors"] == len(chunks)


def test_schema_mismatch_is_refused(store):
    store.finalize_ingest(
        embed_name="m1", embed_stub=True, dim=64, chunk_size=60, chunk_overlap=10, notes=[]
    )
    store.assert_compatible(embed_name="m1", dim=64, chunk_size=60, chunk_overlap=10)  # ok
    with pytest.raises(SchemaMismatch):
        store.assert_compatible(embed_name="m2", dim=64, chunk_size=60, chunk_overlap=10)
    with pytest.raises(SchemaMismatch):
        store.assert_compatible(embed_name="m1", dim=128, chunk_size=60, chunk_overlap=10)
    with pytest.raises(SchemaMismatch):
        store.assert_compatible(embed_name="m1", dim=64, chunk_size=99, chunk_overlap=10)


def test_clear_empties_the_index(store, tmp_path):
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "doc.md"
    p.write_text("content", encoding="utf-8")
    _ingest(store, p, embedder, root=tmp_path)
    assert store.stats()["chunks"] > 0
    store.clear()
    assert store.stats()["chunks"] == 0
    assert store.search_bm25("content", 5) == []


def test_chunks_containing_resolves_evidence_substrings(store, tmp_path):
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "doc.md"
    p.write_text("passwords must be rotated every 90 days for all accounts", encoding="utf-8")
    _, chunks = _ingest(store, p, embedder, root=tmp_path)
    found = store.chunks_containing("rotated every 90 days")
    assert found == [c.chunk_id for c in chunks]
    assert store.chunks_containing("not present anywhere") == []


def test_chunks_containing_is_case_and_whitespace_insensitive(store, tmp_path):
    """Evidence labels are written by hand, so they must match across line wraps."""
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "doc.md"
    p.write_text(
        "progress is 100% complete and under_score is literal\n\n"
        "A data-processing\nagreement must be in place before any data is sent.\n",
        encoding="utf-8",
    )
    _ingest(store, p, embedder, root=tmp_path)
    # Literal punctuation survives.
    assert store.chunks_containing("100%")
    assert store.chunks_containing("under_score")
    # A phrase written with a space matches text that contains a newline there.
    assert store.chunks_containing("data-processing agreement must be in place")
    # Case is ignored.
    assert store.chunks_containing("DATA-PROCESSING Agreement")
    # And a genuine non-match still returns nothing.
    assert store.chunks_containing("no such phrase in the corpus") == []


def test_evidence_cache_is_invalidated_on_write(store, tmp_path):
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "doc.md"
    p.write_text("first version mentions apples only", encoding="utf-8")
    _ingest(store, p, embedder, root=tmp_path)
    assert store.chunks_containing("apples")
    assert store.chunks_containing("oranges") == []

    p.write_text("second version mentions oranges only", encoding="utf-8")
    _ingest(store, p, embedder, root=tmp_path)
    # Stale cache would still report apples here.
    assert store.chunks_containing("oranges")
    assert store.chunks_containing("apples") == []


def test_numpy_fallback_backend_works(tmp_path):
    """The documented fallback must be a first-class citizen, not an afterthought."""
    s = SqliteStore(tmp_path / "np.sqlite3", vector_backend="numpy")
    s.init_schema()
    assert s.vector_backend == "numpy-brute-force"
    embedder = HashEmbedder(dim=16)
    p = tmp_path / "d.md"
    p.write_text("alpha beta gamma delta epsilon", encoding="utf-8")
    doc = load_document(p, root=tmp_path)
    chunks = chunk_document(doc, size=40, overlap=5, min_size=5)
    vectors = embedder.embed_documents([c.text for c in chunks])
    s.upsert_document(doc, chunks, vectors)
    hits = s.search_vector(embedder.embed_query("alpha beta"), 3)
    assert hits and hits[0][0] in {c.chunk_id for c in chunks}
    s.close()


def test_sqlite_vec_and_numpy_backends_agree_on_the_top_hit(tmp_path):
    """Cross-check the two backends: they must at least agree on the nearest chunk."""
    embedder = HashEmbedder(dim=32)
    p = tmp_path / "d.md"
    p.write_text(" ".join(["deployment", "rollback", "image"] * 20), encoding="utf-8")
    doc = load_document(p, root=tmp_path)
    chunks = chunk_document(doc, size=40, overlap=5, min_size=5)
    vectors = embedder.embed_documents([c.text for c in chunks])
    query = embedder.embed_query("deployment rollback image")

    results = {}
    for backend in ("numpy", "sqlite"):
        try:
            s = SqliteStore(tmp_path / f"{backend}.sqlite3", vector_backend=backend)
        except Exception:
            continue  # sqlite-vec not available in this environment
        s.init_schema()
        s.upsert_document(doc, chunks, vectors)
        hits = s.search_vector(query, 3)
        results[backend] = hits[0][0]
        s.close()
    if len(results) == 2:
        assert results["numpy"] == results["sqlite"]
