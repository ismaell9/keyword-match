"""SQLite storage: documents, chunks, BM25 (FTS5) and vectors in one file.

Why one file for everything: the whole promise of this design is that you can
`cp data/index.sqlite3 backup-$(date).sqlite3` and have a complete, consistent
snapshot of the search index -- keyword index, vectors, and provenance. That is
the operational property an enterprise reviewer looks for, and it is impossible
to get wrong (no index/config skew) because there is nothing to keep in sync.

Concurrency model: WAL mode, one connection per thread (thread-local), and a
process-wide write lock. SQLite allows many concurrent readers with WAL, and
serialising writes is correct and fast at this scale. This is deliberately a
single-writer design; see docs/architecture.md for when to move to Postgres.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .types import Chunk, Document
from .vectorindex import (
    NumpyBruteForceIndex,
    SqliteVecIndex,
    VectorIndex,
    load_sqlite_vec,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

# FTS5 tokenizer choice: porter stemming (so "rotate"/"rotated"/"rotating" match)
# over unicode61's plain tokenization. The trade-off is slightly less precise
# phrase matching; for a BM25 leg whose job is recall, stemming wins.
_FTS_TOKENIZER = "porter unicode61"

_DDL = f"""
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    path       TEXT NOT NULL UNIQUE,
    title      TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    n_chunks   INTEGER NOT NULL DEFAULT 0,
    meta       TEXT NOT NULL DEFAULT '{{}}',
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   INTEGER PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    page       INTEGER,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    n_words    INTEGER NOT NULL,
    UNIQUE (doc_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- External-content FTS5: the text lives once (in `chunks`); FTS stores only the
-- inverted index and reads text back from `chunks` when needed.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='{_FTS_TOKENIZER}'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.chunk_id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.chunk_id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.chunk_id, new.text);
END;
"""


class StoreError(Exception):
    pass


class SchemaMismatch(StoreError):
    """Raised when the index was built with a different embedder/chunker than now."""


class TokenizeError(StoreError):
    """Raised when a user query cannot be turned into a safe FTS5 MATCH expression."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_fts_query(text: str, *, mode: str = "or") -> str:
    """Turn free text into a *safe* FTS5 MATCH expression.

    FTS5 MATCH is not SQL (no placeholders inside it) and it has its own grammar
    where characters like `"`, `*`, `-`, `:` and `NEAR` have meaning. Passing a
    raw user question straight into MATCH is both a correctness bug and an
    injection surface. We therefore extract alphanumeric word tokens, double-quote
    each one (which escapes it as a quoted string in FTS5), and join with OR.

    OR (not AND) is deliberate: this is the *recall* leg. Precision is the
    reranker's job, and requiring every term would drop chunks that answer the
    question with different vocabulary than the question used.
    """
    words = [w for w in _split_words(text) if len(w) > 1]
    if not words:
        raise TokenizeError("query contains no searchable terms")
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(f'"{w}"' for w in words)


def _split_words(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def normalize_ws(text: str) -> str:
    """Lowercase and collapse every whitespace run to a single space.

    Used for evidence matching so that a label written by a human matches chunk
    text that contains the source document's line breaks.
    """
    return " ".join(text.lower().split())


class SqliteStore:
    """Everything the retrieval and ingestion layers need, over one SQLite file."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        vector_backend: str = "auto",
        create: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._vector_backend_pref = vector_backend
        self._sqlite_vec_error: str | None = None
        self._vector_index: VectorIndex | None = None
        self._dim: int | None = None
        # Lazily-built cache of whitespace-normalised chunk text for evidence
        # matching; invalidated on every write.
        self._norm_cache: list[tuple[int, str, str]] | None = None

        if create:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.db_path.exists():
            raise StoreError(f"index not found at {self.db_path}")

        self._resolve_vector_backend()

    # ------------------------------------------------------------------ plumbing

    def _resolve_vector_backend(self) -> None:
        """Decide sqlite-vec vs numpy once, up front, and record why."""
        pref = self._vector_backend_pref
        probe = sqlite3.connect(":memory:")
        try:
            err = load_sqlite_vec(probe)
        finally:
            probe.close()
        self._sqlite_vec_error = err

        if pref in ("auto", "sqlite") and err is None:
            self._vector_index = SqliteVecIndex()
        elif pref == "sqlite":
            raise StoreError(f"vector_backend='sqlite' requested but sqlite-vec unavailable: {err}")
        else:
            self._vector_index = NumpyBruteForceIndex()
            if pref == "numpy":
                log.info("using numpy brute-force vector index (explicitly requested)")
            else:
                log.warning("sqlite-vec unavailable (%s); falling back to numpy brute force", err)

    @property
    def vector_index(self) -> VectorIndex:
        assert self._vector_index is not None
        return self._vector_index

    @property
    def vector_backend(self) -> str:
        return self.vector_index.name

    @property
    def sqlite_vec_error(self) -> str | None:
        return self._sqlite_vec_error

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        if self.vector_backend == "sqlite-vec":
            err = load_sqlite_vec(conn)
            if err is not None:  # pragma: no cover - would be a mid-life failure
                raise StoreError(f"sqlite-vec was available at startup but failed to reload: {err}")
        return conn

    def conn(self) -> sqlite3.Connection:
        """Thread-local connection. Callers must not close it."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._new_connection()
            self._local.conn = conn
        return conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Serialise writes process-wide and wrap them in a transaction."""
        conn = self.conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # -------------------------------------------------------------------- schema

    def init_schema(self) -> None:
        conn = self.conn()
        with self._write_lock:
            conn.executescript("BEGIN; " + _DDL + " COMMIT;")
        self._set_meta_default("schema_version", SCHEMA_VERSION)

    def _set_meta_default(self, key: str, value: str) -> None:
        with self.write() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO NOTHING",
                (key, value),
            )

    def set_meta(self, key: str, value: str) -> None:
        with self.write() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def meta_dict(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn().execute("SELECT key, value FROM meta")}

    def assert_compatible(self, *, embed_name: str, dim: int, chunk_size: int, chunk_overlap: int) -> None:
        """Refuse to mix indexes built with a different embedder/chunker.

        Silent mixing produces a vector space where distances are meaningless and
        recall quietly collapses -- a failure that looks like "the model got
        worse" and wastes hours. We fail loudly instead, with the fix in the
        message.
        """
        existing_dim = self.get_meta("embed_dim")
        existing_model = self.get_meta("embed_model")
        existing_size = self.get_meta("chunk_size_words")
        existing_overlap = self.get_meta("chunk_overlap_words")

        if existing_dim is None:
            return  # fresh index
        problems: list[str] = []
        if int(existing_dim) != int(dim):
            problems.append(f"embed_dim {existing_dim} != {dim}")
        if existing_model and existing_model != embed_name:
            problems.append(f"embed_model {existing_model!r} != {embed_name!r}")
        if existing_size and int(existing_size) != int(chunk_size):
            problems.append(f"chunk_size_words {existing_size} != {chunk_size}")
        if existing_overlap and int(existing_overlap) != int(chunk_overlap):
            problems.append(f"chunk_overlap_words {existing_overlap} != {chunk_overlap}")
        if problems:
            raise SchemaMismatch(
                "index was built with different settings: "
                + "; ".join(problems)
                + f". Rebuild with `--rebuild` (deletes {self.db_path}) or point DB_PATH elsewhere."
            )

    # ------------------------------------------------------------------- ingest

    def document_unchanged(self, path: str, sha256: str) -> str | None:
        """Return the existing doc_id if this exact content is already indexed."""
        row = self.conn().execute(
            "SELECT doc_id, sha256 FROM documents WHERE path = ?", (path,)
        ).fetchone()
        if row and row["sha256"] == sha256:
            return str(row["doc_id"])
        return None

    def upsert_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: np.ndarray | None,
    ) -> dict[str, int]:
        """Replace any prior version of this document (by path or doc_id) and insert the new one.

        Returns counts. Vectors are optional so the keyword-only smoke path can
        still store documents (vector search will simply return nothing for them).
        """
        conn = self.conn()
        dim = int(vectors.shape[1]) if vectors is not None and vectors.size else None
        with self.write() as w:
            # A changed file gets a new doc_id (content-addressed), so clear by
            # path as well as by doc_id, otherwise the old copy would linger.
            old_rows = w.execute(
                "SELECT doc_id FROM documents WHERE path = ? OR doc_id = ?",
                (document.path, document.doc_id),
            ).fetchall()
            old_ids = [str(r["doc_id"]) for r in old_rows]
            for old in old_ids:
                self._delete_document_tx(w, old)

            w.execute(
                "INSERT INTO documents(doc_id, path, title, sha256, page_count, n_chunks, meta, indexed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document.doc_id,
                    document.path,
                    document.title,
                    document.sha256,
                    document.page_count,
                    len(chunks),
                    json.dumps(document.meta),
                    _now(),
                ),
            )
            inserted: list[int] = []
            for chunk in chunks:
                # chunk_id is intentionally NOT inserted: chunk.chunk_id is a
                # placeholder (-1) from the chunker, and letting SQLite assign the
                # rowid is both simpler and race-free.
                cur = w.execute(
                    "INSERT INTO chunks(doc_id, ordinal, text, page, char_start, char_end, n_words)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        document.doc_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.page,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.n_words,
                    ),
                )
                cid = int(cur.lastrowid)
                chunk.chunk_id = cid  # back-fill so callers can index vectors by id
                inserted.append(cid)

            if vectors is not None and len(inserted):
                self._dim = dim or self._dim
                self.vector_index.ensure(w, dim or 0)
                self.vector_index.add(w, inserted, vectors)

        self._norm_cache = None
        return {"chunks": len(chunks), "replaced_documents": len(old_ids)}

    def _delete_document_tx(self, conn: sqlite3.Connection, doc_id: str) -> None:
        """Delete a document and everything derived from it. Caller holds the write lock."""
        rows = conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
        chunk_ids = [int(r["chunk_id"]) for r in rows]
        if chunk_ids:
            self.vector_index.remove(conn, chunk_ids)
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def delete_document(self, doc_id: str) -> bool:
        with self.write() as w:
            exists = w.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if not exists:
                return False
            self._delete_document_tx(w, doc_id)
        self._norm_cache = None
        return True

    def clear(self) -> None:
        """Drop all content (used by `--rebuild`)."""
        conn = self.conn()
        with self._write_lock:
            conn.executescript("BEGIN; DROP TABLE IF EXISTS chunks_fts; COMMIT;")
            for table in ("chunks", "documents", "meta", "chunk_vectors"):
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
                except sqlite3.OperationalError:
                    pass
            if self.vector_backend == "sqlite-vec":
                try:
                    conn.execute("DROP TABLE IF EXISTS vec_chunks")
                except sqlite3.OperationalError:  # pragma: no cover
                    pass
            conn.execute("VACUUM")
        if isinstance(self.vector_index, NumpyBruteForceIndex):
            self.vector_index = NumpyBruteForceIndex()
        self._norm_cache = None
        self.init_schema()

    def finalize_ingest(self, *, embed_name: str, embed_stub: bool, dim: int, chunk_size: int, chunk_overlap: int, notes: list[str]) -> None:
        """Persist the build fingerprint so later runs can detect a mismatch."""
        self.set_meta("embed_model", embed_name)
        self.set_meta("embed_is_stub", "true" if embed_stub else "false")
        self.set_meta("embed_dim", str(dim))
        self.set_meta("chunk_size_words", str(chunk_size))
        self.set_meta("chunk_overlap_words", str(chunk_overlap))
        self.set_meta("vector_backend", self.vector_backend)
        self.set_meta("build_notes", json.dumps(notes))
        self.set_meta("updated_at", _now())
        self._dim = dim

    # -------------------------------------------------------------------- reads

    def search_bm25(self, query: str, k: int, *, mode: str = "or") -> list[tuple[int, float]]:
        """BM25 keyword search. Returns [(chunk_id, bm25)] with bm25 ascending (best first).

        FTS5's `bm25()` returns *negative* scores where more negative == more
        relevant, so ascending order is best-first. We return the raw value and
        let the fusion layer use ranks (which is scale-free and therefore safer
        than trying to normalise BM25 against cosine distances).
        """
        match = build_fts_query(query, mode=mode)
        rows = self.conn().execute(
            "SELECT f.rowid AS chunk_id, bm25(chunks_fts) AS score "
            "FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY score ASC LIMIT ?",
            (match, int(k)),
        ).fetchall()
        return [(int(r["chunk_id"]), float(r["score"])) for r in rows]

    def search_vector(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Vector search. Returns [(chunk_id, distance)] best-first."""
        if self.vector_index.count(self.conn()) == 0:
            return []
        return self.vector_index.search(self.conn(), query_vector, k)

    def get_chunks(self, chunk_ids: Sequence[int]) -> dict[int, Chunk]:
        """Fetch chunks by id, preserving nothing about order (callers re-sort)."""
        if not chunk_ids:
            return {}
        out: dict[int, Chunk] = {}
        # SQLite has a 999-variable default limit; chunk the id list defensively.
        ids = list(dict.fromkeys(int(i) for i in chunk_ids))
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn().execute(
                f"SELECT chunk_id, doc_id, ordinal, text, page, char_start, char_end, n_words "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                batch,
            ).fetchall()
            for r in rows:
                out[int(r["chunk_id"])] = Chunk(
                    chunk_id=int(r["chunk_id"]),
                    doc_id=str(r["doc_id"]),
                    ordinal=int(r["ordinal"]),
                    text=str(r["text"]),
                    page=int(r["page"]) if r["page"] is not None else None,
                    char_start=int(r["char_start"]),
                    char_end=int(r["char_end"]),
                    n_words=int(r["n_words"]),
                )
        return out

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.conn().execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["meta"] = json.loads(d.get("meta") or "{}")
        except json.JSONDecodeError:
            d["meta"] = {}
        return d

    def document_titles(self, doc_ids: Sequence[str]) -> dict[str, str]:
        if not doc_ids:
            return {}
        ids = list(dict.fromkeys(doc_ids))
        out: dict[str, str] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn().execute(
                f"SELECT doc_id, title FROM documents WHERE doc_id IN ({placeholders})", batch
            ).fetchall()
            for r in rows:
                out[str(r["doc_id"])] = str(r["title"])
        return out

    def document_paths(self, doc_ids: Sequence[str]) -> dict[str, str]:
        if not doc_ids:
            return {}
        ids = list(dict.fromkeys(doc_ids))
        out: dict[str, str] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn().execute(
                f"SELECT doc_id, path FROM documents WHERE doc_id IN ({placeholders})", batch
            ).fetchall()
            for r in rows:
                out[str(r["doc_id"])] = str(r["path"])
        return out

    def all_chunk_ids(self) -> list[int]:
        return [int(r["chunk_id"]) for r in self.conn().execute("SELECT chunk_id FROM chunks")]

    def chunks_containing(self, needle: str) -> list[int]:
        """Chunk ids whose text contains `needle`, ignoring case and whitespace runs.

        Used by the evaluation harness to resolve *evidence strings* into the set
        of relevant chunks at eval time. Defining relevance by evidence text
        rather than by hard-coded chunk ids keeps the labelled dataset valid when
        chunk size or overlap changes -- otherwise every labelling decision would
        silently rot the moment anyone tuned the chunker.

        Whitespace-insensitivity is not a nicety: source documents wrap lines, so
        a human writing an evidence phrase ("data-processing agreement must be in
        place") will almost never reproduce the embedded newline. Matching on a
        newline-collapsed form is what makes the labels durable.

        Cost: a linear scan of chunk texts, cached in memory (see
        `_normalized_chunks`). Acceptable for evaluation on corpora of this size
        and called out in docs/evaluation.md.
        """
        norm = normalize_ws(needle)
        if not norm:
            return []
        return [cid for cid, _doc, text in self._normalized_chunks() if norm in text]

    def doc_ids_containing(self, needle: str) -> list[str]:
        norm = normalize_ws(needle)
        if not norm:
            return []
        seen: list[str] = []
        for _cid, doc_id, text in self._normalized_chunks():
            if norm in text and doc_id not in seen:
                seen.append(doc_id)
        return seen

    def _normalized_chunks(self) -> list[tuple[int, str, str]]:
        """[(chunk_id, doc_id, lowercased whitespace-collapsed text)], cached.

        Invalidated on any write (upsert/delete/clear) so the cache can never
        serve stale evidence matches.
        """
        if self._norm_cache is None:
            rows = self.conn().execute("SELECT chunk_id, doc_id, text FROM chunks").fetchall()
            self._norm_cache = [
                (int(r["chunk_id"]), str(r["doc_id"]), normalize_ws(str(r["text"]))) for r in rows
            ]
        return self._norm_cache

    def stats(self) -> dict[str, Any]:
        conn = self.conn()
        docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        words = conn.execute("SELECT coalesce(sum(n_words), 0) AS n FROM chunks").fetchone()["n"]
        return {
            "documents": int(docs),
            "chunks": int(chunks),
            "words": int(words),
            "vectors": int(self.vector_index.count(conn)),
            "vector_backend": self.vector_backend,
            "db_path": str(self.db_path),
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "meta": self.meta_dict(),
        }

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self.conn().execute(
            "SELECT doc_id, path, title, sha256, page_count, n_chunks, indexed_at "
            "FROM documents ORDER BY path"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
