"""Vector index backends, behind a single interface.

Two implementations:

* :class:`SqliteVecIndex` -- the ``vec0`` virtual table from the ``sqlite-vec``
  extension. This is the intended production path: the index lives in the same
  SQLite file as the keyword index and the metadata, so there is exactly one
  artefact to back up, snapshot, or ship.
* :class:`NumpyBruteForceIndex` -- a fallback that stores raw float32 blobs and
  scans them with numpy. Used when the extension cannot be loaded. It is
  O(n) per query and holds the matrix in memory; that is fine for the corpora
  this project targets (hundreds to low thousands of chunks) and is stated as a
  limitation rather than hidden.

Metric note: both backends operate on L2-normalised vectors. ``sqlite-vec``
returns L2 distance; the numpy backend returns ``1 - cosine_similarity``. For
unit vectors these are monotonic transforms of each other, so result *ranking*
is identical. The two distance numbers are therefore not directly comparable to
each other -- only within a backend -- and the API reports which backend ran.
"""

from __future__ import annotations

import logging
import sqlite3
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


class VectorIndexError(Exception):
    pass


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


class VectorIndex(ABC):
    """Minimal index interface. All methods take the caller's connection."""

    name: str = "unknown"

    @abstractmethod
    def ensure(self, conn: sqlite3.Connection, dim: int) -> None:
        """Create the index structures for `dim`-dimensional vectors if absent."""

    @abstractmethod
    def add(self, conn: sqlite3.Connection, chunk_ids: list[int], vectors: np.ndarray) -> None:
        """Insert vectors. Must be idempotent per chunk_id (upsert semantics)."""

    @abstractmethod
    def remove(self, conn: sqlite3.Connection, chunk_ids: list[int]) -> None:
        """Delete vectors for the given chunk ids."""

    @abstractmethod
    def search(self, conn: sqlite3.Connection, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(chunk_id, distance)] ordered by ascending distance (best first)."""

    @abstractmethod
    def count(self, conn: sqlite3.Connection) -> int:
        """Number of indexed vectors."""


class SqliteVecIndex(VectorIndex):
    name = "sqlite-vec"

    _TABLE = "vec_chunks"

    def ensure(self, conn: sqlite3.Connection, dim: int) -> None:
        # vec0 tables are typed at creation time (float[N]); a dimension change
        # requires a rebuild, which the store enforces via its meta table.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._TABLE} "
            f"USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{int(dim)}])"
        )

    def add(self, conn: sqlite3.Connection, chunk_ids: list[int], vectors: np.ndarray) -> None:
        if not chunk_ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunk_ids):
            raise VectorIndexError("chunk_ids and vectors length mismatch")
        # vec0 rejects plain INSERT on conflict, so delete-then-insert keeps this
        # method idempotent (which re-ingesting a changed document relies on).
        conn.executemany(f"DELETE FROM {self._TABLE} WHERE chunk_id = ?", [(i,) for i in chunk_ids])
        conn.executemany(
            f"INSERT INTO {self._TABLE}(chunk_id, embedding) VALUES (?, ?)",
            [(cid, _to_blob(v)) for cid, v in zip(chunk_ids, vectors, strict=True)],
        )

    def remove(self, conn: sqlite3.Connection, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        conn.executemany(f"DELETE FROM {self._TABLE} WHERE chunk_id = ?", [(i,) for i in chunk_ids])

    def search(self, conn: sqlite3.Connection, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        # `k = ?` is the vec0 KNN constraint; ORDER BY distance makes the ordering
        # explicit even though vec0 already returns nearest-first.
        rows = conn.execute(
            f"SELECT chunk_id, distance FROM {self._TABLE} "
            f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_to_blob(query), int(k)),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def count(self, conn: sqlite3.Connection) -> int:
        try:
            return int(conn.execute(f"SELECT count(*) FROM {self._TABLE}").fetchone()[0])
        except sqlite3.OperationalError:
            return 0


class NumpyBruteForceIndex(VectorIndex):
    """Fallback: float32 blobs + in-memory matrix, scanned with numpy."""

    name = "numpy-brute-force"

    _TABLE = "chunk_vectors"

    def __init__(self) -> None:
        self._dirty = True
        self._ids: np.ndarray | None = None
        self._mat: np.ndarray | None = None
        self._dim: int | None = None

    def ensure(self, conn: sqlite3.Connection, dim: int) -> None:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._TABLE} ("
            "  chunk_id INTEGER PRIMARY KEY,"
            "  dim INTEGER NOT NULL,"
            "  vec BLOB NOT NULL"
            ")"
        )
        self._dim = int(dim)

    def add(self, conn: sqlite3.Connection, chunk_ids: list[int], vectors: np.ndarray) -> None:
        if not chunk_ids:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[0] != len(chunk_ids):
            raise VectorIndexError("chunk_ids and vectors length mismatch")
        dim = int(vectors.shape[1])
        rows = [(int(cid), dim, _to_blob(v)) for cid, v in zip(chunk_ids, vectors, strict=True)]
        # ON CONFLICT makes re-ingestion idempotent.
        conn.executemany(
            f"INSERT INTO {self._TABLE}(chunk_id, dim, vec) VALUES (?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            rows,
        )
        self._dirty = True

    def remove(self, conn: sqlite3.Connection, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        conn.executemany(f"DELETE FROM {self._TABLE} WHERE chunk_id = ?", [(i,) for i in chunk_ids])
        self._dirty = True

    def _load(self, conn: sqlite3.Connection) -> None:
        if not self._dirty and self._mat is not None:
            return
        rows = conn.execute(f"SELECT chunk_id, dim, vec FROM {self._TABLE}").fetchall()
        if not rows:
            self._ids = np.zeros((0,), dtype=np.int64)
            self._mat = np.zeros((0, self._dim or 0), dtype=np.float32)
            self._dirty = False
            return
        dims = {int(r[1]) for r in rows}
        if len(dims) != 1:
            raise VectorIndexError(f"mixed vector dimensions in {self._TABLE}: {sorted(dims)}")
        dim = dims.pop()
        self._dim = dim
        self._ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
        self._mat = np.vstack([_from_blob(r[2], dim) for r in rows]).astype(np.float32)
        # Cache is valid until the next add/remove; see docs/architecture.md.
        self._dirty = False

    def search(self, conn: sqlite3.Connection, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        self._load(conn)
        assert self._ids is not None and self._mat is not None
        if self._mat.shape[0] == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        # Vectors are unit length, so dot == cosine similarity.
        sims = self._mat @ q
        k = min(int(k), sims.shape[0])
        # argpartition for the top-k, then sort just those k (avoids a full sort).
        if k < sims.shape[0]:
            top = np.argpartition(-sims, k - 1)[:k]
        else:
            top = np.arange(sims.shape[0])
        order = top[np.argsort(-sims[top])]
        return [(int(self._ids[i]), float(1.0 - sims[i])) for i in order]

    def count(self, conn: sqlite3.Connection) -> int:
        try:
            return int(conn.execute(f"SELECT count(*) FROM {self._TABLE}").fetchone()[0])
        except sqlite3.OperationalError:
            return 0


def load_sqlite_vec(conn: sqlite3.Connection) -> str | None:
    """Try to load the sqlite-vec extension onto `conn`.

    Returns None on success, or a human-readable reason string on failure. The
    reason is propagated to /health and the ingest report so a numpy fallback is
    always visible rather than mysterious.
    """
    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except ImportError as exc:
        return f"sqlite_vec package not importable: {exc}"
    try:
        if not hasattr(conn, "enable_load_extension"):
            return "this Python build has SQLite extension loading disabled"
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
        return None
    except Exception as exc:  # pragma: no cover - platform dependent
        return f"could not load sqlite_vec extension: {exc}"


__all__ = [
    "NumpyBruteForceIndex",
    "SqliteVecIndex",
    "VectorIndex",
    "VectorIndexError",
    "load_sqlite_vec",
]
