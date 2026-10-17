"""Embedding backends.

Two backends ship:

* ``fastembed`` -- real ONNX bi-encoder (default ``BAAI/bge-small-en-v1.5``,
  384-dim). Runs on CPU via onnxruntime, no torch. This is the production path.
* ``hash`` -- a deterministic hashing-trick STUB. It is **not a semantic model**
  and must never be presented as one. It exists so the pipeline, the API, the
  evaluation harness, and the tests can all run in environments where the ONNX
  wheels or the model download are unavailable. ``Embedder.is_stub`` is True and
  the API/health/README surface that fact.

Vectors are always L2-normalised. sqlite-vec's default metric is L2; for
unit-length vectors L2 is a monotonic function of cosine distance
(||a-b||^2 = 2 - 2 cos), so ranking by L2 == ranking by cosine. Normalising here
means the store never has to care which metric it is.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


class Embedder(ABC):
    """Interface every embedding backend implements."""

    name: str = "unknown"
    dim: int = 0
    is_stub: bool = False

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised document vectors."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 L2-normalised query vector.

        Kept separate from `embed_documents` because asymmetric retrievers
        (e.g. BGE) prepend a query-side instruction that must NOT be applied to
        documents; collapsing the two would silently degrade recall.
        """

    def embed_documents_iter(self, texts: Sequence[str], batch_size: int = 32) -> Iterable[np.ndarray]:
        for i in range(0, len(texts), batch_size):
            yield self.embed_documents(texts[i : i + batch_size])


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows are left as zeros (no NaNs)."""
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


# --------------------------------------------------------------------------- real


class FastEmbedEmbedder(Embedder):
    """ONNX bi-encoder via fastembed. No torch dependency."""

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on env
            raise EmbeddingError("fastembed is not installed") from exc
        self.name = model_name
        self.batch_size = batch_size
        try:
            self._model = TextEmbedding(model_name=model_name)
        except Exception as exc:
            raise EmbeddingError(f"failed to load fastembed model {model_name!r}: {exc}") from exc
        # Probe the real output dimension rather than trusting config.
        probe = np.asarray(list(self._model.embed(["dimension probe"]))[0], dtype=np.float32)
        self.dim = int(probe.shape[0])
        self.is_stub = False

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self._model.embed(list(texts), batch_size=self.batch_size))
        return l2_normalize(np.vstack(vecs))

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed applies the model's query-side prefix when it has one.
        try:
            vec = list(self._model.query_embed([text]))[0]
        except Exception:  # pragma: no cover - older fastembed
            vec = list(self._model.embed([text]))[0]
        return l2_normalize(np.asarray(vec, dtype=np.float32).reshape(1, -1))[0]


# --------------------------------------------------------------------------- stub


class HashEmbedder(Embedder):
    """Deterministic hashing-trick embedder. **STUB -- NOT A SEMANTIC MODEL.**

    Why it exists: CI and air-gapped machines should still be able to exercise
    ingestion, storage, fusion, the API, and the eval harness end to end.

    Why it must be labelled: it has no notion of meaning. It scores lexical
    overlap only (like a bag-of-words), so any retrieval-quality number produced
    with it is *not* evidence of a working semantic retriever. The README and the
    eval report both refuse to present hash-backed numbers as quality metrics.
    """

    is_stub = True

    def __init__(self, dim: int = 384) -> None:
        self.dim = max(8, int(dim))
        self.name = f"hash-stub-v1-dim{self.dim}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        # Unigrams + bigrams: bigrams give a weak word-order signal so the stub is
        # not *quite* a pure bag of words. Still not semantic.
        grams = list(words)
        grams += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return grams

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        (val,) = struct.unpack("<Q", digest)
        idx = val % self.dim
        sign = 1.0 if (val >> 63) & 1 == 0 else -1.0
        return int(idx), sign

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            idx, sign = self._bucket(tok)
            vec[idx] += sign
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.vstack([self._embed_one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._embed_one(text).reshape(1, -1))[0]


# --------------------------------------------------------------------------- factory


def build_embedder(
    *,
    backend: str = "auto",
    model_name: str = "BAAI/bge-small-en-v1.5",
    hash_dim: int = 384,
    batch_size: int = 32,
) -> tuple[Embedder, list[str]]:
    """Build an embedder. Returns (embedder, notes) where notes explain the choice.

    `notes` is surfaced in /health and the eval report so a reader can always
    tell whether real embeddings or the stub produced a given result.
    """
    notes: list[str] = []
    backend = backend.lower()

    if backend in ("auto", "fastembed"):
        try:
            emb = FastEmbedEmbedder(model_name, batch_size=batch_size)
            notes.append(f"embeddings: fastembed {model_name} (dim={emb.dim})")
            return emb, notes
        except EmbeddingError as exc:
            msg = f"embeddings: fastembed unavailable ({exc})"
            if backend == "fastembed":
                raise
            notes.append(msg + "; falling back to hash STUB")
            log.warning(msg)

    emb = HashEmbedder(hash_dim)
    notes.append(
        f"embeddings: HASH STUB (dim={emb.dim}) -- NOT semantic; "
        "quality metrics from this backend are meaningless"
    )
    return emb, notes
