"""Runtime configuration.

Everything is environment-driven so the same image can run locally (SQLite file +
Ollama) or against a hosted OpenAI-compatible endpoint without code changes.

Design note: we deliberately keep *one* flat Settings object instead of a config
tree. The service is small enough that a flat object stays readable, and it makes
`/health` trivial to render (dump the settings). When this grows past ~40 knobs,
split it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, computed from this file's location (src/ragqa/config.py -> repo root).
# Used to resolve default relative paths so the service works from any CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ storage
    db_path: Path = Field(default=REPO_ROOT / "data" / "index.sqlite3")
    corpus_dir: Path = Field(default=REPO_ROOT / "data" / "corpus")

    # Which vector backend to use.
    #   auto   -> sqlite-vec if it loads, else numpy brute force (default)
    #   sqlite -> force vec0 (fail loudly if the extension is missing)
    #   numpy  -> force the brute-force fallback
    vector_backend: str = Field(default="auto", pattern="^(auto|sqlite|numpy)$")

    # Per-document incremental ingestion: skip files whose sha256 already matches
    # the indexed copy. Set false to force a full rebuild on every ingest.
    incremental_ingest: bool = True

    # --------------------------------------------------------------- chunking
    # "tokens" here are whitespace-split words: a deliberate, documented proxy for
    # real subword tokens, because it keeps chunking dependency-free and stable
    # across embedder choices. See docs/architecture.md.
    chunk_size_words: int = Field(default=220, ge=32, le=2000)
    chunk_overlap_words: int = Field(default=40, ge=0, le=500)
    chunk_min_words: int = Field(default=40, ge=1)

    # ------------------------------------------------------------- embeddings
    # auto -> fastembed if importable+model loads, else the labelled hash STUB.
    embed_backend: str = Field(default="auto", pattern="^(auto|fastembed|hash)$")
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = Field(default=384, ge=8, le=4096)  # used by the hash stub
    embed_batch_size: int = Field(default=32, ge=1, le=512)

    # ---------------------------------------------------------------- retrieval
    bm25_top_k: int = Field(default=50, ge=1, le=1000)
    vector_top_k: int = Field(default=50, ge=1, le=1000)
    # Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper
    # and is insensitive in practice; exposed so it can be tuned per corpus.
    rrf_k: int = Field(default=60, ge=1, le=1000)
    fusion_top_k: int = Field(default=50, ge=1, le=1000)
    retrieve_top_k: int = Field(default=8, ge=1, le=100)  # chunks handed to the LLM

    # ---------------------------------------------------------------- re-ranking
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = Field(default=50, ge=1, le=500)  # candidates sent to the cross-encoder
    # If a cross-encoder score is below this, the top result is treated as weak
    # evidence and (unless the LLM overrides) we abstain. Only used when a real
    # cross-encoder is active; see generation.evidence_confidence.
    rerank_min_score: float = 0.0

    # ---------------------------------------------------------------- generation
    llm_enabled: bool = True
    # Default points at a local Ollama OpenAI-compatible endpoint. Swap for any
    # OpenAI-compatible gateway (vLLM, llama.cpp server, OpenAI, Azure, Groq...).
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "LLM_BASE_URL"),
    )
    llm_api_key: str = Field(
        default="not-needed",
        validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_API_KEY"),
    )
    llm_model: str = Field(default="qwen2.5:7b-instruct")
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=600, ge=16)
    # Passthrough for OpenAI-compatible `reasoning_effort`. Thinking models
    # (e.g. Qwen3.5 served by Ollama) otherwise spend the whole token budget on
    # their reasoning channel and return empty `content`. Setting this to "none"
    # disables that. Left unset by default so non-reasoning endpoints are untouched.
    llm_reasoning_effort: str | None = None
    # Arbitrary extra JSON merged into the chat-completions request body, for
    # gateway-specific parameters (vLLM `chat_template_kwargs`, vendor flags...).
    # Example: LLM_EXTRA_BODY={"top_p": 0.9}
    llm_extra_body: dict[str, Any] = Field(default_factory=dict)

    # The exact phrase the model must emit when it cannot answer from the
    # retrieved context. Kept as config (not a magic string) so the abstention
    # check and the system prompt can never drift apart.
    abstain_phrase: str = "I don't have enough information in the provided documents to answer that."

    # Pre-generation abstention gate: minimum evidence confidence in [0, 1].
    # 0 disables the gate (the LLM's own abstention still applies).
    min_evidence_confidence: float = Field(default=0.15, ge=0.0, le=1.0)

    # ------------------------------------------------------------------- api
    api_title: str = "rag-doc-qa"
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)

    def resolved(self) -> "Settings":
        """Return a copy with paths made absolute (callers shouldn't guess CWD)."""
        clone = self.model_copy(deep=True)
        clone.db_path = _absolute(clone.db_path)
        clone.corpus_dir = _absolute(clone.corpus_dir)
        return clone


def _absolute(p: Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached so tests can `get_settings.cache_clear()`)."""
    return Settings().resolved()
