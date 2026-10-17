# Architecture

This document describes how `rag-doc-qa` is put together and, more importantly,
*why* each piece is shaped the way it is. `SPEC.md` states the requirements; this
is the implementation rationale.

## 1. Big picture

```
                 ┌──────────────────────────────────────────────────────┐
   corpus dir ──►│ ingest.py                                            │
   (.md/.txt/.pdf)│  chunking.py ──► Chunk(text, doc_id, offsets, page) │
                 │       │                                             │
                 │       ├──► embeddings.py ──► vectors ──┐            │
                 │       └──► store.py ──► FTS5 + chunks ─┤            │
                 └───────────────────────┬─────────────────┘            │
                                         ▼                              │
   query ────────────────────────► retrieval.py                         │
                                   │                                    │
              ┌────────────────────┼────────────────────┐               │
              ▼                    ▼                    ▼               │
        store.bm25()        vectorindex.search()   (ranks only)         │
        (FTS5 BM25)         (vec0 / numpy)                             │
              │                    │                                    │
              └────────► fusion.rrf_fuse() ◄────────┘                   │
                              │  fused candidates                      │
                              ▼                                        │
                       rerank.py  cross-encoder (optional)             │
                              │  top-k chunks + evidence_confidence    │
                              ▼                                        │
                    ┌─── abstention gate (min_evidence_confidence)     │
                    │ low confidence? ──► abstain (no LLM call)        │
                    ▼                                                  │
              generation.py                                            │
                build prompt (full chunk text, [S1]..[Sn] markers)     │
                call OpenAI-compatible endpoint (httpx)                │
                parse citations, detect abstention                     │
                    │                                                  │
                    ▼                                                  │
        Answer(status, text, citations[], evidence_confidence, ms)     │
                    │                                                  │
        ┌───────────┴────────────┐                                     │
        ▼                        ▼                                     │
     api.py (FastAPI)         cli.py (argparse)                        │
        └──────────────── evaluation.py ──► metrics.py ──► report ─────┘
                     (recall@k, nDCG, MRR, groundedness, abstention, latency)
```

Single process, single SQLite file, no message queue, no cache server. Every
arrow above is a synchronous call except the LLM HTTP request.

## 2. Module map

| Module | Responsibility | Notes |
|---|---|---|
| `config.py` | One flat `Settings` object, env-driven via pydantic-settings | `resolved()` absolutizes paths so nothing guesses the CWD |
| `types.py` | Dataclasses shared across layers: `Document`, `Chunk`, `Citation`, `RetrievedChunk`, `Answer` | `Citation` carries both a short `snippet` (for display) and the full `context` (what the LLM actually saw) |
| `chunking.py` | Word-count chunking with overlap, preferring paragraph/heading boundaries | "words" = whitespace tokens, a documented dependency-free proxy for subword tokens |
| `ingest.py` | Directory scan, sha256 dedup, PDF page tracking, drives chunking + embedding + indexing | Incremental by default |
| `embeddings.py` | `Embedder` protocol; fastembed implementation + labelled hash stub | Stub exists for offline smoke tests only |
| `vectorindex.py` | `VectorIndex` protocol; sqlite-vec (`vec0`) and numpy brute-force implementations | L2-normalized vectors, so L2 ordering ≡ cosine ordering |
| `store.py` | SQLite schema: `chunks`, FTS5 `chunks_fts`, `documents`; BM25 queries; evidence matching cache | The FTS5 index and the vec table live in the same file |
| `fusion.py` | Reciprocal Rank Fusion over ranked lists | `k=60`, from the original RRF paper |
| `rerank.py` | Cross-encoder scoring of (query, chunk) pairs | fastembed's `TextCrossEncoder`; identity when disabled |
| `retrieval.py` | Orchestrates bm25 + vector + fuse + rerank; produces `RetrievedChunk`s with per-stage ranks and `evidence_confidence` | Pure retrieval, no LLM |
| `generation.py` | Prompt construction, OpenAI-compatible client, citation parsing, abstention detection | The prompt uses full chunk `context`, not truncated snippets |
| `metrics.py` | recall@k, precision@k, nDCG@k, MRR, hit@k, and the abstention/groundedness primitives | No I/O, no models — pure functions |
| `evaluation.py` | Drives the pipeline over labelled questions, calls `metrics`, builds the report, formats the summary | Holds the `by_k` cut-off table and the threshold sweep |
| `pipeline.py` | `Pipeline`: wires store + index + embedder + reranker + generator together; used by both API and CLI | Built once, reused |
| `api.py` | FastAPI app and routes | Deliberately does **not** use `from __future__ import annotations` |

`cli.py` (repo root) is a thin argparse front-end over `Pipeline` and
`evaluation`.

## 3. Key design decisions and why

### 3.1 SQLite + FTS5 + sqlite-vec in one file
The whole index is one `data/index.sqlite3`. FTS5 gives BM25 with no external
search service; `sqlite-vec` adds a `vec0` virtual table for ANN-ish vector search.
Both live in the same file, so `rm data/index.sqlite3` is a complete reset, the
index is trivially inspectable with the `sqlite3` CLI, and there is no server to
operate. For a portfolio project aimed at demonstrating *retrieval quality* rather
than *scale*, this is the right trade: it removes an entire class of infrastructure
complexity so the effort goes into chunking, fusion, re-ranking, and evaluation.

The `VectorIndex` protocol exists precisely because this is a trade, not a
forever-answer. `numpy` brute-force cosine is a fully working fallback behind the
same interface, and both are exercised by the test suite and reported in
`/health` and eval reports. Swapping in pgvector/Qdrant/FAISS later means writing
one new class.

### 3.2 RRF instead of score interpolation
BM25 scores and cosine similarities live on incomparable scales, and the scale of
each depends on corpus statistics and model choice. RRF (`Σ 1/(k + rank)`) fuses
*ranks*, discarding the raw magnitudes, so it needs no per-corpus score
normalization and is robust when one retriever's scores are badly calibrated. The
cost is that it throws away confidence information — which is why
`evidence_confidence` is taken from the **cross-encoder** stage (a calibrated-ish
score over the final pair) rather than from the fused score.

### 3.3 Cross-encoder re-ranking as a separate stage
Bi-encoders (dense retrieval) are fast but score the query and document
independently. A cross-encoder sees `(query, chunk)` jointly and is markedly
better at the fine-grained "does this chunk actually answer this question"
judgement. It is applied to the fused top-N (not the whole corpus) because it is
~100x more expensive per pair. This is the classic retrieve-then-rerank shape.

### 3.4 Citations carry full context, not just a snippet
`Citation` has both `snippet` (a short display string for the CLI/API response)
and `context` (the **full chunk text**). The prompt is built from `context`, so
the model sees all the evidence it is being asked to cite — not a 320-character
window. This distinction was found by running the real eval: a model prompted
with snippets answered correctly only about what happened to be inside the snippet
window. Display and evidence are separate concerns and must not share a variable.

### 3.5 Two-layer abstention
There are two independent opportunities to refuse:

1. **Pre-generation gate (deterministic).** If the cross-encoder's top score is
   below `min_evidence_confidence`, we never call the LLM and abstain immediately.
   This is cheap, reproducible, LLM-independent, and measurable with the
   retrieval-only eval path.
2. **In-generation abstention (model-driven).** The system prompt instructs the
   model to emit a fixed `abstain_phrase` when the context does not answer the
   question. The client detects that phrase and reports
   `abstained_insufficient_evidence` rather than `answered`.

Layer 1 catches "nothing relevant was retrieved" (e.g. asking about annual leave
against a security corpus). Layer 2 catches "relevant material was retrieved but
doesn't actually answer the question." Neither alone is sufficient; measured
numbers for both are in `README.md` and `docs/evaluation.md`.

### 3.6 A pluggable LLM client, with a real offline mode
`generation.py` talks to any OpenAI-compatible `/v1/chat/completions` endpoint
through `httpx` — no vendor SDK. The default target is local Ollama; pointing at
vLLM, llama.cpp server, OpenAI, or a gateway is a config change, not a code change.
If no LLM is reachable, `/ask` returns a structured `no_llm_configured` result with
the retrieval evidence intact, so the retrieval half of the product remains useful
(and evaluable) with zero LLM configured.

### 3.7 The eval harness is a first-class citizen, not a script
Retrieval quality is only trustworthy if it is measured, so `metrics.py` is pure
and dependency-free, and `evaluation.py` reports per-stage *and* per-cut-off
numbers. Two details worth calling out:

- **Evidence labels are verbatim substrings**, not chunk ids. A label like
  `"400 days"` points at the fact, not at `chunk 3`; so if chunk size changes and
  every chunk id shifts, the labels still mean the same thing. Matching is case- and
  whitespace-insensitive, with a write-invalidated cache.
- **Quality metrics are refused when stubs are active.** If the hash embedder or a
  disabled reranker is in play, the report sets `quality_metrics_valid=false` with
  an explicit warning. A stub run can never be mistaken for a real measurement.

### 3.8 Determinism
`temperature=0`, fixed chunking, and stable ranking ties keep runs comparable.
Where a dependency cannot be made fully deterministic (ONNX float reduction order,
for instance), this is documented rather than papered over.

## 4. Data model

One SQLite file with:

- `documents(doc_id, path, sha256, size_bytes, ingested_at, meta_json)`
- `chunks(chunk_id, doc_id, ordinal, text, char_start, char_end, page)`
- `chunks_fts` — FTS5 external-content table over `chunks.text`
- `vec_chunks` — `vec0` virtual table `(chunk_id, embedding)` when sqlite-vec is active

`chunk_id` is a SQLite rowid, assigned on insert (an earlier explicit-id scheme
collided across re-ingests; see `docs/SETUP-NOTES.md`). `doc_id` is a stable
sha256-derived key for the *content*, used for incremental re-ingest.

## 5. Request lifecycle (ask)

1. `Pipeline.ask(question, top_k, use_rerank, min_confidence)`.
2. `retrieval.retrieve()` → BM25 top-50 + vector top-50 → RRF → cross-encoder
   rerank → top-`retrieve_top_k` `RetrievedChunk`s, each with provenance and
   per-stage ranks; overall `evidence_confidence` from the top cross-encoder score.
3. If `evidence_confidence < min_confidence` → return an abstention, no LLM call.
4. Else `generation.generate()` builds the prompt from the full chunk contexts and
   calls the endpoint.
5. Parse `[S1]`-style markers → `Citation`s (validating each marker resolves).
   Detect the abstain phrase.
6. Return `Answer` with status, text, citations, confidence, and latency. The API
   serializes it; the CLI renders it.

## 6. Failure behaviour

| Failure | Behaviour |
|---|---|
| No LLM configured / unreachable | `status=no_llm_configured`, retrieval evidence returned, HTTP 200 with explicit status (not a 500) |
| LLM returns empty `content` but has reasoning | `LLMError` with an actionable message (set `llm_reasoning_effort=none` / adjust token budget) — never faked as an abstention |
| LLM emits an unknown `[S9]` marker | Marker counted as invalid and reported in `n_invalid_citations`; other citations still resolve |
| sqlite-vec missing | `auto` falls back to numpy; `sqlite` fails loudly at startup |
| fastembed/model unavailable | Hash-stub embedder, flagged `stub=True` everywhere it surfaces |
| Re-ingest of unchanged files | sha256 match → skipped |

## 7. What would change at scale

Honest list, not a roadmap:

- **Vector index:** replace sqlite-vec with an ANN index (HNSW) or a dedicated
  store; the `VectorIndex` protocol contains the blast radius.
- **Chunking:** word-count chunking is a starting point. Real corpora want
  structure-aware or semantic chunking, and possibly late chunking / contextual
  retrieval to fix the "answer spans a boundary" failure mode.
- **Abstention:** re-sweep the threshold per corpus (the harness already does), or
  replace the fixed threshold with a calibrated/learned score.
- **Groundedness:** the answer-support metric is a lexical proxy by design;
  a NLI/faithfulness model would be the rigorous upgrade.
- **Concurrency:** the pipeline is built per-process; a service at real load would
  want a shared read-only index handle and batched embedding.
- **Streaming:** the API returns whole answers; SSE/streaming is a generation-layer
  change, not a retrieval change.
