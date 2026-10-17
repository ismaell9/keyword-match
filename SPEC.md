# SPEC — rag-doc-qa

Version: 0.1.0
Status: implemented, measured, honest about its limits. See `README.md` for measured
numbers and `docs/SETUP-NOTES.md` for what was actually verified in this environment.

## 1. What this is

A document question-answering service that:

1. Ingests a directory of documents (Markdown, plain text, PDF).
2. Retrieves the chunks most likely to contain the answer using **hybrid search**
   (BM25 keyword + dense vector) fused with **Reciprocal Rank Fusion (RRF)** and
   re-ranked by a **cross-encoder**.
3. Generates an answer with an LLM **grounded only in the retrieved chunks**, with
   inline citations that resolve to a concrete document, chunk, and page/offset.
4. **Abstains** — and says so — when the retrieved evidence is insufficient, instead
   of hallucinating.
5. Reports retrieval and generation quality via an **evaluation harness** whose
   numbers are measured, reproducible, and never fabricated.

The point is not "yet another LangChain demo." The point is that every stage is
observable, every number in the README came out of `cli.py eval` on this machine,
and the parts that are stubs say so loudly.

## 2. Non-goals

- Not a hosted product. No auth, no multi-tenancy, no rate limiting beyond a max
  upload size.
- Not a vector database. SQLite (FTS5 + sqlite-vec) is deliberate: one file, no
  server, trivially inspectable. The interfaces (`VectorIndex`) exist so a real
  ANN index (pgvector, Qdrant, FAISS) can replace it without touching callers.
- Not a streaming UI. The API returns structured JSON; the CLI renders text.
- Not benchmarked at scale. The sample corpus is 6 documents / 14 chunks. Numbers
  are real but measured on a tiny corpus — see the limitations in `README.md`.

## 3. Functional requirements

### 3.1 Ingestion
- **IN-1** Scan a directory for `.md`, `.markdown`, `.txt`, `.pdf`.
- **IN-2** Chunk documents by word count with overlap, preferring paragraph and
  heading boundaries. Each chunk records: `doc_id`, `chunk_id` (ordinal within the
  doc), `text`, `char_start`/`char_end` offsets, and `page` for PDFs.
- **IN-3** Compute a sha256 per source file. Re-ingesting an unchanged file is a
  no-op (`RAGQA_INCREMENTAL_INGEST=true`); a changed file is re-chunked and its
  old chunks removed.
- **IN-4** Embed every chunk via the configured backend and store vectors in the
  active `VectorIndex`.
- **IN-5** Index chunk text in SQLite FTS5 for BM25.
- **IN-6** Report counts (documents, chunks, vectors) and be idempotent.

### 3.2 Retrieval
- **RT-1** Run BM25 (FTS5) and dense-vector search independently, each returning
  its top-N candidates with scores/ranks.
- **RT-2** Fuse the two ranked lists with RRF (`score = Σ 1/(k + rank)` over lists,
  `k=60` default).
- **RT-3** Re-rank the fused candidates with a cross-encoder over
  `(query, chunk_text)`, and return the top-`retrieve_top_k` chunks.
- **RT-4** Every returned chunk carries its provenance: `doc_id`, `chunk_id`,
  `page` (if any), `char_start`/`char_end`, and the score plus per-stage ranks, so
  a caller can see *why* a chunk was picked.
- **RT-5** Each retrieval produces an `evidence_confidence` in `[0,1]`, derived
  from the cross-encoder score of the top chunk (calibrated to the reranker in use).
  This is the input to the abstention gate.
- **RT-6** The vector backend is swappable behind an interface: `sqlite` (vec0),
  `numpy` (brute-force cosine), or `auto`. Switching backends must not change
  retrieval semantics beyond float precision.

### 3.3 Generation
- **GN-1** Build a prompt containing: the question, each retrieved chunk labelled
  `[S1]…[Sn]` with its **full text** (not a truncated snippet), and an instruction
  to answer **only** from that context, citing `[S1]`-style markers, and to state
  when the context does not contain the answer.
- **GN-2** Call any OpenAI-compatible `/v1/chat/completions` endpoint
  (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LLM_MODEL`). Default target is local Ollama.
- **GN-3** If no LLM is reachable/configured, return a structured
  `status="no_llm_configured"` result with the retrieval evidence intact — never a
  fake answer, never a crash.
- **GN-4** If `evidence_confidence < min_evidence_confidence`, **skip the LLM** and
  abstain deterministically. This is the retrieval-only abstention gate.
- **GN-5** Parse `[S1]`-style citations out of the answer. Every marker must map to
  a chunk that was actually provided; invalid markers are reported, not silently
  dropped.
- **GN-6** Abstention detection: the answer matches the configured `abstain_phrase`
  (exact or as a trailing sentence). Abstentions are reported as
  `status="abstained_insufficient_evidence"`, distinct from `answered`.
- **GN-7** Thinking models may return empty `content` with a populated reasoning
  channel. `llm_reasoning_effort` (e.g. `"none"`) and `llm_extra_body` are exposed
  to handle this; an empty answer with reasoning present raises an actionable error
  rather than faking an abstention.

### 3.4 API
- **API-1** `POST /ingest` — ingest a directory (path optional, defaults to corpus).
- **API-2** `POST /ask` — `{question, top_k?, use_rerank?, min_confidence?}` →
  `{status, answer, citations[], evidence_confidence, latency_ms, ...}`.
- **API-3** `POST /retrieve` — retrieval only, no LLM. Same evidence payload.
- **API-4** `GET /health` — liveness + resolved config + index stats.
- **API-5** `POST /upload` — add a single document (respecting `max_upload_bytes`).

### 3.5 CLI
Subcommands: `ingest`, `ask`, `retrieve`, `eval`, `stats`, `config`, `serve`.
`eval` writes a JSON report and prints a human-readable summary.

### 3.6 Evaluation harness
- **EV-1** Load labelled questions from JSONL. Each answerable question lists
  `relevant` evidence as **verbatim substrings** (not hard-coded chunk ids) so the
  labels survive chunk-size changes; unanswerable questions are labelled as such.
- **EV-2** Retrieval metrics per stage (`bm25_only`, `vector_only`, `hybrid`,
  `hybrid+rerank`) and per cut-off `k ∈ {1,3,5,8}`: recall@k, precision@k, nDCG@k,
  MRR, hit@k.
- **EV-3** Abstention metrics via the retrieval-only gate, including a threshold
  sweep and the best threshold by accuracy.
- **EV-4** Generation metrics (only when an LLM is configured): citation validity
  (fraction of `[S1]` markers that resolve to a provided chunk), sentence citation
  coverage (fraction of answer sentences carrying a citation), an answer-support
  proxy (lexical support of each cited sentence by its cited chunk), answerable
  false-abstain rate, unanswerable abstain rate, and a status histogram.
- **EV-5** Latency: retrieval and end-to-end p50/p95.
- **EV-6** The report records the environment (models, stub flags, backend). If any
  quality metric is computed on stub components, it is flagged
  `quality_metrics_valid=false` with a warning — **a stub run must never look like
  a real measurement**.
- **EV-7** Numbers are never invented. If a metric cannot be computed (no LLM, no
  labels), it is omitted or reported as null with a stated reason, not filled in.

## 4. Non-functional requirements

- **NF-1** Deterministic by default: `temperature=0`, fixed chunking, fixed seeds
  where the stack allows.
- **NF-2** Single-file storage; `rm data/index.sqlite3` fully resets the index.
- **NF-3** Runs offline after model download. No network calls except the LLM
  endpoint (which may itself be localhost).
- **NF-4** Tests cover chunking, RRF fusion, citation parsing/formatting, and
  abstention logic without network access or model downloads.
- **NF-5** All knobs are environment-driven (`.env`), validated by Pydantic.
- **NF-6** The API never uses `from __future__ import annotations` (it breaks
  FastAPI's local-type dependency resolution — documented in SETUP-NOTES).

## 5. Explicitly stubbed / degraded paths

| Component | Real path | Stub path | How the stub is labelled |
|---|---|---|---|
| Embeddings | fastembed `BAAI/bge-small-en-v1.5` | deterministic hash embedder | `stub=True` in env/report, `/health` warns, eval sets `quality_metrics_valid=false` |
| Reranker | cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` | identity (no re-rank) | reported as `reranker: disabled`, not as a fake score |
| Vector index | sqlite-vec `vec0` | numpy brute-force cosine | `vector_backend` reported everywhere |
| LLM | any OpenAI-compatible endpoint | none | `generation: not configured`, answers return `no_llm_configured` |

Nothing in the stub path pretends to be the real path.

## 6. Open decisions

1. **Chunk granularity vs. citation precision.** 220-word chunks with 40-word
   overlap work here; longer policy documents may need sliding windows or
   late chunking. Re-evaluate with a bigger corpus before tuning further.
2. **Abstention calibration.** The evidence-confidence gate is calibrated on this
   corpus's cross-encoder score distribution. On a different corpus the threshold
   must be re-swept (the harness already does this); a learned/percentile-based
   threshold is the natural next step.
3. **Answer-support proxy vs. a real faithfulness model.** The current groundedness
   metric is a lexical proxy, deliberately cheap and dependency-free. Swapping in a
   NLI/faithfulness model would be more rigorous but adds a heavy dependency and a
   second model download; the hook exists (`evaluation._groundedness`).
