# SETUP-NOTES

Development log: what worked, what didn't, and the bugs found along the way.
Written while building this on the machine described below, so a future reader can
tell which parts are verified facts and which are expectations.

## Environment this was actually built and measured on

- Python **3.14.4**, Linux x86_64
- SQLite **3.53.0** (bundled with CPython), with **FTS5** and loadable extensions
  confirmed working
- No Docker daemon, no Postgres, no external API keys available at build time
- Local LLMs via **Ollama** (OpenAI-compatible `/v1/chat/completions`)

## Python 3.14: what installed cleanly

Verified by installing and importing, not by reading changelogs:

| Package | Version | Notes |
|---|---|---|
| fastapi | 0.143.3 | works |
| starlette | 1.8.0 | works |
| uvicorn | 0.54.0 | works |
| pydantic / pydantic-settings | 2.14.1 / 2.16.0 | works |
| numpy | 2.5.4 | works |
| **fastembed** | **0.9.0** | **installs and runs on 3.14** |
| onnxruntime | 1.31.0 | first release with a **cp314 wheel** — this is what unblocks fastembed on 3.14 |
| tokenizers | 0.23.3 | abi3 wheel, works on 3.14 |
| sqlite-vec | 0.1.9 | `vec0` extension **loads** on this SQLite build |
| pypdf | 6.21.0 | works, page numbers extracted correctly |
| httpx | 0.28.1 | works |
| pytest | 9.1.1 | works |

Real models verified end-to-end on this setup:

- Embeddings: `BAAI/bge-small-en-v1.5` (384-dim, fastembed)
- Cross-encoder reranker: `Xenova/ms-marco-MiniLM-L-6-v2`

Because fastembed works here, the **real** embedding and reranking path is used —
the hash-stub embedder is a labelled smoke-test fallback, not what produced the
numbers in the README.

### sqlite-vec specifics confirmed

- `vec0` virtual table creation and load: works
- Both `k = ?` and `LIMIT n` neighbour syntaxes work
- Returns **L2 distance**. Since we L2-normalize all vectors, L2 ordering is
  identical to cosine ordering, and the numpy fallback returns `1 - cosine` so the
  two backends agree on ranking (up to float precision).

## Ollama notes

- Available models on this machine: `qwen3.5:2b`, `qwen3.5:4b-q4_K_M`,
  `qwen2.5-coder:0.5b`, `qwen2.5-coder:1.5b`.
- The documented default `qwen2.5:7b-instruct` is **not pulled** here. The project
  default in `config.py` is what a normal setup would use; on this machine the
  measured runs used `LLM_MODEL=qwen3.5:2b`.
- **Qwen3.5 is a thinking model.** Over `/v1/chat/completions` it can spend the
  entire token budget on its reasoning channel and return **empty `content`**.
  Setting `LLM_REASONING_EFFORT=none` (passed through as `reasoning_effort`)
  disables thinking and produces normal answers. Without it, an answer can come
  back empty while `reasoning` is populated — see bug #6 below.
- A cold first request to a not-yet-loaded model can take many seconds (model load
  from disk). Subsequent requests in the same eval run are much faster; this shows
  up in the latency percentiles (first question slow, rest warm).

## Bugs found and fixed (the useful part)

These were found by running the thing, not by reading it. Each one has a test or
a code comment tying it to the fix.

1. **Explicit `chunk_id = -1` collided on re-ingest.** Inserting chunks with a
   hand-computed `chunk_id` (starting at -1) collided across re-ingests because
   SQLite rowids reused values. Fix: omit the column and let SQLite assign the
   rowid. Re-ingestion is now idempotent.

2. **`from __future__ import annotations` in `api.py` broke every route.** The
   future import turns annotations into strings, and FastAPI resolves
   dependencies from *local* names at decoration time — the `PipelineDep` alias
   resolved to nothing, so every route returned **422**. Fix: removed the future
   import from `api.py`. **Do not re-add it** (noted in `architecture.md` too).

3. **Evidence labels spanning line breaks failed to match.** Evidence substrings
   from source documents sometimes crossed a line break, so a verbatim label
   didn't literally appear in the chunk text. Fix: normalize whitespace both when
   building labels and when matching (`normalize_ws`), with a cached normalized
   chunk view invalidated on index writes.

4. **`config.py` missing `Any` import.** `llm_extra_body: dict[str, Any]` with
   `from __future__ import annotations` deferred evaluation, so the missing import
   only surfaced as a `PydanticUserError` at settings instantiation, not at import
   time. Fix: added the import.

5. **Abstention check was too naive.** A model that ends its answer with the
   refusal sentence (rather than emitting only the refusal) wasn't detected.
   Fix: length guard + `endswith(target)` matching in addition to exact/substring.

6. **Empty `content` with populated `reasoning` faked an abstention.** A thinking
   model (Qwen3.5) returned empty `content` with a full reasoning channel; the
   client read that as "the model abstained", which is wrong — it never got to
   answer. Fix: `generation.py` now raises an `LLMError` with an actionable
   message (set `llm_reasoning_effort=none` / raise the token budget) instead of
   reporting a fabricated abstention.

7. **The prompt showed the model truncated snippets, not full chunks.** The most
   consequential one. The prompt was built from 320-character *display* snippets
   (meant for the CLI citation listing), so the model only ever saw a sliver of
   each chunk and could only cite what happened to fall inside that window.
   Found by running a real `cli.py ask` and seeing the answer ignore evidence that
   was in the chunk but outside the snippet. Fix: `Citation` now carries both a
   short `snippet` (display) and the full `context` (evidence); the prompt and the
   groundedness metric both use `context`. **Lesson: never let a display field
   double as the evidence field.**

## What was deliberately *not* done

- No Docker verification. No Docker daemon was available in this environment, so
  `Dockerfile` and `docker-compose.yml` are written to standard patterns
  (python-slim base, volumes for the index and model caches, Ollama sidecar) but
  have **not been built or run here**. Treat them as untested scaffolding.
- No Postgres/pgvector. Not available, and out of scope for the single-file design;
  the `VectorIndex` interface is where that would go.
- No hosted-LLM measurement. All generation numbers are from a local Ollama model.
- No load/concurrency testing.

## Reproducing the measured numbers

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps

# 1. build the index (downloads embedding + rerank models on first run)
.venv/bin/python cli.py ingest

# 2. retrieval-only eval (no LLM needed) -> data/eval/results-retrieval.json
.venv/bin/python cli.py eval

# 3. full eval incl. generation (needs an LLM) -> data/eval/results-full.json
LLM_MODEL=qwen3.5:2b LLM_REASONING_EFFORT=none LLM_TIMEOUT_SECONDS=180 \
  .venv/bin/python cli.py eval --out data/eval/results-full.json
```

`results-*.json` in `data/eval/` are the actual reports from the runs whose
numbers appear in the README. If you re-run, expect the retrieval numbers to match
closely (deterministic ranking, `temperature=0`); latency and generation metrics
will vary with hardware and model.

## Gotchas worth remembering

- `rm data/index.sqlite3` fully resets the index; re-run `cli.py ingest`.
- If you change `RAGQA_CHUNK_SIZE_WORDS`, re-run the eval — the verbatim evidence
  labels survive the change (that's the point of labelling by substring), but the
  *numbers* will move.
- If you change corpus, embedding model, or reranker, re-sweep
  `RAGQA_MIN_EVIDENCE_CONFIDENCE`; the threshold is calibrated to the score
  distribution of the current setup, and the eval report contains the sweep.
- The hash-stub embedder produces meaningless retrieval. If `/health` or an eval
  report says `stub=True`, do not trust the quality numbers — the harness refuses
  to compute them for exactly this reason.
