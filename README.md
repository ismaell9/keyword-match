# rag-doc-qa

Document question-answering with citations: hybrid retrieval (BM25 + dense
vectors, RRF fusion, cross-encoder re-ranking), an LLM grounded **only** in the
retrieved chunks with inline citations, a two-layer abstention mechanism, and an
evaluation harness whose numbers are **measured, reproducible, and never
fabricated**.

Built to be honest: every number in [Results](#results) came out of `cli.py eval`
on this machine. The parts that are stubs say so loudly, and a stub run cannot be
mistaken for a real measurement.

```
$ .venv/bin/python cli.py ask "What retention period applies to security audit logs?"
...
Security audit logs have a retention period of 400 days [S1][S2]. Incident response
is triaged by severity levels defined as SEV-1 (confirmed breach or data
exfiltration), SEV-2 (credible attempt or exposure without confirmed loss), and
SEV-3 (policy violation or suspicious activity) [S1][S2].

citations
  [S1] Data Retention and Privacy Policy  [data-retention-policy.md chunk 0]
  [S2] IT Security Policy  [it-security-policy.md chunk 1]
```

## Highlights

- **Hybrid retrieval.** FTS5 BM25 + sqlite-vec dense search, fused with Reciprocal
  Rank Fusion, then re-ranked by a cross-encoder. Every returned chunk carries its
  provenance (document, chunk, page/offset) and per-stage ranks.
- **Grounded generation.** The model is shown only the retrieved chunks (full
  text, labelled `[S1]…[Sn]`) and must cite them. Citations that don't resolve to
  a provided chunk are counted and reported, not silently dropped.
- **Abstention that is actually measured.** A deterministic pre-generation gate on
  cross-encoder confidence, plus a model-level "I don't have enough information"
  path. Both are evaluated on labelled unanswerable questions.
- **Evaluation harness as a first-class citizen.** recall@k, precision@k, nDCG@k,
  MRR, hit@k per stage and per cut-off; citation validity, sentence citation
  coverage, an answer-support proxy; abstention accuracy + threshold sweep;
  latency percentiles. Reports record models, stub flags, and backend.
- **One SQLite file.** FTS5 + vec0 in `data/index.sqlite3`. `rm` it and you have a
  clean slate. No search server, no cache server.
- **Pluggable LLM.** Any OpenAI-compatible endpoint (`OPENAI_BASE_URL`). Defaults
  to local Ollama. With no LLM configured it still serves retrieval and returns an
  explicit `no_llm_configured` result instead of a fake answer.

## Architecture

```
   corpus (.md/.txt/.pdf)
        │
        ▼
   ingest ──► chunking ──► Chunk(text, doc_id, char offsets, page)
        │                      │
        │                      ├──► embeddings (fastembed, 384-dim) ──► vectors
        │                      └──► store (SQLite: chunks + FTS5)
        ▼
      query ──► retrieval
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
   BM25 (FTS5)        dense (sqlite-vec / numpy)
        └─────────┬──────────┘
                  ▼
          RRF fusion (k=60)
                  ▼
      cross-encoder re-rank  ──► top-k chunks + evidence_confidence
                  │
        confidence < threshold?  ──► abstain (no LLM call)
                  ▼
            generation
        prompt: question + [S1]..[Sn] (full chunk text)
        call OpenAI-compatible endpoint
        parse citations, detect abstention
                  ▼
   Answer(status, text, citations[], evidence_confidence, latency)
        │
   ┌────┴─────┐
   ▼          ▼
  API      CLI          └──► evaluation ──► metrics ──► JSON report
 (FastAPI) (argparse)
```

Full rationale in [`docs/architecture.md`](docs/architecture.md); requirements in
[`SPEC.md`](SPEC.md).

## Quickstart

Requires Python ≥3.11 (developed and measured on 3.14). An LLM endpoint is
optional — retrieval and the eval harness work without one.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps

# Build the index from the sample corpus (downloads embedding + rerank models once)
.venv/bin/python cli.py ingest

# Retrieval only, no LLM needed
.venv/bin/python cli.py retrieve "What is the API rate limit?"

# Ask a question (needs an LLM; default target is local Ollama on :11434)
.venv/bin/python cli.py ask "What retention period applies to security audit logs?"

# Run the evaluation harness
.venv/bin/python cli.py eval

# Serve the API
.venv/bin/python cli.py serve   # then GET /health, POST /ask
```

`make setup ingest test ask eval serve` wraps the same commands.

### Configuration

Everything is env-driven (see [`.env.example`](.env.example)). The ones that
matter most:

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | `not-needed` | Key (local Ollama ignores it) |
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Model name at that endpoint |
| `LLM_REASONING_EFFORT` | *(unset)* | Set `none` for thinking models (e.g. Qwen3.5 on Ollama) that otherwise return empty `content` |
| `RAGQA_VECTOR_BACKEND` | `auto` | `sqlite` (vec0), `numpy` (brute-force cosine), or `auto` |
| `RAGQA_MIN_EVIDENCE_CONFIDENCE` | `0.15` | Pre-generation abstention gate |
| `RAGQA_CHUNK_SIZE_WORDS` | `220` | Chunk size (whitespace "words") |

### Docker

```bash
docker compose up --build          # API + Ollama sidecar
docker compose run --rm cli ingest
docker compose run --rm cli ask "What is the availability SLO?"
docker compose run --rm cli eval
```

To use an existing LLM endpoint instead, drop the `ollama` service and point
`OPENAI_BASE_URL` at it.

## Results

Measured on this machine with the real stack — fastembed
`BAAI/bge-small-en-v1.5`, cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`,
sqlite-vec, and `qwen3.5:2b` (via Ollama, reasoning disabled) for generation.
Corpus: 6 documents / 14 chunks. Questions: 16 (12 answerable, 4 unanswerable).
Full reports in `data/eval/`. How to read these — and what they don't prove — in
[`docs/evaluation.md`](docs/evaluation.md).

### Retrieval by stage (k=8)

| Stage | recall@8 | precision@8 | nDCG@8 | MRR | hit@8 |
|---|---|---|---|---|---|
| bm25_only | 1.000 | 0.146 | 0.969 | 0.958 | 1.000 |
| vector_only | 1.000 | 0.146 | 0.879 | 0.854 | 1.000 |
| hybrid | 1.000 | 0.146 | 0.969 | 0.958 | 1.000 |
| hybrid+rerank | 1.000 | 0.146 | 0.969 | 0.958 | 1.000 |

### Retrieval by cut-off (recall@k / nDCG@k)

| k | bm25_only | vector_only | hybrid | hybrid+rerank |
|---|---|---|---|---|
| 1 | 0.833 / 0.917 | 0.667 / 0.750 | 0.833 / 0.917 | 0.833 / 0.917 |
| 3 | 1.000 / 0.969 | 0.875 / 0.823 | 1.000 / 0.969 | 1.000 / 0.969 |
| 5 | 1.000 / 0.969 | 1.000 / 0.879 | 1.000 / 0.969 | 1.000 / 0.969 |
| 8 | 1.000 / 0.969 | 1.000 / 0.879 | 1.000 / 0.969 | 1.000 / 0.969 |

At k=8 every stage saturates, which is a property of the tiny corpus (8 of 14
chunks), not evidence the stages are equal. The informative comparison is at small
k, where BM25 clearly beats dense-only (0.833 vs 0.667 recall@1) and hybrid fusion
recovers BM25's ranking quality. The hybrid gain is insurance against BM25's
out-of-vocabulary failure mode, which this particular corpus does not stress.

### Abstention (retrieval-only gate, no LLM)

At the configured threshold `0.15`: accuracy **0.9375**, unanswerable
correct-abstain **0.75**, answerable false-abstain **0.00**. Accuracy is flat at
0.9375 across thresholds `0.05 … 0.6` (best: 0.05) and degrades at 0.7.

### Generation (`qwen3.5:2b`)

| Metric | Value |
|---|---|
| answered / abstained | 13 / 3 |
| citation validity | 1.000 |
| sentence citation coverage | 0.859 |
| answer-support (proxy) | 0.730 |
| answerable false-abstain rate | 0.000 |
| unanswerable abstain rate | 0.750 |

The one unanswerable question missed by both the gate and the model is *"Which
cloud provider hosts the primary region?"* — the question shares vocabulary with
answerable vendor-tier content (evidence confidence 0.878) even though the corpus
never names a provider. The model cited what evidence existed and explicitly said
the name isn't present rather than inventing one. That failure mode — lexical
overlap with related material masquerading as evidence — is the honest weak spot
of a confidence gate.

### Latency

| Path | p50 | p95 |
|---|---|---|
| retrieval only | ~508 ms | ~593 ms |
| end-to-end (incl. LLM) | ~1445 ms | ~1657 ms |

Retrieval latency is dominated by the cross-encoder re-rank, not SQLite. End-to-end
is dominated by the local 2B model. One machine, warm cache, no concurrency —
indicative, not a benchmark.

## Limitations

Stated plainly, because overstating them would defeat the point of the project:

1. **Tiny corpus.** 6 documents / 14 chunks. Recall@8 is saturated for every
   stage, so the retrieval comparison carries real signal only at k=1 and k=3.
2. **Small, hand-made eval set.** 16 questions written alongside the corpus; no
   inter-annotator agreement, one person's relevance judgement.
3. **answer-support is a lexical proxy, not a faithfulness model.** It checks
   whether a cited sentence's content words appear in the cited chunk. It rewards
   extractive answers and cannot catch a lexically-present-but-logically-wrong
   sentence. The hook to swap in an NLI model exists
   (`evaluation._groundedness`).
4. **Abstention is corpus-calibrated.** The threshold was swept on this corpus's
   cross-encoder score distribution; a different corpus, reranker, or chunk size
   requires re-sweeping (the harness does this automatically).
5. **One generation model, one run, no human eval.** Generation numbers come from
   a single local 2B model at `temperature=0`. No variance across models or seeds.
6. **Latency is one warm machine.** No cold-start, no load, no concurrency.

## What is stubbed (and how you'd know)

| Component | Real path | Fallback | How it's labelled |
|---|---|---|---|
| Embeddings | fastembed `BAAI/bge-small-en-v1.5` | deterministic hash stub | `stub=True` in `/health` and every eval report; quality metrics are refused |
| Reranker | cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` | disabled (identity) | reported as `reranker: disabled`, never as a fake score |
| Vector index | sqlite-vec `vec0` | numpy brute-force cosine | `vector_backend` reported in `/health` and reports |
| LLM | any OpenAI-compatible endpoint | none | `generation: not configured`; `/ask` returns `no_llm_configured` with evidence intact |

No fallback pretends to be the real thing.

## Project layout

```
rag-doc-qa/
├── cli.py                    # ingest / ask / retrieve / eval / stats / config / serve
├── conftest.py
├── src/ragqa/
│   ├── config.py             # env-driven settings (pydantic)
│   ├── types.py              # Document, Chunk, Citation, RetrievedChunk, Answer
│   ├── chunking.py           # word-count chunking, boundary-aware
│   ├── ingest.py             # directory scan, sha256 dedup, PDF pages
│   ├── embeddings.py         # Embedder protocol: fastembed + labelled hash stub
│   ├── vectorindex.py        # VectorIndex protocol: sqlite-vec + numpy fallback
│   ├── store.py              # SQLite: chunks, FTS5, evidence matching
│   ├── fusion.py             # Reciprocal Rank Fusion
│   ├── rerank.py             # cross-encoder re-ranking
│   ├── retrieval.py          # orchestrates bm25 + vector + fuse + rerank
│   ├── generation.py         # prompt, OpenAI-compatible client, citations, abstention
│   ├── metrics.py            # recall@k, nDCG, MRR, hit@k (pure functions)
│   ├── evaluation.py         # runs the harness, builds the report
│   ├── pipeline.py           # wires everything together
│   └── api.py                # FastAPI app
├── data/
│   ├── corpus/               # sample documents (incl. a generated PDF)
│   └── eval/
│       ├── questions.jsonl   # 16 labelled questions
│       └── results-*.json    # measured reports
├── docs/
│   ├── architecture.md       # design rationale
│   ├── evaluation.md         # what's measured, how, and the honest caveats
│   ├── PORTFOLIO.md          # how to talk about this project
│   └── SETUP-NOTES.md        # what worked / what didn't, per environment
├── tests/                    # 149 tests: chunking, RRF, citations, abstention, ...
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .env.example
└── SPEC.md
```

## Testing

```bash
.venv/bin/python -m pytest -q     # 148 passed
```

The suite covers chunking, RRF fusion, citation parsing/formatting, abstention
logic, metrics, storage, retrieval, the LLM client, and the API — without network
access or model downloads.

## License

MIT
