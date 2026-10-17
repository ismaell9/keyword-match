# PORTFOLIO — how to talk about this project

This file exists so the project can be described accurately and confidently in
applications and interviews **without overclaiming**. Every claim below is backed
by something in this repo; where a claim is weak, it says so.

## The one-line version

> A document Q&A service with citations: hybrid retrieval (BM25 + dense vectors +
> RRF + cross-encoder re-ranking) feeding an LLM that must ground its answers in
> the retrieved chunks, with a measured abstention mechanism and an evaluation
> harness that reports real numbers.

## The 30-second version

> I built a RAG service end to end — ingestion, chunking, hybrid retrieval, cross-
> encoder re-ranking, LLM orchestration with citations, and an evaluation harness.
> The part I care most about is the evaluation: recall@k, nDCG, MRR per retrieval
> stage and per cut-off, citation validity, an answer-support proxy, and abstention
> accuracy on labelled unanswerable questions. The retrieval numbers, the
> abstention threshold sweep, and the generation metrics in the README were all
> produced by running the harness on my machine — they're not estimates. The corpus
> is small, so I'm careful about what those numbers do and don't show.

## What this demonstrates

Concrete, defensible skills, mapped to the code:

| Skill | Where it shows up |
|---|---|
| Hybrid retrieval (BM25 + dense) and why | `store.py` (FTS5), `vectorindex.py`, `fusion.py`, and the measured stage comparison |
| Re-ranking with a cross-encoder | `rerank.py`; measured latency cost vs. ranking quality |
| LLM orchestration, grounding, citations | `generation.py`; citation validity = 1.000 measured |
| Knowing when *not* to answer | two-layer abstention in `retrieval.py` + `generation.py`; threshold sweep in the eval report |
| Evaluation discipline | `metrics.py` (pure), `evaluation.py`; per-stage, per-cut-off, stub-integrity flags |
| Production-shaped Python service | FastAPI app, env-driven config, incremental ingestion, Docker, one-file storage, a real test suite |
| Honest engineering | the Limitations sections in `README.md` and `docs/evaluation.md` |

## Suggested talking points for interviews

**On hybrid retrieval and why not just vectors.**
> BM25 and dense retrieval fail in different ways — BM25 on vocabulary mismatch,
> dense on fine-grained keyword facts. I fused them with RRF rather than score
> interpolation because their raw scores live on incomparable scales and RRF only
> needs ranks. Measured on my corpus, BM25 actually wins at small k (recall@1
> 0.833 vs 0.667 for dense-only); the hybrid path is insurance against BM25's
> failure mode, which my small corpus doesn't stress. I'd rather say that plainly
> than pretend hybrid won a race it didn't.

**On abstention.**
> Two independent layers: a deterministic gate on the cross-encoder's top score
> before we spend an LLM call, and the model's own instructed refusal. I evaluate
> them separately — the gate is measurable without any LLM. At threshold 0.15 the
> gate gets accuracy 0.9375 with zero false-abstains on answerable questions, and
> it's flat across thresholds 0.05–0.6, so it's not knife-edge. The one question it
> misses is instructive: it shares vocabulary with related material, so lexical
> overlap masquerades as evidence. That's a known weakness of confidence gates.

**On citations and groundedness.**
> The prompt shows the model the full text of each retrieved chunk, labelled [S1]…,
> and I validate that every marker in the answer resolves to a chunk actually
> provided — measured citation validity is 1.000. My groundedness metric is a
> lexical support proxy, not a real faithfulness model, and I document that
> explicitly. Swapping in an NLI model is the obvious upgrade; I left the hook.

**On evaluation methodology.**
> Relevance labels are verbatim evidence substrings, not chunk ids, so they survive
> changes to chunk size. And if any stub component is active, the report refuses to
> compute quality metrics — a stub run can't be mistaken for a real measurement.

**On the honest limitations (expect to be asked).**
> The corpus is 14 chunks, so recall@8 is saturated for every stage; the retrieval
> comparison only carries signal at k=1 and k=3. The eval set is 16 hand-made
> questions with no second annotator. Generation is one local 2B model, one run, no
> human eval. I'd rather list those than let anyone infer more from the numbers
> than they support.

## What this project is *not*

Do not claim, and do not let the reader infer:

- Not large-scale. No load testing, no concurrency story, one warm machine.
- Not a vector database benchmark. sqlite-vec was chosen for inspectability, not
  speed, and there's a `VectorIndex` interface precisely because it's replaceable.
- Not a benchmark of any commercial model or service. Generation numbers are from
  one local 2B model.
- Not multi-tenant, authenticated, or production-secured.

## If asked "what would you do next?"

Have a real answer — this is usually where the interesting conversation is:

1. **A bigger, independently-labelled corpus** with a second annotator, so the
   retrieval numbers separate the stages instead of saturating.
2. **Structure-aware or semantic chunking** (and possibly late chunking), since
   word-count chunking is the weakest link for long policy documents.
3. **Replace the lexical groundedness proxy** with an NLI/faithfulness model and
   re-measure, so the generation numbers stop being partly proxy.
4. **Corpus-adaptive abstention** — a percentile- or learned threshold instead of
   a fixed one, re-fit automatically per corpus.
5. **Move the index behind a real ANN store** (the interface already contains this)
   once the corpus outgrows brute-force-in-SQLite.

## Résumé bullet (use as-is or adapt)

> **rag-doc-qa** — Document Q&A service with citations (Python, FastAPI, SQLite +
> FTS5 + sqlite-vec, fastembed, OpenAI-compatible LLM). Built hybrid retrieval
> (BM25 + dense, RRF fusion) with cross-encoder re-ranking, grounded LLM
> generation with validated inline citations, and a two-layer abstention mechanism.
> Authored an evaluation harness reporting recall@k / nDCG / MRR per stage and
> cut-off, citation validity, and abstention accuracy on labelled unanswerable
> questions; all reported numbers are measured, with stub components refusing to
> produce quality metrics. 148 passing tests; Dockerized; single-file index.

Keep the "all reported numbers are measured" phrase — it's the differentiator, and
it's true.
