# Evaluation

This is the part of the project that is meant to be the differentiator: **every
number is measured by `cli.py eval` on this machine, and nothing is estimated,
rounded up, or invented.** This document explains what is measured, how, what the
current numbers are, and — importantly — what the numbers do *not* prove.

Reproduce with:

```bash
.venv/bin/python cli.py eval --out data/eval/results-full.json
```

Reports are written to `data/eval/`:

- `results-retrieval.json` — retrieval + abstention only (no LLM required)
- `results-full.json` — the same, plus generation metrics (LLM required)

## 1. The dataset

`data/eval/questions.jsonl` — 16 hand-labelled questions over the sample corpus
(6 documents, 14 chunks):

- **12 answerable.** Each carries `relevant`: a list of **verbatim evidence
  substrings** that must appear in a retrieved chunk for it to count as relevant.
- **4 unanswerable** (`answerable: false`) — questions the corpus genuinely does
  not answer (annual leave, sabbatical policy, the analytics vendor's price, which
  cloud provider hosts the primary region).

### Why verbatim substrings instead of chunk ids

The label `"400 days"` points at the *fact*, not at `chunk 3`. If you change
`chunk_size_words` and every chunk id shifts, the labels still mean exactly the
same thing, so results stay comparable across chunking experiments. Chunk-id
labels would silently break. Matching is case- and whitespace-insensitive, with a
cache invalidated on index writes.

Consequence to be aware of: a chunk counts as relevant if it *contains* the
evidence substring, so a larger chunk size makes labels easier to satisfy (a big
chunk swallows more evidence strings). This is a real property of the measure, not
a bug, and is one reason the corpus is kept small and the chunk size fixed for the
reported numbers.

## 2. What is measured

### 2.1 Retrieval (no LLM)

Four stages are evaluated independently on the same questions:

| Stage | What it is |
|---|---|
| `bm25_only` | FTS5 BM25 keyword search alone |
| `vector_only` | Dense vector search alone |
| `hybrid` | RRF fusion of the two |
| `hybrid+rerank` | Fusion, then cross-encoder re-ranking (the production path) |

For each, at cut-offs `k ∈ {1, 3, 5, 8}`: **recall@k**, **precision@k**,
**nDCG@k**, **MRR**, **hit@k**. Gain is binary (relevant / not relevant).

### 2.2 Abstention (retrieval-only gate)

The pre-generation gate abstains when the top cross-encoder score falls below a
threshold. Measured without any LLM:

- **accuracy** over all questions
- **unanswerable correct-abstain rate** (of the 4 unanswerable, how many correctly abstained)
- **answerable false-abstain rate** (of the 12 answerable, how many wrongly abstained)

Plus a **threshold sweep** and the best threshold by accuracy.

### 2.3 Generation (LLM required)

- **citation validity** — fraction of `[S1]`-style markers in the answer that
  resolve to a chunk actually provided to the model. Catches invented citations.
- **sentence citation coverage** — fraction of answer sentences carrying at least
  one citation.
- **answer-support (proxy)** — for each cited sentence, a lexical support score
  against the cited chunk's full text; averaged. A cheap, dependency-free proxy
  for groundedness, **not** a true faithfulness model (see limitations).
- **answerable false-abstain rate** and **unanswerable abstain rate** — the
  end-to-end abstention behaviour, on top of the gate.
- **status histogram** — `answered` vs `abstained_insufficient_evidence`.

### 2.4 Latency

Retrieval-only and end-to-end **p50 / p95**, measured over the whole eval run.

## 3. Stub integrity

If the hash embedder or a disabled reranker is active, the report sets
`quality_metrics_valid: false` with a warning naming the stub. A stub run cannot
be mistaken for a real measurement. The `environment` block records the exact
models, stub flags, and vector backend for every report.

## 4. Measured results

Environment: Python 3.14.4, Linux x86_64. Embeddings `BAAI/bge-small-en-v1.5`
(384-dim, fastembed), reranker `Xenova/ms-marco-MiniLM-L-6-v2` (cross-encoder),
vector backend sqlite-vec, generation model `qwen3.5:2b` via local Ollama with
`reasoning_effort=none`. Corpus: 6 documents / 14 chunks. All real; none stubbed.

### 4.1 Retrieval by stage (k=8)

| Stage | recall@8 | precision@8 | nDCG@8 | MRR | hit@8 |
|---|---|---|---|---|---|
| bm25_only | 1.000 | 0.146 | 0.969 | 0.958 | 1.000 |
| vector_only | 1.000 | 0.146 | 0.879 | 0.854 | 1.000 |
| hybrid | 1.000 | 0.146 | 0.969 | 0.958 | 1.000 |
| hybrid+rerank | 1.000 | 0.146 | 0.969 | 0.958 | 1.000 |

### 4.2 Retrieval by cut-off (recall@k / nDCG@k)

| k | bm25_only | vector_only | hybrid | hybrid+rerank |
|---|---|---|---|---|
| 1 | 0.833 / 0.917 | 0.667 / 0.750 | 0.833 / 0.917 | 0.833 / 0.917 |
| 3 | 1.000 / 0.969 | 0.875 / 0.823 | 1.000 / 0.969 | 1.000 / 0.969 |
| 5 | 1.000 / 0.969 | 1.000 / 0.879 | 1.000 / 0.969 | 1.000 / 0.969 |
| 8 | 1.000 / 0.969 | 1.000 / 0.879 | 1.000 / 0.969 | 1.000 / 0.969 |

**Reading these honestly.** At k=8 every stage saturates at recall 1.000, which
is a property of the tiny corpus (8 of 14 chunks is more than half of it), *not*
evidence that the stages are equivalent. The informative comparison is at small
k, where BM25 clearly beats dense-only (0.833 vs 0.667 recall@1; 1.000 vs 0.875
recall@3), and where hybrid fusion recovers BM25's ranking quality while adding
the vocabulary-matching robustness that BM25 alone lacks on out-of-vocabulary
queries. On this corpus the hybrid path ties BM25 rather than beating it — the
gain from hybrid is insurance against BM25's failure mode, which this particular
corpus does not stress. That is the honest read; a larger, more varied corpus is
what would show the separation.

### 4.3 Abstention (retrieval-only gate)

At the configured threshold `0.15`: **accuracy 0.9375**, unanswerable correct-abstain
**0.75**, answerable false-abstain **0.0**.

Threshold sweep: accuracy is flat at 0.9375 across `0.05 … 0.6`, and degrades at
`0.7` (accuracy 0.875, answerable false-abstain rises to 0.0833). Best threshold
by accuracy: **0.05**. The gate is not knife-edge sensitive — a wide band of
thresholds gives the same result on this corpus.

The one unanswerable question the gate does *not* catch is
*"Which cloud provider hosts the primary region?"* — its evidence confidence is
0.878, because the question shares vocabulary with answerable vendor-tier content
even though the corpus never names a provider. This is a genuine limitation of a
confidence gate: **lexical/topic overlap with related material can masquerade as
evidence.**

### 4.4 Generation (`qwen3.5:2b`, reasoning disabled)

| Metric | Value |
|---|---|
| answered / abstained | 13 / 3 |
| citation validity | 1.000 |
| sentence citation coverage | 0.859 |
| answer-support (proxy) | 0.730 |
| answerable false-abstain rate | 0.000 |
| unanswerable abstain rate | 0.750 |

The model abstained on 3 of the 4 unanswerable questions and never falsely
abstained on an answerable one. The one it did answer (the cloud-provider question
above) is the same question the gate also misses — and the model handled it about
as well as a grounded system can: it cited what evidence existed and explicitly
stated the provider's name is not in the documents, rather than inventing one.

### 4.5 Latency

| Path | p50 | p95 |
|---|---|---|
| retrieval only | ~508 ms | ~593 ms |
| end-to-end (incl. LLM) | ~1445 ms | ~1657 ms |

Retrieval latency is dominated by the cross-encoder re-rank over the fused
candidates, not by SQLite. End-to-end is dominated by the local 2B model's
time-to-first-token. Both were measured on this machine, on this corpus, with a
warm model — treat them as indicative for this setup, not as a benchmark.

## 5. Limitations of these numbers

Stated plainly, because a metric you cannot trust is worse than no metric:

1. **The corpus is tiny.** 14 chunks. Recall@8 saturates at 1.000 for every
   stage, so at k=8 the retrieval comparison carries no signal. The useful signal
   is at k=1 and k=3 only.
2. **The eval set is small and hand-made.** 16 questions, written by the same
   person who wrote the corpus. There is no inter-annotator agreement, and the
   labels encode one person's judgement of relevance.
3. **answer-support is a lexical proxy, not a faithfulness model.** It measures
   whether a cited sentence's content words appear in the cited chunk. It rewards
   extractive answers and cannot catch a sentence that is lexically present but
   logically wrong. A NLI-based metric would be the upgrade.
4. **Abstention calibration is corpus-specific.** The threshold was swept on this
   corpus's score distribution. A different corpus, reranker, or chunk size moves
   that distribution, and the threshold must be re-swept (the harness does this;
   the default value should not be trusted blindly elsewhere).
5. **One generation model, one run.** The generation numbers are from a single
   local 2B model, `temperature=0`. No variance across models or seeds is
   reported, and no human evaluation of answer quality was done.
6. **Latency is one machine, warm cache.** Not a benchmark; no concurrency, no
   cold-start, no load.

## 6. How to extend this

- Add questions to `data/eval/questions.jsonl`; use `relevant` substrings that pin
  the fact you care about.
- Re-run after any change to chunk size, embedding model, reranker, or fusion
  constant — the harness is the only thing that will tell you whether the change
  helped.
- If you change the corpus meaningfully, re-sweep `min_evidence_confidence`
  (`abstention.sweep` in the report gives you the curve for free).
- To make the groundedness metric rigorous, replace the lexical proxy in
  `evaluation._groundedness` with an NLI/faithfulness model and update this
  document with the new measured numbers.
