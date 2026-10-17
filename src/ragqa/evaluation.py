"""Offline evaluation harness.

This is the part of the project that separates it from a RAG tutorial: it does not
claim the system works, it *measures* whether it works and reports where it fails.

What it measures
----------------
Retrieval (per question, against evidence-derived relevance):
  recall@k, precision@k, nDCG@k, MRR -- reported for four configurations so the
  contribution of each stage is visible:
      bm25_only      lexical leg alone
      vector_only    dense leg alone
      hybrid         RRF fusion of the two (pre-rerank)
      hybrid+rerank  fusion followed by the cross-encoder

Abstention:
  * retrieval-only abstention, from the evidence-confidence gate. This is the
    meaningful number when no LLM is attached, and it is computed for the
    configured threshold *and* swept over a grid so a reader can see the
    precision/recall trade-off instead of trusting one magic number.
  * end-to-end abstention, which additionally counts the model refusing. Only
    reported as meaningful when generation is actually configured.

Answer grounding (only when generation runs):
  * citation validity  -- fraction of citation markers that resolve to a chunk we
    actually supplied (out-of-range markers are hallucinations and are counted).
  * citation coverage  -- fraction of answer sentences carrying a citation.
  * answer-support     -- fraction of the answer's content terms present in the
    supplied context. A *proxy* for groundedness, not entailment.

Latency: p50/p95 for retrieval and for the full query.

Honesty rules baked into this module:
  * Numbers are only emitted for slices that actually ran; nothing is defaulted
    to a flattering value.
  * If the embedder is the hash STUB, the report sets ``quality_metrics_valid``
    to False with a loud warning, because hash embeddings are not semantic and
    any retrieval score from them is not evidence about the real system.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ingest import IngestError, load_jsonl
from .metrics import (
    hit_at_k,
    mrr,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    split_sentences,
    term_support,
    mean,
)
from .pipeline import Pipeline
from .retrieval import content_terms
from .types import Answer

log = logging.getLogger(__name__)

STAGES = ("bm25_only", "vector_only", "hybrid", "hybrid+rerank")


@dataclass(slots=True)
class EvalQuestion:
    """One labelled question.

    Schema (JSONL, one object per line):
      id              str, unique
      question        str
      answerable      bool -- false means the corpus deliberately does NOT contain
                      the answer; the correct system behaviour is abstention.
      evidence        list[str] -- verbatim substrings that appear in a chunk that
                      *does* answer the question. Used to derive the relevant chunk
                      set at eval time. Required when answerable is true.
      relevant_doc_ids list[str] -- optional; doc-level recall. Derived from
                      evidence when omitted.
      gold_answer     str -- optional reference answer for eyeballing.
      notes           str -- optional.
    """

    id: str
    question: str
    answerable: bool
    evidence: list[str] = field(default_factory=list)
    relevant_doc_ids: list[str] = field(default_factory=list)
    gold_answer: str | None = None
    notes: str | None = None

    def validate(self) -> None:
        if not self.id:
            raise IngestError("eval question is missing 'id'")
        if not self.question.strip():
            raise IngestError(f"eval question {self.id!r} has an empty question")
        if self.answerable and not self.evidence:
            raise IngestError(
                f"eval question {self.id!r} is marked answerable but has no 'evidence'; "
                "without evidence strings the harness cannot derive relevant chunks"
            )


def load_eval_set(path: str | Path) -> list[EvalQuestion]:
    rows = load_jsonl(Path(path))
    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for row in rows:
        q = EvalQuestion(
            id=str(row.get("id", "")).strip(),
            question=str(row.get("question", "")).strip(),
            answerable=bool(row.get("answerable", True)),
            evidence=[str(e) for e in (row.get("evidence") or [])],
            relevant_doc_ids=[str(d) for d in (row.get("relevant_doc_ids") or [])],
            gold_answer=row.get("gold_answer"),
            notes=row.get("notes"),
        )
        q.validate()
        if q.id in seen:
            raise IngestError(f"duplicate eval question id {q.id!r}")
        seen.add(q.id)
        questions.append(q)
    if not questions:
        raise IngestError(f"no evaluation questions found in {path}")
    return questions


def resolve_relevant(pipeline: Pipeline, q: EvalQuestion) -> tuple[set[int], set[str]]:
    """Map a question's evidence strings to the chunk/doc ids that satisfy it."""
    chunk_ids: set[int] = set()
    doc_ids: set[str] = set(q.relevant_doc_ids)
    for ev in q.evidence:
        chunk_ids.update(pipeline.store.chunks_containing(ev))
        if not q.relevant_doc_ids:
            doc_ids.update(pipeline.store.doc_ids_containing(ev))
    return chunk_ids, doc_ids


def _stage_metrics(retrieved: Sequence[int], relevant: set[int], k: int) -> dict[str, float]:
    if not relevant:
        return {}
    return {
        f"recall@{k}": round(recall_at_k(retrieved, relevant, k), 4),
        f"precision@{k}": round(precision_at_k(retrieved, relevant, k), 4),
        f"ndcg@{k}": round(ndcg_at_k(retrieved, relevant, k), 4),
        "mrr": round(mrr(retrieved, relevant), 4),
        f"hit@{k}": round(hit_at_k(retrieved, relevant, k), 4),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: round(mean([r[key] for r in rows if key in r]), 4) for key in keys}


def _metrics_by_k(
    stage_ids_by_q: list[dict[str, list[int]]],
    relevant_by_q: list[set[int]],
    *,
    ks: Iterable[int],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate recall/nDCG/hit per stage per cut-off, from the stored id lists."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for kk in sorted(set(ks)):
        if kk < 1:
            continue
        per_stage: dict[str, dict[str, float]] = {}
        for stage in STAGES:
            recalls: list[float] = []
            ndcgs: list[float] = []
            hits: list[float] = []
            for ids, relevant in zip(stage_ids_by_q, relevant_by_q, strict=True):
                retrieved = ids.get(stage, [])
                recalls.append(recall_at_k(retrieved, relevant, kk))
                ndcgs.append(ndcg_at_k(retrieved, relevant, kk))
                hits.append(hit_at_k(retrieved, relevant, kk))
            if recalls:
                per_stage[stage] = {
                    f"recall@{kk}": round(mean(recalls), 4),
                    f"ndcg@{kk}": round(mean(ndcgs), 4),
                    f"hit@{kk}": round(mean(hits), 4),
                }
        if per_stage:
            out[str(kk)] = per_stage
    return out


def _groundedness(answer: Answer) -> dict[str, Any]:
    """Citation + support diagnostics for a single generated answer."""
    from .generation import extract_citation_indices  # local import avoids a cycle

    if answer.status != "answered":
        return {}
    referenced = extract_citation_indices(answer.text)
    valid = answer.usage.get("citation_indices", [])
    invalid = answer.usage.get("invalid_citation_indices", [])
    total_markers = len(valid) + len(invalid)
    sentences = split_sentences(answer.text)
    cited_sentences = sum(1 for s in sentences if extract_citation_indices(s))
    # Full context, not snippet: the model saw the full text, so the support
    # proxy must be measured against what it actually saw.
    context = "\n".join(c.context or c.snippet for c in answer.citations)
    aterms = content_terms(answer.text)
    return {
        "n_citations": len(valid),
        "n_invalid_citations": len(invalid),
        "citation_validity": round(len(valid) / total_markers, 4) if total_markers else None,
        "n_sentences": len(sentences),
        "sentence_citation_coverage": round(cited_sentences / len(sentences), 4) if sentences else None,
        "answer_support_proxy": round(term_support(answer.text, context, aterms), 4) if aterms else None,
        "uncited": bool(answer.usage.get("uncited", False)),
        "referenced_markers": referenced,
    }


def evaluate(
    pipeline: Pipeline,
    questions: Sequence[EvalQuestion],
    *,
    k: int | None = None,
    run_generation: bool = True,
    sweep_thresholds: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation and return a JSON-serialisable report."""
    settings = pipeline.settings
    k = k or settings.retrieve_top_k
    generation_configured = pipeline.llm.is_configured()

    per_question: list[dict[str, Any]] = []
    stage_rows: dict[str, list[dict[str, float]]] = {s: [] for s in STAGES}
    # Per-question id lists per stage, kept so metrics can be recomputed at
    # several cut-offs without re-running retrieval (see `by_k` below).
    stage_ids_by_q: list[dict[str, list[int]]] = []
    relevant_by_q: list[set[int]] = []
    retr_latencies: list[float] = []
    e2e_latencies: list[float] = []
    confidences: list[float] = []
    conf_by_answerable: list[tuple[bool, float]] = []
    grounded: list[dict[str, Any]] = []
    end_to_end_abstain: list[tuple[bool, bool]] = []  # (answerable, abstained)
    skipped_retrieval = 0

    for q in questions:
        relevant_chunks, relevant_docs = resolve_relevant(pipeline, q)
        if q.answerable and not relevant_chunks:
            # The label references text that is not in the index: either the
            # corpus changed or the evidence string is wrong. Surface it instead
            # of quietly scoring the question as a miss.
            log.warning("question %s: evidence matched no indexed chunk (label/index drift?)", q.id)
            skipped_retrieval += 1

        t0 = time.perf_counter()
        result = pipeline.retrieve(q.question)
        retr_ms = (time.perf_counter() - t0) * 1000.0
        retr_latencies.append(retr_ms)

        conf = result.diagnostics.evidence_confidence
        confidences.append(conf)
        conf_by_answerable.append((q.answerable, conf))

        ref = [sc.chunk.chunk_id for sc in result.hits]
        retrieved_by_stage = {
            "bm25_only": [i for i, _ in result.bm25],
            "vector_only": [i for i, _ in result.vector],
            "hybrid": list(result.fused_ids),
            "hybrid+rerank": ref,
        }
        stages: dict[str, dict[str, float]] = {}
        for stage, ids in retrieved_by_stage.items():
            m = _stage_metrics(ids, relevant_chunks, k)
            if m:
                stages[stage] = m
                stage_rows[stage].append(m)
        # Only questions that actually have relevant chunks enter the aggregate,
        # so an unanswerable question cannot drag recall down to a fake value.
        if relevant_chunks:
            stage_ids_by_q.append(retrieved_by_stage)
            relevant_by_q.append(relevant_chunks)

        doc_recall = None
        if relevant_docs:
            got_docs = {sc.chunk.doc_id for sc in result.hits}
            doc_recall = round(len(got_docs & relevant_docs) / len(relevant_docs), 4)

        row: dict[str, Any] = {
            "id": q.id,
            "question": q.question,
            "answerable": q.answerable,
            "n_relevant_chunks": len(relevant_chunks),
            "evidence_confidence": round(conf, 4),
            "retrieval_latency_ms": round(retr_ms, 1),
            "stages": stages,
            "doc_recall": doc_recall,
            "top_chunks": [
                {
                    "chunk_id": sc.chunk.chunk_id,
                    "doc_id": sc.chunk.doc_id,
                    "ordinal": sc.chunk.ordinal,
                    "page": sc.chunk.page,
                    "score": round(sc.score, 4),
                    "source": sc.source,
                    "ranks": sc.ranks,
                }
                for sc in result.hits
            ],
        }

        if run_generation:
            t1 = time.perf_counter()
            answer = pipeline.generator.answer(q.question, result)
            e2e_ms = (time.perf_counter() - t1) * 1000.0 + retr_ms
            e2e_latencies.append(e2e_ms)
            row["status"] = answer.status
            row["abstained"] = answer.abstained
            row["answer"] = answer.text
            row["reason"] = answer.reason
            row["n_citations"] = len(answer.citations)
            g = _groundedness(answer)
            if g:
                grounded.append(g)
                row["grounding"] = g
            # End-to-end abstention uses the *retrieval-only* gate when no LLM is
            # configured, otherwise the full path (gate + model refusal).
            if generation_configured:
                end_to_end_abstain.append((q.answerable, answer.abstained))
            else:
                end_to_end_abstain.append((q.answerable, conf < settings.min_evidence_confidence))
            row["end_to_end_latency_ms"] = round(e2e_ms, 1)

        per_question.append(row)

    # ---------------------------------------------------------------- retrieval
    retrieval_report = {
        "k": k,
        "n_questions": len(questions),
        "n_answerable": sum(1 for q in questions if q.answerable),
        "n_unanswerable": sum(1 for q in questions if not q.answerable),
        "questions_with_no_relevant_chunk_in_index": skipped_retrieval,
        "by_stage": {stage: _aggregate(rows) for stage, rows in stage_rows.items()},
        # Metrics recomputed at several cut-offs from the same runs, which makes
        # the effect of the cut-off (and of re-ranking at rank 1) visible.
        "by_k": _metrics_by_k(stage_ids_by_q, relevant_by_q, ks=(1, 3, 5, k) if k > 5 else (1, 3, k)),
        "doc_recall_mean": round(
            mean([r["doc_recall"] for r in per_question if r["doc_recall"] is not None]), 4
        )
        if any(r["doc_recall"] is not None for r in per_question)
        else None,
    }

    # --------------------------------------------------------------- abstention
    abstention = _abstention_report(
        conf_by_answerable,
        threshold=settings.min_evidence_confidence,
        sweep=sweep_thresholds,
    )

    # -------------------------------------------------------------- generation
    gen_report: dict[str, Any] | None = None
    if run_generation:
        answered = [r for r in per_question if r.get("status") == "answered"]
        gen_report = {
            "generation_configured": generation_configured,
            "llm_client": pipeline.llm.name,
            "n_answered": len(answered),
            "n_abstained": sum(1 for r in per_question if r.get("abstained")),
            "status_histogram": _histogram(r.get("status") for r in per_question),
            "citation_validity_mean": _mean_of(grounded, "citation_validity"),
            "sentence_citation_coverage_mean": _mean_of(grounded, "sentence_citation_coverage"),
            "answer_support_proxy_mean": _mean_of(grounded, "answer_support_proxy"),
            "uncited_answers": sum(1 for g in grounded if g["uncited"]),
            "invalid_citation_answers": sum(1 for g in grounded if g["n_invalid_citations"] > 0),
            "note": (
                "Citation/support metrics are proxies, not entailment checks. "
                "answer_support_proxy only detects vocabulary not present in context."
            ),
            "answerable_false_abstain_rate": _rate(
                [abstained for answerable, abstained in end_to_end_abstain if answerable]
            ),
            "unanswerable_abstain_rate": _rate(
                [abstained for answerable, abstained in end_to_end_abstain if not answerable]
            ),
        }

    # ------------------------------------------------------------------ latency
    latency = {
        "retrieval_ms": {
            "p50": round(percentile(retr_latencies, 50), 1),
            "p95": round(percentile(retr_latencies, 95), 1),
            "mean": round(mean(retr_latencies), 1),
            "n": len(retr_latencies),
        }
    }
    if e2e_latencies:
        latency["end_to_end_ms"] = {
            "p50": round(percentile(e2e_latencies, 50), 1),
            "p95": round(percentile(e2e_latencies, 95), 1),
            "mean": round(mean(e2e_latencies), 1),
            "n": len(e2e_latencies),
        }

    # ----------------------------------------------------------------- metadata
    stub = pipeline.embedder.is_stub
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quality_metrics_valid": not stub,
        "warning": (
            "EMBEDDINGS ARE THE HASH STUB: they are not semantic. Retrieval numbers "
            "below measure a bag-of-words toy and MUST NOT be presented as the "
            "quality of the real (fastembed) system."
        )
        if stub
        else None,
        "environment": {
            "embeddings": {"backend": pipeline.embedder.name, "dim": pipeline.embedder.dim, "is_stub": stub},
            "vectors": {"backend": pipeline.store.vector_backend, "sqlite_vec_error": pipeline.store.sqlite_vec_error},
            "reranker": {"backend": pipeline.reranker.name, "is_stub": pipeline.reranker.is_stub},
            "generation": {"configured": generation_configured, "client": pipeline.llm.name},
            "notes": pipeline.notes,
            "index": pipeline.store.stats(),
        },
        "retrieval": retrieval_report,
        "abstention": abstention,
        "generation": gen_report,
        "latency": latency,
        "per_question": per_question,
    }
    return report


def _mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(vals), 4) if vals else None


def _rate(flags: list[bool]) -> float | None:
    return round(sum(1 for f in flags if f) / len(flags), 4) if flags else None


def _histogram(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _abstention_report(
    conf_by_answerable: Sequence[tuple[bool, float]],
    *,
    threshold: float,
    sweep: bool,
) -> dict[str, Any]:
    """Retrieval-only abstention accuracy, plus a threshold sweep."""
    if not conf_by_answerable:
        return {}

    def at(t: float) -> dict[str, Any]:
        # Abstain when confidence < t. "Correct" means: abstained on an
        # unanswerable question, or did not abstain on an answerable one.
        tp = sum(1 for ans, c in conf_by_answerable if not ans and c < t)
        fn = sum(1 for ans, c in conf_by_answerable if not ans and c >= t)
        fp = sum(1 for ans, c in conf_by_answerable if ans and c < t)
        tn = sum(1 for ans, c in conf_by_answerable if ans and c >= t)
        total = tp + fn + fp + tn
        return {
            "threshold": round(t, 4),
            "accuracy": round((tp + tn) / total, 4) if total else None,
            "unanswerable_correct_abstain_rate": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "answerable_false_abstain_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "n_unanswerable": tp + fn,
            "n_answerable": fp + tn,
        }

    report: dict[str, Any] = {
        "method": "retrieval-only evidence-confidence gate (LLM-independent)",
        "configured": at(threshold),
    }
    if sweep:
        grid = sorted({0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7})
        report["sweep"] = [at(t) for t in grid]
        best = max(
            (row for row in report["sweep"] if row["accuracy"] is not None),
            key=lambda row: (row["accuracy"], -(row["answerable_false_abstain_rate"] or 0)),
            default=None,
        )
        report["best_by_accuracy"] = best
    return report


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def format_summary(report: dict[str, Any]) -> str:
    """Human-readable console summary. Kept stable so it can be piped into docs."""
    lines: list[str] = []
    env = report["environment"]
    lines.append("=" * 72)
    lines.append("rag-doc-qa evaluation summary")
    lines.append("=" * 72)
    lines.append(f"generated:        {report['generated_at']}")
    lines.append(f"embeddings:       {env['embeddings']['backend']} (dim={env['embeddings']['dim']}, stub={env['embeddings']['is_stub']})")
    lines.append(f"vector backend:   {env['vectors']['backend']}")
    lines.append(f"reranker:         {env['reranker']['backend']} (stub={env['reranker']['is_stub']})")
    lines.append(f"generation:       {'configured' if env['generation']['configured'] else 'OFFLINE / not configured'}")
    if report.get("warning"):
        lines.append("")
        lines.append(f"!! {report['warning']}")
    lines.append("")
    r = report["retrieval"]
    kk = r["k"]
    lines.append(f"retrieval (k={kk}, n={r['n_questions']}, answerable={r['n_answerable']}, unanswerable={r['n_unanswerable']})")
    lines.append(f"  {'stage':<16} {'recall@k':>9} {'prec@k':>8} {'ndcg@k':>8} {'mrr':>7} {'hit@k':>7}")
    for stage in STAGES:
        m = r["by_stage"].get(stage)
        if not m:
            continue
        lines.append(
            f"  {stage:<16} {m.get(f'recall@{kk}', 0):>9.3f} {m.get(f'precision@{kk}', 0):>8.3f} "
            f"{m.get(f'ndcg@{kk}', 0):>8.3f} {m.get('mrr', 0):>7.3f} {m.get(f'hit@{kk}', 0):>7.3f}"
        )
    if r.get("questions_with_no_relevant_chunk_in_index"):
        lines.append(f"  NOTE: {r['questions_with_no_relevant_chunk_in_index']} answerable question(s) matched no indexed chunk")
    by_k = r.get("by_k") or {}
    if by_k:
        lines.append("")
        lines.append("  metrics by cut-off (recall@k / ndcg@k):")
        for kk in sorted(by_k, key=int):
            hybrid = by_k[kk].get("hybrid") or {}
            reranked = by_k[kk].get("hybrid+rerank") or {}
            bm25 = by_k[kk].get("bm25_only") or {}
            vec = by_k[kk].get("vector_only") or {}
            lines.append(
                f"    k={kk:<3} bm25 {bm25.get(f'recall@{kk}', 0):.3f}/{bm25.get(f'ndcg@{kk}', 0):.3f}"
                f"  vector {vec.get(f'recall@{kk}', 0):.3f}/{vec.get(f'ndcg@{kk}', 0):.3f}"
                f"  hybrid {hybrid.get(f'recall@{kk}', 0):.3f}/{hybrid.get(f'ndcg@{kk}', 0):.3f}"
                f"  +rerank {reranked.get(f'recall@{kk}', 0):.3f}/{reranked.get(f'ndcg@{kk}', 0):.3f}"
            )
    lines.append("")
    ab = report.get("abstention") or {}
    if ab.get("configured"):
        c = ab["configured"]
        lines.append(
            f"abstention (retrieval-only): threshold={c['threshold']} accuracy={c['accuracy']} "
            f"unanswerable-correct={c['unanswerable_correct_abstain_rate']} answerable-false-abstain={c['answerable_false_abstain_rate']}"
        )
        if ab.get("best_by_accuracy"):
            b = ab["best_by_accuracy"]
            lines.append(f"  best threshold by accuracy: {b['threshold']} -> accuracy={b['accuracy']}")
    gen = report.get("generation")
    lines.append("")
    if gen:
        lines.append(f"generation: answered={gen['n_answered']} abstained={gen['n_abstained']}")
        lines.append(f"  status histogram: {gen['status_histogram']}")
        if gen.get("citation_validity_mean") is not None:
            lines.append(
                f"  citation validity={gen['citation_validity_mean']} "
                f"sentence citation coverage={gen['sentence_citation_coverage_mean']} "
                f"answer-support(proxy)={gen['answer_support_proxy_mean']}"
            )
        lines.append(
            f"  answerable false-abstain rate={gen['answerable_false_abstain_rate']} "
            f"unanswerable abstain rate={gen['unanswerable_abstain_rate']}"
        )
    else:
        lines.append("generation: not run (retrieval-only evaluation)")
    lines.append("")
    lat = report["latency"]
    lines.append(
        f"latency: retrieval p50={lat['retrieval_ms']['p50']}ms p95={lat['retrieval_ms']['p95']}ms"
        + (
            f" | end-to-end p50={lat['end_to_end_ms']['p50']}ms p95={lat['end_to_end_ms']['p95']}ms"
            if "end_to_end_ms" in lat
            else ""
        )
    )
    lines.append("=" * 72)
    return "\n".join(lines)
