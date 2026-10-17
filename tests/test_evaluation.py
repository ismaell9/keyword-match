"""End-to-end tests for the evaluation harness.

The harness is the project's differentiator, so it gets tested like production
code: label loading/validation, evidence resolution, metric wiring, and -- most
importantly -- that it refuses to present stub-embedder numbers as quality.
"""

from __future__ import annotations

import json

import pytest

from ragqa.evaluation import (
    EvalQuestion,
    _abstention_report,
    evaluate,
    format_summary,
    load_eval_set,
    resolve_relevant,
    write_report,
)
from ragqa.ingest import IngestError


def _questions() -> list[EvalQuestion]:
    return [
        EvalQuestion(id="a1", question="how often must passwords be rotated", answerable=True, evidence=["rotated every 90 days"]),
        EvalQuestion(id="a2", question="is multi-factor authentication required", answerable=True, evidence=["multi-factor authentication"]),
        EvalQuestion(id="u1", question="what is the sabbatical policy", answerable=False, evidence=[]),
        EvalQuestion(id="u2", question="how much does the vendor charge per year", answerable=False, evidence=[]),
    ]


def test_load_eval_set_reads_jsonl(tmp_path):
    p = tmp_path / "q.jsonl"
    rows = [
        {"id": "q1", "question": "x?", "answerable": True, "evidence": ["y"]},
        {"id": "q2", "question": "z?", "answerable": False},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    questions = load_eval_set(p)
    assert [q.id for q in questions] == ["q1", "q2"]
    assert questions[0].answerable is True
    assert questions[1].answerable is False


def test_load_eval_set_rejects_answerable_without_evidence(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"id": "q1", "question": "x?", "answerable": True}) + "\n", encoding="utf-8")
    with pytest.raises(IngestError):
        load_eval_set(p)


def test_load_eval_set_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(
        json.dumps({"id": "dup", "question": "x?", "answerable": False})
        + "\n"
        + json.dumps({"id": "dup", "question": "y?", "answerable": False})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(IngestError):
        load_eval_set(p)


def test_load_eval_set_rejects_empty_file(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(IngestError):
        load_eval_set(p)


def test_resolve_relevant_maps_evidence_to_chunks(indexed_pipeline):
    q = EvalQuestion(id="x", question="?", answerable=True, evidence=["rotated every 90 days"])
    chunks, docs = resolve_relevant(indexed_pipeline, q)
    assert chunks, "evidence string should match at least one chunk"
    assert docs, "doc ids should be derived from the evidence"


def test_evaluate_produces_the_full_report_shape(indexed_pipeline):
    report = evaluate(indexed_pipeline, _questions(), k=3, run_generation=False)

    assert report["quality_metrics_valid"] is False  # hash stub -> numbers are not quality
    assert "HASH STUB" in report["warning"]

    env = report["environment"]
    assert env["embeddings"]["is_stub"] is True
    assert env["vectors"]["backend"] in ("sqlite-vec", "numpy-brute-force")

    r = report["retrieval"]
    assert r["k"] == 3
    assert r["n_questions"] == 4
    assert r["n_answerable"] == 2
    assert r["n_unanswerable"] == 2
    # The answerable questions' evidence IS in the small sample corpus, so the
    # relevant chunk sets must resolve (this is a fixture-correctness assertion,
    # not a quality claim).
    assert r["questions_with_no_relevant_chunk_in_index"] == 0
    for stage in ("bm25_only", "vector_only", "hybrid", "hybrid+rerank"):
        assert stage in r["by_stage"]

    # Without a reranker, hybrid and hybrid+rerank must be identical.
    assert r["by_stage"]["hybrid"] == r["by_stage"]["hybrid+rerank"]

    assert report["abstention"]["configured"]["threshold"] == pytest.approx(0.15)
    assert report["abstention"]["sweep"]
    assert report["generation"] is None  # not run
    assert report["latency"]["retrieval_ms"]["n"] == 4
    assert len(report["per_question"]) == 4


def test_retrieval_only_run_has_no_generation_metrics(indexed_pipeline):
    report = evaluate(indexed_pipeline, _questions(), run_generation=False)
    assert report["generation"] is None
    assert "end_to_end_ms" not in report["latency"]


def test_hybrid_beats_or_matches_the_weakest_single_leg(indexed_pipeline):
    """Sanity property (not a quality claim): fusion should not be worse than the
    weaker leg on recall, because RRF is a union of ranks."""
    report = evaluate(indexed_pipeline, _questions(), k=5, run_generation=False)
    stages = report["retrieval"]["by_stage"]
    hybrid = stages["hybrid"][f"recall@5"]
    weakest = min(stages["bm25_only"][f"recall@5"], stages["vector_only"][f"recall@5"])
    assert hybrid >= weakest - 1e-9


def test_report_is_json_serialisable_and_written(indexed_pipeline, tmp_path):
    report = evaluate(indexed_pipeline, _questions(), run_generation=False)
    out = write_report(report, tmp_path / "nested" / "report.json")
    assert out.exists()
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["retrieval"]["n_questions"] == 4


def test_summary_mentions_the_stub_warning(indexed_pipeline):
    report = evaluate(indexed_pipeline, _questions(), run_generation=False)
    summary = format_summary(report)
    assert "HASH STUB" in summary
    assert "retrieval (k=" in summary


def test_abstention_report_confusion_matrix_is_correct():
    # confidence: answerable -> high, unanswerable -> low
    data = [(True, 0.9), (True, 0.8), (False, 0.02), (False, 0.05)]
    rep = _abstention_report(data, threshold=0.15, sweep=False)
    c = rep["configured"]
    assert c["accuracy"] == pytest.approx(1.0)
    assert c["unanswerable_correct_abstain_rate"] == pytest.approx(1.0)
    assert c["answerable_false_abstain_rate"] == pytest.approx(0.0)


def test_abstention_threshold_sweep_finds_the_separating_threshold():
    data = [(True, 0.6), (True, 0.7), (False, 0.1), (False, 0.2)]
    rep = _abstention_report(data, threshold=0.9, sweep=True)
    best = rep["best_by_accuracy"]
    assert best is not None
    assert best["accuracy"] == pytest.approx(1.0)
    assert 0.2 < best["threshold"] <= 0.6


def test_abstention_report_on_empty_input():
    assert _abstention_report([], threshold=0.15, sweep=True) == {}
