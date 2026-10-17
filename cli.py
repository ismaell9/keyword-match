#!/usr/bin/env python3
"""Command-line interface for rag-doc-qa.

Examples
--------
    python cli.py ingest                       # ingest data/corpus
    python cli.py ingest --rebuild             # wipe the index first
    python cli.py ask "how often must passwords be rotated?"
    python cli.py retrieve "incident response steps" --top-k 5
    python cli.py eval --out data/eval/results.json
    python cli.py stats
    python cli.py config
    python cli.py serve

The CLI deliberately adds `src/` to sys.path so it works from a fresh checkout
with no install step. Installing the package (`pip install -e .`) works too and
takes precedence when present.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragqa.config import get_settings  # noqa: E402
from ragqa.evaluation import (  # noqa: E402
    evaluate,
    format_summary,
    load_eval_set,
    write_report,
)
from ragqa.ingest import IngestError  # noqa: E402
from ragqa.pipeline import build_pipeline, iter_supported_paths  # noqa: E402

DIM = "\033[2m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    return text if not sys.stdout.isatty() else f"{color}{text}{RESET}"


def _banner(pipeline) -> None:
    """Print the honest configuration banner: real vs stub, online vs offline."""
    print(_c("configuration", BOLD))
    for note in pipeline.notes:
        print(f"  - {note}")
    if pipeline.embedder.is_stub:
        print(_c("  !! EMBEDDINGS ARE THE HASH STUB (not semantic; numbers are meaningless)", RED))
    if not pipeline.llm.is_configured():
        print(_c("  !! generation offline: retrieval works, answers will not be produced", YELLOW))
    stats = pipeline.store.stats()
    print(f"  index: {stats['documents']} documents, {stats['chunks']} chunks, {stats['vectors']} vectors")
    print()


# ------------------------------------------------------------------- commands


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(load_llm=False)
    if args.rebuild:
        print(_c("rebuilding index (dropping all indexed content) ...", YELLOW))
        pipeline.rebuild()

    if args.paths:
        report = pipeline.ingest_paths(iter_supported_paths([Path(p) for p in args.paths]), force=args.force)
    else:
        report = pipeline.ingest_corpus(force=args.force)

    out = report.as_dict()
    for entry in out["indexed"]:
        print(f"  {_c('indexed', GREEN)} {entry['path']}  ({entry.get('chunks', '?')} chunks)")
    for entry in out["skipped"]:
        print(f"  {_c('skipped', DIM)} {entry['path']}  ({entry.get('reason', '')})")
    for entry in out["errors"]:
        print(f"  {_c('error', RED)}   {entry['path']}  ({entry['reason']})")
    print(_c(f"done: {out['counts']}", BOLD))
    if args.json:
        print(json.dumps(out, indent=2))
    pipeline.close()
    return 1 if out["counts"]["errors"] else 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(load_llm=not args.no_llm)
    _banner(pipeline)
    answer = pipeline.query(args.question, top_k=args.top_k)

    status_color = GREEN if answer.status == "answered" else YELLOW
    print(f"{_c('status:', BOLD)} {_c(answer.status, status_color)}")
    if answer.reason:
        print(f"{_c('reason:', BOLD)} {answer.reason}")
    if answer.text:
        print()
        print(answer.text)
    if answer.citations:
        print()
        print(_c("citations", BOLD))
        for c in answer.citations:
            page = f" p.{c.page}" if c.page is not None else ""
            print(f"  {c.marker} {c.doc_title}  [{c.path}{page} chunk {c.ordinal}]")
            print(_c(f"      {c.snippet[:200]}", DIM))
    if answer.retrieved and args.show_context:
        print()
        print(_c("retrieved chunks (all)", BOLD))
        for sc in answer.retrieved:
            print(f"  #{sc.chunk.chunk_id} score={sc.score:.4f} src={sc.source} ranks={sc.ranks}")
            print(_c(f"    {sc.chunk.text[:180]}", DIM))
    print()
    print(
        _c(
            f"latency {answer.latency_ms:.0f}ms | evidence_conf {answer.diagnostics.evidence_confidence:.3f} "
            f"| bm25={answer.diagnostics.candidates_bm25} vector={answer.diagnostics.candidates_vector}",
            DIM,
        )
    )
    pipeline.close()
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(load_llm=False)
    _banner(pipeline)
    result = pipeline.retrieve(args.question, top_k=args.top_k)
    print(f"{_c('bm25 candidates:', BOLD)} {len(result.bm25)}   {_c('vector candidates:', BOLD)} {len(result.vector)}")
    print(f"{_c('evidence confidence:', BOLD)} {result.diagnostics.evidence_confidence:.4f}")
    print()
    for i, sc in enumerate(result.hits, start=1):
        page = f" p.{sc.chunk.page}" if sc.chunk.page is not None else ""
        print(f"{i:>2}. #{sc.chunk.chunk_id} score={sc.score:.4f} source={sc.source}{page}")
        print(f"    ranks={sc.ranks} doc={sc.chunk.doc_id}")
        print(_c(f"    {sc.chunk.text[:240]}", DIM))
    pipeline.close()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    if not dataset.exists():
        print(_c(f"dataset not found: {dataset}", RED))
        return 2
    pipeline = build_pipeline(
        load_llm=not args.no_llm,
        embed_backend=args.embed_backend,
        rerank_enabled=(False if args.no_rerank else None),
    )
    _banner(pipeline)
    questions = load_eval_set(dataset)
    print(f"loaded {len(questions)} questions from {dataset}")
    report = evaluate(
        pipeline,
        questions,
        k=args.k,
        run_generation=not args.no_generation,
        sweep_thresholds=not args.no_sweep,
    )
    print()
    print(format_summary(report))
    if args.out:
        path = write_report(report, args.out)
        print(f"\nfull report written to {path}")
    if args.json:
        print(json.dumps(report, indent=2))
    pipeline.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(load_llm=False)
    print(json.dumps(pipeline.store.stats(), indent=2))
    pipeline.close()
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    settings = get_settings()
    print(json.dumps(json.loads(settings.model_dump_json()), indent=2))
    print()
    pipeline = build_pipeline(load_llm=not args.no_llm)
    _banner(pipeline)
    pipeline.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host
    print(_c(f"serving on http://{host}:{args.port} (docs at /docs)", BOLD))
    if host not in ("127.0.0.1", "localhost"):
        print(_c("WARNING: binding to a non-loopback address with no authentication", RED))
    uvicorn.run("ragqa.api:app", host=host, port=args.port, log_level=args.log_level, reload=args.reload)
    return 0


# ---------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="rag-doc-qa: hybrid retrieval + grounded answers with citations",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest documents into the index")
    p.add_argument("paths", nargs="*", help="files or directories (default: CORPUS_DIR)")
    p.add_argument("--rebuild", action="store_true", help="drop all indexed content first")
    p.add_argument("--force", action="store_true", help="re-embed unchanged files")
    p.add_argument("--json", action="store_true", help="also print the raw JSON report")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question and print the grounded answer")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--no-llm", action="store_true", help="retrieval + abstention only, no model call")
    p.add_argument("--show-context", action="store_true", help="print every retrieved chunk")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("retrieve", help="show ranked chunks for a query (no LLM)")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=None)
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("eval", help="run the evaluation harness")
    p.add_argument("--dataset", default=str(REPO_ROOT / "data" / "eval" / "questions.jsonl"))
    p.add_argument("--k", type=int, default=None, help="cut-off for recall/nDCG (default: RETRIEVE_TOP_K)")
    p.add_argument("--out", default=None, help="write the full JSON report here")
    p.add_argument("--json", action="store_true", help="also print the full JSON report")
    p.add_argument("--no-llm", action="store_true", help="do not build an LLM client at all")
    p.add_argument("--no-generation", action="store_true", help="skip the generation stage entirely")
    p.add_argument("--no-rerank", action="store_true", help="disable the cross-encoder (ablation)")
    p.add_argument("--no-sweep", action="store_true", help="skip the abstention threshold sweep")
    p.add_argument(
        "--embed-backend",
        choices=["auto", "fastembed", "hash"],
        default=None,
        help="override EMBED_BACKEND (hash = labelled stub, for smoke tests)",
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("stats", help="print index statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("config", help="print resolved configuration and capability banner")
    p.add_argument("--no-llm", action="store_true")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # HuggingFace/ONNX progress bars are noisy in a CLI; quiet them unless -v.
    if not args.verbose:
        for noisy in ("httpx", "urllib3", "huggingface_hub", "onnxruntime", "filelock"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return int(args.func(args))
    except IngestError as exc:
        print(_c(f"ingest error: {exc}", RED))
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print(_c("interrupted", YELLOW))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
