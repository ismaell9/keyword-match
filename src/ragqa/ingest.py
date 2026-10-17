"""Document loading: PDF, Markdown, plain text -- with graceful failure.

Ingestion must never take down the pipeline because one file in a folder is
corrupt, encrypted, or an unsupported type. Every loader either returns a
`Document` or raises `IngestError` with a human-readable reason that the caller
records and moves past.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .types import Document, Segment, make_doc_id

# Extensions we claim to handle. Anything else is skipped with a recorded reason
# (not silently) so /ingest can report partial success honestly.
TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".json", ".yaml", ".yml"}
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | PDF_EXTS


class IngestError(Exception):
    """Raised when a single file cannot be turned into a Document."""


@dataclass(slots=True)
class IngestReport:
    """Outcome of ingesting a set of files. `errors`/`skipped` are first-class."""

    indexed: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    errors: list[dict[str, Any]]

    @property
    def n_indexed(self) -> int:
        return len(self.indexed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "indexed": self.indexed,
            "skipped": self.skipped,
            "errors": self.errors,
            "counts": {
                "indexed": len(self.indexed),
                "skipped": len(self.skipped),
                "errors": len(self.errors),
            },
        }


def iter_files(root: Path) -> Iterator[Path]:
    """Yield supported files under `root`, deterministically ordered.

    Deterministic order matters: ingestion is incremental by content hash, and a
    stable order makes logs and chunk ids reproducible across runs.
    """
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("."):
            yield p


def _load_text(path: Path) -> list[Segment]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1 rather than failing: banking exports and old
        # Windows files are frequently cp1252-ish. Lossy but better than a 500.
        raw = path.read_text(encoding="latin-1", errors="replace")
    return [Segment(text=raw, page=None)]


def _load_pdf(path: Path) -> list[Segment]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise IngestError("pypdf is not installed; cannot read PDF") from exc

    try:
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # many "encrypted" PDFs use an empty password
            except Exception as exc:
                raise IngestError("PDF is encrypted and cannot be opened") from exc
        segments: list[Segment] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # a single bad page shouldn't kill the doc
                text = f"[page {i}: extraction failed: {exc}]"
            segments.append(Segment(text=text, page=i))
        if not any(s.text.strip() for s in segments):
            # A scanned PDF with no text layer. We do NOT silently OCR: that is a
            # separate, expensive pipeline (out of scope for v0) and pretending
            # otherwise would be the kind of overclaim this project avoids.
            raise IngestError("PDF has no extractable text layer (likely scanned; OCR not implemented)")
        return segments
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"failed to parse PDF: {exc}") from exc


def load_document(path: Path, *, root: Path | None = None) -> Document:
    """Load one file into a Document. Raises IngestError on unsupported/corrupt input."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise IngestError(f"unsupported extension '{suffix or '(none)'}'")

    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    rel = str(path.relative_to(root)) if root and path.is_relative_to(root) else path.name

    if suffix in PDF_EXTS:
        segments = _load_pdf(path)
    else:
        segments = _load_text(path)

    title = _derive_title(path, segments)
    doc_id = make_doc_id(rel, sha)
    meta: dict[str, Any] = {
        "extension": suffix,
        "size_bytes": len(data),
        "loader": "pdf" if suffix in PDF_EXTS else "text",
    }
    return Document(
        doc_id=doc_id,
        path=rel,
        title=title,
        segments=segments,
        sha256=sha,
        meta=meta,
    )


def _derive_title(path: Path, segments: list[Segment]) -> str:
    """Use the first Markdown H1 if present, else the file stem."""
    if segments:
        for line in segments[0].text.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            if line:
                break
    return path.stem.replace("-", " ").replace("_", " ").strip() or path.name


def load_documents(root: Path) -> tuple[list[Document], IngestReport]:
    """Load every supported file under `root`, returning docs AND a full report."""
    base = root if root.is_dir() else root.parent
    docs: list[Document] = []
    indexed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for path in iter_files(root):
        try:
            doc = load_document(path, root=base)
        except IngestError as exc:
            errors.append({"path": str(path), "reason": str(exc)})
            continue
        except Exception as exc:  # defensive: never let one file abort the scan
            errors.append({"path": str(path), "reason": f"unexpected: {exc}"})
            continue
        docs.append(doc)
        indexed.append(
            {
                "path": doc.path,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "segments": len(doc.segments),
                "chars": len(doc.text),
            }
        )
    return docs, IngestReport(indexed=indexed, skipped=skipped, errors=errors)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines. Raises with line context on bad JSON."""
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def iter_supported(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p
