"""Text chunking.

The chunker is deliberately *paragraph aware* and *offset preserving*:

* It splits on paragraph boundaries first, only breaking mid-paragraph when a
  single paragraph is longer than `chunk_size_words`.
* Every chunk keeps exact `char_start`/`char_end` offsets into its source
  segment, which is what makes citations verifiable: a citation is not just
  "somewhere in this document" -- you can slice the original text.
* Chunks never span `Segment`s. For PDFs a Segment is one page, so a citation's
  page number is always the page the chunk actually starts on -- no guessing.

"Words" here are whitespace-delimited tokens: a documented proxy for subword
tokens. It keeps chunking embedder-agnostic and dependency-free. See
docs/architecture.md for the rationale and the accepted limitation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Chunk, Document, Segment

# A word is a run of non-whitespace. Good enough for English prose; the chunk
# *boundaries* are what matter, not perfect tokenisation.
_WORD_RE = re.compile(r"\S+")
# Blank-line separated paragraphs.
_PARA_RE = re.compile(r"\n\s*\n")


@dataclass(slots=True)
class _Tok:
    w: str
    start: int
    end: int
    para: int  # paragraph index within the segment


def _tokenize(text: str) -> list[_Tok]:
    """Tokenise into words tagged with a paragraph index and exact char offsets."""
    # Build paragraph spans, then tokenise each paragraph independently. Doing it
    # per-paragraph (rather than per-char) keeps the paragraph id trivially correct.
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _PARA_RE.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))

    tokens: list[_Tok] = []
    for para_idx, (a, b) in enumerate(spans):
        for m in _WORD_RE.finditer(text, a, b):
            tokens.append(_Tok(m.group(0), m.start(), m.end(), para_idx))
    return tokens


def _chunk_tokens(
    tokens: list[_Tok],
    *,
    size: int,
    overlap: int,
    min_size: int,
) -> list[tuple[int, int]]:
    """Return [start_idx, end_idx) token ranges.

    Ends a chunk at a paragraph boundary when one is available in the last 40% of
    the window; otherwise hard-cuts at `size`. Then slides forward by
    `size - overlap`, i.e. consecutive chunks share `overlap` words.
    """
    n = len(tokens)
    if n == 0:
        return []
    if overlap >= size:
        overlap = max(0, size - 1)

    spans: list[tuple[int, int]] = []
    i = 0
    while i < n:
        hard_end = min(i + size, n)
        end = hard_end
        if hard_end < n:
            lower = i + max(min_size, int(size * 0.6))
            k = hard_end
            while k > lower:
                if tokens[k - 1].para != tokens[k].para:
                    end = k
                    break
                k -= 1
        if end <= i:  # pathological; guarantee forward progress
            end = min(i + 1, n)
        spans.append((i, end))
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return spans


def _merge_runt(out: list[Chunk], words: list[_Tok], segment_text: str) -> None:
    """Fold a trailing sub-`min_size` fragment into the previous chunk.

    Rationale: a final chunk made only of overlap residue adds noise (it
    duplicates content already indexed) while still costing a vector slot.
    Folding keeps the content searchable without creating a near-duplicate.
    """
    if not out:
        return
    prev = out[-1]
    prev.text = segment_text[prev.char_start : words[-1].end]
    prev.char_end = words[-1].end
    prev.n_words = len(_WORD_RE.findall(prev.text))


def chunk_segment(
    segment: Segment,
    doc_id: str,
    *,
    size: int,
    overlap: int,
    min_size: int,
    ordinal_offset: int = 0,
) -> list[Chunk]:
    """Chunk one Segment. `chunk_id` stays -1 until the store assigns a rowid."""
    tokens = _tokenize(segment.text)
    if not tokens:
        return []
    ranges = _chunk_tokens(tokens, size=size, overlap=overlap, min_size=min_size)
    out: list[Chunk] = []
    for ordinal, (a, b) in enumerate(ranges):
        words = tokens[a:b]
        if not words:
            continue
        is_trailing_runt = len(words) < min_size and len(ranges) > 1 and b >= len(tokens)
        if is_trailing_runt:
            _merge_runt(out, words, segment.text)
            continue
        out.append(
            Chunk(
                chunk_id=-1,
                doc_id=doc_id,
                ordinal=ordinal_offset + ordinal,
                text=segment.text[words[0].start : words[-1].end],
                page=segment.page,
                char_start=words[0].start,
                char_end=words[-1].end,
                n_words=len(words),
            )
        )
    return out


def chunk_document(document: Document, *, size: int, overlap: int, min_size: int) -> list[Chunk]:
    """Chunk all segments of a Document, assigning contiguous ordinals across pages."""
    chunks: list[Chunk] = []
    for segment in document.segments:
        new = chunk_segment(
            segment,
            document.doc_id,
            size=size,
            overlap=overlap,
            min_size=min_size,
            ordinal_offset=len(chunks),
        )
        chunks.extend(new)
    # Re-number ordinals so they are strictly contiguous even after runt folding.
    for idx, chunk in enumerate(chunks):
        chunk.ordinal = idx
    return chunks
