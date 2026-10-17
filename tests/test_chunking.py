"""Chunking: offsets must be exact, overlap must be respected, pages preserved."""

from __future__ import annotations

from ragqa.chunking import chunk_document, chunk_segment
from ragqa.ingest import load_document
from ragqa.types import Document, Segment


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_offsets_slice_back_to_exact_chunk_text():
    """A citation is only verifiable if text[char_start:char_end] == chunk.text."""
    text = _words(500)
    chunks = chunk_segment(Segment(text=text), "doc", size=100, overlap=20, min_size=30)
    assert chunks
    for c in chunks:
        assert text[c.char_start : c.char_end] == c.text


def test_no_chunk_exceeds_size_unless_it_folds_a_runt():
    text = _words(1000)
    chunks = chunk_segment(Segment(text=text), "doc", size=100, overlap=20, min_size=30)
    for c in chunks[:-1]:
        # The final chunk may be larger because a trailing runt is folded in.
        assert c.n_words <= 100


def test_consecutive_chunks_overlap():
    text = _words(400)
    chunks = chunk_segment(Segment(text=text), "doc", size=100, overlap=30, min_size=25)
    assert len(chunks) >= 3
    for prev, cur in zip(chunks, chunks[1:]):
        # Overlap means the next chunk starts before the previous one ended.
        assert cur.char_start < prev.char_end


def test_all_content_is_covered_by_at_least_one_chunk():
    text = " ".join(f"token{i}" for i in range(300))
    chunks = chunk_segment(Segment(text=text), "doc", size=80, overlap=10, min_size=20)
    joined = " ".join(c.text for c in chunks)
    for i in range(300):
        assert f"token{i}" in joined


def test_page_numbers_are_preserved_per_segment():
    doc = Document(
        doc_id="d",
        path="p.pdf",
        title="t",
        segments=[
            Segment(text=_words(300), page=1),
            Segment(text=_words(300), page=2),
        ],
        sha256="x",
    )
    chunks = chunk_document(doc, size=50, overlap=10, min_size=15)
    assert chunks
    # A chunk never spans segments, so every chunk's page is unambiguous.
    assert {c.page for c in chunks} == {1, 2}


def test_chunks_are_ordered_by_page():
    doc = Document(
        doc_id="d",
        path="p.pdf",
        title="t",
        segments=[Segment(text=_words(200), page=1), Segment(text=_words(200), page=2)],
        sha256="x",
    )
    pages = [c.page for c in chunk_document(doc, size=50, overlap=10, min_size=15)]
    assert pages == sorted(pages)


def test_ordinals_are_contiguous_from_zero():
    doc = Document(
        doc_id="d",
        path="p.md",
        title="t",
        segments=[Segment(text=_words(500)), Segment(text=_words(500))],
        sha256="x",
    )
    chunks = chunk_document(doc, size=60, overlap=10, min_size=20)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_empty_and_whitespace_only_text_produce_no_chunks():
    assert chunk_segment(Segment(text=""), "d", size=50, overlap=5, min_size=10) == []
    assert chunk_segment(Segment(text="   \n\n  \t "), "d", size=50, overlap=5, min_size=10) == []


def test_short_document_still_produces_one_chunk():
    """A doc smaller than min_size must still be retrievable, not dropped."""
    chunks = chunk_segment(Segment(text="just a few words here"), "d", size=100, overlap=10, min_size=50)
    assert len(chunks) == 1
    assert chunks[0].text == "just a few words here"


def test_overlap_larger_than_size_does_not_loop_forever():
    chunks = chunk_segment(Segment(text=_words(60)), "d", size=10, overlap=99, min_size=2)
    assert chunks
    assert len(chunks) < 200  # would explode if progress were not guaranteed


def test_paragraph_boundary_is_preferred_when_available():
    # Two clean paragraphs; the first chunk should end exactly at the paragraph
    # break rather than mid-paragraph, because a boundary is available.
    para1 = "alpha " * 20
    para2 = "beta " * 20
    text = para1.strip() + "\n\n" + para2.strip()
    chunks = chunk_segment(Segment(text=text), "d", size=25, overlap=2, min_size=10)
    assert len(chunks) >= 2
    # Everything after the first chunk's end should be the second paragraph.
    assert text[chunks[0].char_end :].lstrip().startswith("beta")


def test_ingest_loads_markdown_title_from_h1(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# My Document Title\n\nbody text here\n", encoding="utf-8")
    doc = load_document(path, root=tmp_path)
    assert doc.title == "My Document Title"
    assert doc.page_count == 0  # markdown has no pages


def test_runt_folding_does_not_drop_content():
    text = _words(100)
    chunks = chunk_segment(Segment(text=text), "d", size=60, overlap=50, min_size=40)
    joined = " ".join(c.text for c in chunks)
    for i in range(100):
        assert f"w{i}" in joined
