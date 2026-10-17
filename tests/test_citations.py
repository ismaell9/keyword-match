"""Citation formatting, marker parsing, and prompt grounding."""

from __future__ import annotations

from ragqa.generation import (
    SYSTEM_PROMPT,
    build_citations,
    build_messages,
    extract_citation_indices,
    is_abstention_text,
    normalize_for_match,
)
from ragqa.types import Chunk, ScoredChunk

ABSTAIN = "I don't have enough information in the provided documents to answer that."


def _hit(chunk_id: int, text: str, page: int | None = None, doc_id: str = "docA") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=chunk_id,
        text=text,
        page=page,
        char_start=0,
        char_end=len(text),
        n_words=len(text.split()),
    )
    return ScoredChunk(chunk=chunk, score=1.0 - chunk_id / 100.0, source="fused", ranks={"bm25": chunk_id})


def test_markers_are_sequential_from_s1():
    hits = [_hit(101, "alpha"), _hit(102, "beta"), _hit(103, "gamma")]
    citations = build_citations(hits, {"docA": "Doc A"}, {"docA": "a.md"})
    assert [c.marker for c in citations] == ["[S1]", "[S2]", "[S3]"]
    assert [c.index for c in citations] == [1, 2, 3]


def test_citation_carries_provenance_and_offsets():
    citations = build_citations([_hit(7, "body text", page=3, doc_id="docZ")], {"docZ": "Z"}, {"docZ": "z.pdf"})
    c = citations[0]
    assert c.chunk_id == 7
    assert c.doc_id == "docZ"
    assert c.doc_title == "Z"
    assert c.path == "z.pdf"
    assert c.page == 3
    assert c.char_end == len("body text")


def test_snippet_is_truncated_without_cutting_a_word():
    long_text = "word " * 200
    c = build_citations([_hit(1, long_text)], {}, {}, max_chars=50)[0]
    assert len(c.snippet) <= 60
    assert c.snippet.endswith("...")


def test_unknown_doc_id_falls_back_to_the_id_itself():
    c = build_citations([_hit(1, "x", doc_id="missing")], {}, {})[0]
    assert c.doc_title == "missing"
    assert c.path == ""


def test_extract_citation_indices_is_ordered_and_deduplicated():
    assert extract_citation_indices("A [S1] then [S3] and again [S1].") == [1, 3]


def test_extract_handles_multi_digit_and_ignores_other_brackets():
    assert extract_citation_indices("[S12] [1] [S2]") == [12, 2]


def test_extract_returns_empty_when_uncited():
    assert extract_citation_indices("This answer has no citations at all.") == []


def test_prompt_contains_every_source_and_its_marker():
    hits = [_hit(1, "first fact", page=1, doc_id="d1"), _hit(2, "second fact", page=2, doc_id="d2")]
    citations = build_citations(hits, {"d1": "Doc One", "d2": "Doc Two"}, {"d1": "one.md", "d2": "two.pdf"})
    messages = build_messages("what is the fact?", citations, abstain_phrase=ABSTAIN)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    for citation in citations:
        assert citation.marker in user
        assert citation.snippet in user
    assert "Doc One" in user and "one.md" in user
    assert "page 2" in user
    assert "what is the fact?" in user


def test_prompt_instructs_grounding_and_uses_the_configured_abstain_phrase():
    messages = build_messages("q", build_citations([_hit(1, "x")], {}, {}), abstain_phrase=ABSTAIN)
    system = messages[0]["content"]
    assert ABSTAIN in system
    assert "ONLY" in system
    assert "cite" in system.lower()
    # The template must have been formatted, not left with a placeholder.
    assert "{abstain_phrase}" not in system


def test_prompt_handles_zero_sources_explicitly():
    messages = build_messages("q", [], abstain_phrase=ABSTAIN)
    assert "(no context retrieved)" in messages[1]["content"]


def test_system_prompt_template_only_has_the_abstain_placeholder():
    import string

    fields = {name for _, name, _, _ in string.Formatter().parse(SYSTEM_PROMPT) if name}
    assert fields == {"abstain_phrase"}


def test_normalize_for_match_folds_quotes_and_punctuation():
    assert normalize_for_match("I  don\u2019t have enough info!") == "i don't have enough info"


def test_abstention_detected_for_exact_phrase():
    assert is_abstention_text(ABSTAIN, ABSTAIN)


def test_abstention_detected_with_trailing_punctuation_and_case():
    assert is_abstention_text("I DON'T HAVE ENOUGH INFORMATION IN THE PROVIDED DOCUMENTS TO ANSWER THAT!!", ABSTAIN)


def test_abstention_detected_for_paraphrase_variants():
    assert is_abstention_text("I don't have enough information to answer that.", ABSTAIN)
    assert is_abstention_text("That is not mentioned in the provided documents.", ABSTAIN)


def test_abstention_detected_for_empty_answer():
    assert is_abstention_text("", ABSTAIN)
    assert is_abstention_text("   \n ", ABSTAIN)


def test_real_answer_is_not_treated_as_abstention():
    assert not is_abstention_text("Passwords must be rotated every 90 days [S1].", ABSTAIN)


def test_long_answer_containing_the_phrase_is_not_a_refusal():
    """A genuine answer that happens to quote the phrase should not be discarded."""
    answer = (
        "The policy states that support staff must reply with the sentence "
        "'I don't have enough information in the provided documents to answer that.' "
        "when a request falls outside scope [S1]. Separate escalation rules apply [S2]."
    )
    assert not is_abstention_text(answer, ABSTAIN)


def test_long_answer_that_ends_on_the_abstain_sentence_is_a_refusal():
    """Small models often reason for several sentences then conclude with the
    refusal. The conclusion is what counts."""
    answer = (
        "Looking at the retrieved sources, S1 covers scope and S3 covers onboarding. "
        "Neither addresses the rotation frequency. "
        "I don't have enough information in the provided documents to answer that."
    )
    assert is_abstention_text(answer, ABSTAIN)


def test_prompt_uses_the_full_chunk_text_not_the_display_snippet():
    """Regression: feeding the model a truncated snippet hides evidence that sits
    past the cut, which is indistinguishable from the model hallucinating."""
    long_text = "intro paragraph. " + "filler words. " * 60 + "The password rotation interval is 90 days."
    citations = build_citations([_hit(1, long_text)], {}, {}, max_chars=100)
    assert citations[0].snippet != long_text
    assert citations[0].context == long_text
    messages = build_messages("how often?", citations, abstain_phrase=ABSTAIN)
    assert "The password rotation interval is 90 days." in messages[1]["content"]


def test_citation_context_is_carried_from_the_chunk():
    c = build_citations([_hit(1, "full text here")], {}, {})[0]
    assert c.context == "full text here"
    assert c.snippet == "full text here"
