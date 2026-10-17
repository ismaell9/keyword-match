"""Abstention logic: the gates that decide whether the system refuses.

These tests pin down the behaviour that makes the project honest:
  * nothing retrieved            -> abstain without calling the model
  * evidence too weak            -> abstain without calling the model
  * no LLM configured            -> report that clearly, still return retrieval
  * model refuses                -> surface as a distinct status
  * model answers with citations -> answered, citations resolved
  * model cites a source that does not exist -> caught and recorded
"""

from __future__ import annotations

import pytest

from ragqa.config import Settings
from ragqa.generation import (
    Generator,
    LLMError,
    LLMResponse,
    LLMUnavailable,
    NullLLM,
)
from ragqa.retrieval import RetrievalResult
from ragqa.types import Chunk, Retrieved, ScoredChunk

ABSTAIN = "I don't have enough information in the provided documents to answer that."


class FakeLLM:
    """Records calls and returns a scripted response (or raises)."""

    name = "fake"

    def __init__(self, response: str | None = None, exc: Exception | None = None, configured: bool = True):
        self.response = response
        self.exc = exc
        self.configured = configured
        self.calls: list[list[dict[str, str]]] = []

    def is_configured(self) -> bool:
        return self.configured

    def chat(self, messages):
        self.calls.append(messages)
        if self.exc:
            raise self.exc
        return LLMResponse(text=self.response or "", model="fake-model", usage={"total_tokens": 5})


def _hit(chunk_id: int = 1, text: str = "Passwords must be rotated every 90 days.", score: float = 0.9) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="docA",
        ordinal=0,
        text=text,
        page=None,
        char_start=0,
        char_end=len(text),
        n_words=len(text.split()),
    )
    return ScoredChunk(chunk=chunk, score=score, source="fused", ranks={"bm25": 1})


def _result(hits, confidence: float) -> RetrievalResult:
    diag = Retrieved(
        candidates_bm25=len(hits),
        candidates_vector=len(hits),
        fused=len(hits),
        reranked=False,
        vector_backend="numpy-brute-force",
        embed_backend="hash",
        evidence_confidence=confidence,
    )
    return RetrievalResult(
        hits=list(hits),
        diagnostics=diag,
        titles={"docA": "Security Policy"},
        paths={"docA": "security.md"},
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_enabled=True, min_evidence_confidence=0.15, abstain_phrase=ABSTAIN)


def test_no_hits_abstains_without_calling_the_model(settings):
    llm = FakeLLM(response="should never be used")
    answer = Generator(llm, settings).answer("anything", _result([], 0.0))
    assert answer.status == "abstained_insufficient_evidence"
    assert answer.abstained
    assert llm.calls == []  # the whole point: no model call


def test_low_evidence_confidence_abstains_without_calling_the_model(settings):
    llm = FakeLLM(response="should never be used")
    answer = Generator(llm, settings).answer("unrelated question", _result([_hit()], 0.05))
    assert answer.status == "abstained_insufficient_evidence"
    assert llm.calls == []
    assert "below threshold" in (answer.reason or "")
    # Retrieval is still returned so the caller can see what was found.
    assert len(answer.retrieved) == 1


def test_confidence_at_threshold_is_not_abstained(settings):
    llm = FakeLLM(response="Rotation is every 90 days [S1].")
    answer = Generator(llm, settings).answer("how often?", _result([_hit()], 0.15))
    assert answer.status == "answered"
    assert len(llm.calls) == 1


def test_zero_threshold_disables_the_gate(settings):
    strict = settings.model_copy(update={"min_evidence_confidence": 0.0})
    llm = FakeLLM(response="Rotation is every 90 days [S1].")
    answer = Generator(llm, strict).answer("how often?", _result([_hit()], 0.001))
    assert answer.status == "answered"


def test_no_llm_configured_reports_clearly_and_keeps_retrieval(settings):
    answer = Generator(NullLLM("disabled"), settings).answer("how often?", _result([_hit()], 0.9))
    assert answer.status == "abstained_no_llm"
    assert answer.abstained
    assert len(answer.retrieved) == 1
    assert len(answer.citations) == 1
    assert "no LLM configured" in (answer.reason or "")


def test_model_refusal_is_a_distinct_status(settings):
    llm = FakeLLM(response=ABSTAIN)
    answer = Generator(llm, settings).answer("unanswerable", _result([_hit(score=0.5)], 0.5))
    assert answer.status == "abstained_model_refused"
    assert answer.text == ABSTAIN


def test_model_refusal_paraphrase_is_detected(settings):
    llm = FakeLLM(response="That is not mentioned in the provided documents.")
    answer = Generator(llm, settings).answer("unanswerable", _result([_hit()], 0.5))
    assert answer.status == "abstained_model_refused"


def test_answered_with_valid_citation(settings):
    llm = FakeLLM(response="Passwords must be rotated every 90 days [S1].")
    answer = Generator(llm, settings).answer("how often?", _result([_hit()], 0.9))
    assert answer.status == "answered"
    assert len(answer.citations) == 1
    assert answer.citations[0].marker == "[S1]"
    assert answer.citations[0].doc_title == "Security Policy"
    assert answer.usage["uncited"] is False
    assert answer.usage["citation_indices"] == [1]
    assert answer.model == "fake-model"


def test_answer_citing_nonexistent_source_is_flagged(settings):
    llm = FakeLLM(response="The policy says something [S9].")
    answer = Generator(llm, settings).answer("q", _result([_hit()], 0.9))
    assert answer.status == "answered"
    assert answer.citations == []  # no resolvable citation
    assert answer.usage["invalid_citation_indices"] == [9]
    assert answer.usage["uncited"] is True
    assert "ungrounded" in (answer.reason or "")


def test_partially_valid_citations_keep_only_the_valid_ones(settings):
    hits = [_hit(1, "first", 0.9), _hit(2, "second", 0.8)]
    llm = FakeLLM(response="Claim one [S1] and claim two [S7].")
    answer = Generator(llm, settings).answer("q", _result(hits, 0.9))
    assert [c.index for c in answer.citations] == [1]
    assert answer.usage["invalid_citation_indices"] == [7]


def test_unreachable_endpoint_is_reported_not_crashed(settings):
    llm = FakeLLM(exc=LLMUnavailable("cannot reach LLM endpoint"))
    answer = Generator(llm, settings).answer("q", _result([_hit()], 0.9))
    assert answer.status == "abstained_llm_unavailable"
    assert len(answer.retrieved) == 1


def test_endpoint_error_is_reported_as_error_status(settings):
    llm = FakeLLM(exc=LLMError("HTTP 500"))
    answer = Generator(llm, settings).answer("q", _result([_hit()], 0.9))
    assert answer.status == "error"
    assert "HTTP 500" in (answer.reason or "")


def test_prompt_sent_to_model_contains_only_retrieved_sources(settings):
    llm = FakeLLM(response="ok [S1]")
    Generator(llm, settings).answer("q", _result([_hit(1, "alpha")], 0.9))
    user = llm.calls[0][1]["content"]
    assert "alpha" in user
    assert "[S1]" in user
    assert "[S2]" not in user
