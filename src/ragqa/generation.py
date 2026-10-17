"""Grounded answer generation with citations and abstention.

This module is the "new layer" on top of retrieval, and it is where an LLM/RAG
project is usually judged. Three things are treated as correctness requirements
rather than nice-to-haves:

1. **Grounding.** The model is instructed to answer *only* from retrieved context
   and to cite every claim. The prompt is built here, not buried in a framework,
   so it can be reviewed line by line in an interview.
2. **Citations that resolve.** A citation marker like ``[S2]`` maps to a concrete
   chunk with a document id, ordinal, and (for PDFs) a page. We parse the markers
   back out of the model's answer and validate them against what was retrieved --
   an answer citing ``[S9]`` when only 5 sources were supplied is a hallucination
   signal, and the evaluator counts it.
3. **Abstention.** The system must refuse when the evidence is insufficient. That
   is enforced twice: a pre-generation *evidence gate* (retrieval confidence below
   threshold -> we never call the model) and a post-generation *refusal check*
   (the model itself emits the abstention sentence).

Offline behaviour: if no LLM is configured or the endpoint is unreachable, the
service returns a clear "no LLM configured/unavailable" answer **plus the
retrieval results**, so the API, CLI, and the retrieval half of the evaluation
harness all keep working with no model present.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .config import Settings
from .retrieval import RetrievalResult
from .types import Answer, Citation, ScoredChunk

log = logging.getLogger(__name__)

# Markers are exactly "[S<digits>]" -- the same form the prompt asks for. Keeping
# the pattern strict makes the parser predictable and avoids accidentally
# matching LaTeX-style "[1]" that models sometimes emit for other reasons.
_CITATION_RE = re.compile(r"\[S(\d+)\]")

# Substrings that we accept as a refusal even if the model paraphrases slightly.
# The exact configured sentence is checked first; these are a safety net for
# smaller local models that do not follow instructions perfectly.
_REFUSAL_HINTS: tuple[str, ...] = (
    "don't have enough information",
    "do not have enough information",
    "not enough information in the provided",
    "insufficient information in the provided",
    "cannot answer this from the provided",
    "can't answer this from the provided",
    "not mentioned in the provided",
)

# A refusal longer than this is treated as an answer that merely quotes a refusal,
# not as a refusal. Chosen to comfortably fit the configured sentence plus a short
# preamble while excluding any genuine multi-sentence answer.
_SHORT_ANSWER_CHARS = 200


class LLMError(Exception):
    """Endpoint responded but the response was unusable / not 2xx."""


class LLMUnavailable(LLMError):
    """We could not reach the endpoint at all (connection/timeout/DNS)."""


class LLMClient(Protocol):
    name: str

    def is_configured(self) -> bool: ...

    def chat(self, messages: list[dict[str, str]]) -> "LLMResponse": ...


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class NullLLM:
    """Used when generation is disabled. Makes the offline path explicit."""

    name = "null"

    def __init__(self, reason: str = "generation disabled") -> None:
        self.reason = reason

    def is_configured(self) -> bool:
        return False

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:  # pragma: no cover
        raise LLMUnavailable(self.reason)


class OpenAICompatibleLLM:
    """Minimal chat-completions client for any OpenAI-compatible HTTP endpoint.

    Works with local Ollama (``http://localhost:11434/v1``), vLLM, llama.cpp's
    server, LM Studio, OpenAI, Azure OpenAI (with a full base_url), Groq, etc.
    Deliberately hand-rolled over httpx instead of pulling in the openai SDK:
    the wire format we need is one POST, and owning it removes a dependency and
    makes the request/response shape inspectable.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 600,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_body = dict(extra_body or {})
        self.name = f"openai-compatible:{model}"
        self._client = client  # injected in tests to avoid real network calls

    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)

    def _http(self):
        if self._client is not None:
            return self._client
        import httpx

        self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """The exact JSON body sent to the endpoint. Pure, so it is testable."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        # Explicit config wins over the passthrough, so a typo in LLM_EXTRA_BODY
        # cannot silently break the model name or the message list.
        for key, value in self.extra_body.items():
            payload.setdefault(key, value)
        return payload

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        import httpx

        url = f"{self.base_url}/chat/completions"
        payload = self.build_payload(messages)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start = time.perf_counter()
        try:
            resp = self._http().post(url, json=payload, headers=headers)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            raise LLMUnavailable(f"cannot reach LLM endpoint {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"LLM request failed: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000.0

        if resp.status_code >= 400:
            body = resp.text[:500]
            raise LLMError(f"LLM endpoint returned HTTP {resp.status_code}: {body}")
        try:
            data = resp.json()
        except Exception as exc:
            raise LLMError(f"LLM endpoint returned non-JSON body: {resp.text[:200]!r}") from exc

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape (no choices[0].message): {data!r}") from exc

        text = message.get("content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise LLMError(f"unexpected content type: {type(text).__name__}")

        # Distinguish "the model said nothing" from "the model spent its whole
        # budget thinking". The latter is a configuration problem, and reporting
        # it as an abstention would turn a broken setup into a fake success.
        reasoning = message.get("reasoning") or message.get("thinking")
        if not text.strip() and reasoning:
            raise LLMError(
                "endpoint returned an empty 'content' but a non-empty reasoning channel: "
                "the model is in thinking mode and consumed its token budget. Set "
                "LLM_REASONING_EFFORT=none (Ollama/hosted), raise LLM_MAX_TOKENS, or use a "
                "non-reasoning instruct model."
            )

        return LLMResponse(
            text=text.strip(),
            model=str(data.get("model", self.model)),
            usage=dict(data.get("usage") or {}),
            latency_ms=latency,
        )


def build_llm(settings: Settings) -> tuple[LLMClient, list[str]]:
    notes: list[str] = []
    if not settings.llm_enabled:
        notes.append("generation: disabled (OFFLINE MODE) -- retrieval and evaluation still run")
        return NullLLM("generation disabled via LLM_ENABLED=false"), notes
    client = OpenAICompatibleLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort,
        extra_body=settings.llm_extra_body,
    )
    notes.append(f"generation: OpenAI-compatible endpoint {settings.llm_base_url} model={settings.llm_model}")
    if settings.llm_reasoning_effort:
        notes.append(f"generation: reasoning_effort={settings.llm_reasoning_effort}")
    return client, notes


# --------------------------------------------------------------------- citations


def build_citations(hits: Sequence[ScoredChunk], titles: dict[str, str], paths: dict[str, str], *, max_chars: int = 320) -> list[Citation]:
    """Number the retrieved chunks S1..Sn and attach provenance for display.

    `snippet` is bounded for display/API payloads; `context` keeps the FULL text
    because that is what the prompt must feed to the model (see Citation.context).
    """
    citations: list[Citation] = []
    for i, sc in enumerate(hits, start=1):
        full = sc.chunk.text.strip()
        snippet = full if len(full) <= max_chars else full[:max_chars].rsplit(" ", 1)[0] + " ..."
        citations.append(
            Citation(
                marker=f"[S{i}]",
                index=i,
                chunk_id=sc.chunk.chunk_id,
                doc_id=sc.chunk.doc_id,
                doc_title=titles.get(sc.chunk.doc_id, sc.chunk.doc_id),
                path=paths.get(sc.chunk.doc_id, ""),
                ordinal=sc.chunk.ordinal,
                page=sc.chunk.page,
                char_start=sc.chunk.char_start,
                char_end=sc.chunk.char_end,
                snippet=snippet,
                score=sc.score,
                context=full,
            )
        )
    return citations


def extract_citation_indices(text: str) -> list[int]:
    """Return the distinct 1-based source indices referenced in `text`, in order."""
    seen: list[int] = []
    for m in _CITATION_RE.finditer(text):
        idx = int(m.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def normalize_for_match(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation-only noise.

    Used for refusal detection so the check is robust to trailing periods,
    smart quotes, and doubled spaces from a model.
    """
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_abstention_text(text: str, phrase: str) -> bool:
    """True if `text` is a refusal rather than an answer.

    The rule is deliberately *length-guarded*: a refusal is a short message. An
    answer that happens to quote the abstention sentence (or the words "not
    enough information") while actually answering must NOT be thrown away --
    treating any occurrence of the phrase as a refusal would silently convert good
    answers into abstentions, which inflates the abstention metric and hides real
    failures.

    Two stages:
      1. exact normalised match -- the model did exactly as instructed;
      2. short text containing the phrase or one of a few known paraphrase stems --
         covers smaller local models that wrap the sentence in a preamble.
    """
    if not text or not text.strip():
        return True  # an empty answer carries no information: treat as a refusal
    norm = normalize_for_match(text)
    target = normalize_for_match(phrase)
    if not target:
        return any(hint in norm for hint in _REFUSAL_HINTS) and len(norm) <= _SHORT_ANSWER_CHARS
    if norm == target:
        return True
    # The model reasoning itself to a refusal and *ending* on the abstention
    # sentence is a refusal regardless of length: nothing follows the conclusion.
    # (A long answer that merely quotes the phrase somewhere inside it does not
    # end with it, so it still counts as an answer -- see the test suite.)
    if norm.endswith(target):
        return True
    # A short answer that contains the phrase or a known paraphrase.
    if len(norm) <= max(len(target) + 120, _SHORT_ANSWER_CHARS) and (
        target in norm or any(hint in norm for hint in _REFUSAL_HINTS)
    ):
        return True
    return False


# -------------------------------------------------------------------- prompting

SYSTEM_PROMPT = """You are a document question-answering assistant working over a fixed corpus of enterprise documents.

Follow these rules exactly:
1. Answer ONLY from the CONTEXT passages provided in the user message. Never use outside knowledge.
2. Cite the sources you use with their bracketed ids, e.g. [S1] or [S2][S3]. Every factual sentence must carry at least one citation.
3. If the CONTEXT does not contain enough information to answer the question, reply with EXACTLY this sentence and nothing else:
   "{abstain_phrase}"
4. Never invent document titles, numbers, dates, or names. If a value is not in the CONTEXT, say it is not stated.
5. When two sources conflict, point out the conflict and cite both.
6. Be concise and direct. Prefer 1-4 sentences unless the question asks for a list."""


def build_messages(
    question: str,
    citations: Sequence[Citation],
    *,
    abstain_phrase: str,
) -> list[dict[str, str]]:
    """Assemble the chat messages. Kept as a pure function so it is unit-testable.

    Context uses each source's FULL text (`c.context`), not its display snippet:
    truncating here would hide evidence that sits past the cut and produce an
    answer that looks like "the document doesn't say", which is the worst kind of
    wrong -- confident and unfalsifiable to the user.
    """
    blocks: list[str] = []
    for c in citations:
        page = f"page {c.page}" if c.page is not None else "no page"
        blocks.append(
            f"[S{c.index}] (document: {c.doc_title} | path: {c.path} | {page} | chunk {c.ordinal})\n"
            f"{c.context or c.snippet}"
        )
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    system = SYSTEM_PROMPT.format(abstain_phrase=abstain_phrase)
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer using only the CONTEXT above, with citations:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ------------------------------------------------------------------- generation


class Generator:
    """Turns a question + retrieval result into a grounded :class:`Answer`."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def answer(self, question: str, retrieval: RetrievalResult) -> Answer:
        s = self.settings
        start = time.perf_counter()
        hits = retrieval.hits
        all_citations = build_citations(hits, retrieval.titles, retrieval.paths)
        diagnostics = retrieval.diagnostics

        def finish(answer: Answer) -> Answer:
            answer.latency_ms = (time.perf_counter() - start) * 1000.0
            return answer

        # ---- gate 1: nothing retrieved -------------------------------------
        if not hits:
            return finish(
                Answer(
                    question=question,
                    status="abstained_insufficient_evidence",
                    text="",
                    citations=[],
                    retrieved=[],
                    diagnostics=diagnostics,
                    reason="no chunks matched the query",
                )
            )

        # ---- gate 2: evidence too weak --------------------------------------
        conf = diagnostics.evidence_confidence
        if s.min_evidence_confidence > 0 and conf < s.min_evidence_confidence:
            return finish(
                Answer(
                    question=question,
                    status="abstained_insufficient_evidence",
                    text="",
                    citations=all_citations,
                    retrieved=hits,
                    diagnostics=diagnostics,
                    reason=(
                        f"evidence confidence {conf:.3f} below threshold "
                        f"{s.min_evidence_confidence:.3f}; LLM not called"
                    ),
                )
            )

        # ---- gate 3: no LLM --------------------------------------------------
        if not self.llm.is_configured():
            return finish(
                Answer(
                    question=question,
                    status="abstained_no_llm",
                    text="",
                    citations=all_citations,
                    retrieved=hits,
                    diagnostics=diagnostics,
                    reason=(
                        "no LLM configured (set OPENAI_BASE_URL / LLM_MODEL, or run "
                        "`ollama serve` and `ollama pull " + s.llm_model + "`). "
                        "Retrieval results are still returned."
                    ),
                )
            )

        # ---- generate --------------------------------------------------------
        messages = build_messages(question, all_citations, abstain_phrase=s.abstain_phrase)
        try:
            response = self.llm.chat(messages)
        except LLMUnavailable as exc:
            return finish(
                Answer(
                    question=question,
                    status="abstained_llm_unavailable",
                    text="",
                    citations=all_citations,
                    retrieved=hits,
                    diagnostics=diagnostics,
                    reason=str(exc),
                )
            )
        except LLMError as exc:
            return finish(
                Answer(
                    question=question,
                    status="error",
                    text="",
                    citations=all_citations,
                    retrieved=hits,
                    diagnostics=diagnostics,
                    reason=str(exc),
                )
            )

        # ---- post-check: model refused --------------------------------------
        if is_abstention_text(response.text, s.abstain_phrase):
            return finish(
                Answer(
                    question=question,
                    status="abstained_model_refused",
                    text=response.text,
                    citations=all_citations,
                    retrieved=hits,
                    diagnostics=diagnostics,
                    model=response.model,
                    reason="model emitted the abstention sentence",
                    usage=response.usage,
                )
            )

        # ---- post-check: citations resolve ----------------------------------
        referenced = extract_citation_indices(response.text)
        valid = [all_citations[i - 1] for i in referenced if 0 < i <= len(all_citations)]
        invalid = [i for i in referenced if not (0 < i <= len(all_citations))]
        if invalid:
            log.warning("model cited out-of-range sources %s (only %d supplied)", invalid, len(all_citations))

        usage = dict(response.usage)
        # Surfaced for the evaluator: an answer with no resolving citation is not
        # grounded, even if it reads well.
        usage["citation_indices"] = referenced
        usage["invalid_citation_indices"] = invalid
        usage["uncited"] = len(valid) == 0

        return finish(
            Answer(
                question=question,
                status="answered",
                text=response.text,
                citations=valid,
                retrieved=hits,
                diagnostics=diagnostics,
                model=response.model,
                reason=None if valid else "answer contains no resolvable citations (ungrounded)",
                usage=usage,
            )
        )
