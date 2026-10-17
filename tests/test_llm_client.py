"""Tests for the OpenAI-compatible client's request building and error handling.

The HTTP layer is never touched: a fake transport is injected, so these run
offline and deterministically.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ragqa.generation import (
    LLMError,
    LLMUnavailable,
    OpenAICompatibleLLM,
)


class FakeTransport(httpx.BaseTransport):
    """Returns a scripted response and records the request that was sent."""

    def __init__(self, status: int = 200, payload: dict | None = None, raw: str | None = None):
        self.status = status
        self.payload = payload
        self.raw = raw
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raw is not None:
            return httpx.Response(self.status, text=self.raw)
        return httpx.Response(self.status, json=self.payload or {})


def _client(transport: FakeTransport, **kwargs) -> OpenAICompatibleLLM:
    http = httpx.Client(transport=transport)
    return OpenAICompatibleLLM(
        base_url="http://localhost:11434/v1",
        api_key="not-needed",
        model="test-model",
        client=http,
        **kwargs,
    )


def _ok(content: str = "answer [S1]") -> FakeTransport:
    return FakeTransport(payload={"model": "test-model", "choices": [{"message": {"content": content}}], "usage": {"total_tokens": 3}})


def test_successful_completion_is_parsed():
    llm = _client(_ok("The answer is 90 days [S1]."))
    resp = llm.chat([{"role": "user", "content": "hi"}])
    assert resp.text == "The answer is 90 days [S1]."
    assert resp.model == "test-model"
    assert resp.usage == {"total_tokens": 3}
    assert resp.latency_ms >= 0.0


def test_payload_defaults_have_no_reasoning_effort():
    payload = _client(_ok()).build_payload([{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in payload
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["model"] == "test-model"


def test_reasoning_effort_is_sent_when_configured():
    llm = _client(_ok(), reasoning_effort="none")
    payload = llm.build_payload([{"role": "user", "content": "x"}])
    assert payload["reasoning_effort"] == "none"


def test_extra_body_is_merged_but_cannot_override_core_fields():
    llm = _client(_ok(), extra_body={"top_p": 0.9, "model": "hijacked", "stream": True})
    payload = llm.build_payload([{"role": "user", "content": "x"}])
    assert payload["top_p"] == 0.9
    # Core fields must win, or a stray extra-body key could break the request.
    assert payload["model"] == "test-model"
    assert payload["stream"] is False


def test_messages_and_auth_header_are_sent():
    transport = _ok()
    llm = _client(transport)
    llm.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    req = transport.requests[0]
    assert str(req.url) == "http://localhost:11434/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer not-needed"
    body = json.loads(req.content)
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_http_error_status_raises_llm_error():
    llm = _client(FakeTransport(status=500, raw="boom"))
    with pytest.raises(LLMError, match="HTTP 500"):
        llm.chat([{"role": "user", "content": "x"}])


def test_non_json_body_raises_llm_error():
    llm = _client(FakeTransport(status=200, raw="<html>not json</html>"))
    with pytest.raises(LLMError, match="non-JSON"):
        llm.chat([{"role": "user", "content": "x"}])


def test_missing_choices_raises_llm_error():
    llm = _client(FakeTransport(payload={"oops": True}))
    with pytest.raises(LLMError, match="unexpected response shape"):
        llm.chat([{"role": "user", "content": "x"}])


def test_thinking_model_with_empty_content_is_an_error_not_an_abstention():
    """A reasoning-only response is a misconfiguration; reporting it as a refusal
    would hide a broken setup behind a plausible-looking abstention."""
    transport = FakeTransport(
        payload={"choices": [{"message": {"content": "", "reasoning": "Thinking Process: ..."}}]}
    )
    llm = _client(transport)
    with pytest.raises(LLMError, match="thinking mode"):
        llm.chat([{"role": "user", "content": "x"}])


def test_genuinely_empty_answer_is_not_an_error():
    """Empty content with no reasoning channel is a (bad) answer, not a bug."""
    llm = _client(_ok(""))
    assert llm.chat([{"role": "user", "content": "x"}]).text == ""


def test_connection_error_raises_llm_unavailable():
    class Exploding(httpx.BaseTransport):
        def handle_request(self, request):
            raise httpx.ConnectError("connection refused")

    llm = _client(Exploding())
    with pytest.raises(LLMUnavailable, match="cannot reach LLM endpoint"):
        llm.chat([{"role": "user", "content": "x"}])


def test_is_configured_requires_url_and_model():
    http = httpx.Client(transport=_ok())
    assert OpenAICompatibleLLM(base_url="http://x/v1", api_key="k", model="m", client=http).is_configured()
    assert not OpenAICompatibleLLM(base_url="", api_key="k", model="m", client=http).is_configured()
    assert not OpenAICompatibleLLM(base_url="http://x/v1", api_key="k", model="", client=http).is_configured()
