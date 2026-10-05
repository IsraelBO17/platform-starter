# tests/test_platform.py
import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

os.environ["USE_MEMORY_HISTORY"] = "1"
os.environ["GROQ_API_KEY"] = "test"

from app.main import app  # noqa: E402
from app.providers.base import CallRequest, CallResult, Message, ProviderError, Usage  # noqa: E402
from app.providers.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from app.services import retry as retry_mod  # noqa: E402
from app.services.tokens import count_message_tokens, fit_to_budget  # noqa: E402


class FakeProvider:
    name = "fake"

    def __init__(self, replies, fail_first=0):
        self.replies = list(replies)
        self.fail_first = fail_first
        self.calls: list[CallRequest] = []

    async def chat(self, request):
        self.calls.append(request)
        if self.fail_first:
            self.fail_first -= 1
            raise ProviderError("429 rate limited", retryable=True)
        return CallResult(text=self.replies.pop(0), model="openai/gpt-oss-20b",
                          usage=Usage(120, 30), latency_ms=42)

    async def stream(self, request):
        for piece in ["Hel", "lo", "!"]:
            yield piece


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_):
        return None
    monkeypatch.setattr(retry_mod.asyncio, "sleep", instant)


def client_with(provider):
    app.state.provider = provider
    if hasattr(app.state, "history"):
        del app.state.history
    return TestClient(app)


def test_stateless_chat_retries_then_succeeds():
    fake = FakeProvider(["Hi there"], fail_first=2)
    with client_with(fake) as c:
        r = c.post("/v1/chat", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "Hi there" and body["input_tokens"] == 120
    assert body["cost_usd"] > 0
    assert len(fake.calls) == 3
    assert fake.calls[0].messages[0].role == "system"


def test_stateful_conversation_remembers():
    fake = FakeProvider(["First answer", "Second answer"])
    with client_with(fake) as c:
        c.post("/v1/conversations/abc/messages", json={"message": "My name is Ada"})
        r = c.post("/v1/conversations/abc/messages", json={"message": "What is my name?"})
    assert r.json()["reply"] == "Second answer"
    sent = [m.content for m in fake.calls[1].messages]
    assert "My name is Ada" in sent and "First answer" in sent


def test_triage_repairs_invalid_output():
    bad = '{"category": "payments", "urgency": "high", "summary": "x"}'
    good = '```json\n{"category": "billing", "urgency": "high", "summary": "Double debit"}\n```'
    fake = FakeProvider([bad, good])
    with client_with(fake) as c:
        r = c.post("/v1/triage", json={"text": "I was charged twice"})
    assert r.status_code == 200, r.text
    assert r.json() == {"category": "billing", "urgency": "high", "summary": "Double debit"}
    assert fake.calls[0].json_schema is not None


def test_triage_rejects_after_failed_repair():
    fake = FakeProvider(["not json", "still not json"])
    with client_with(fake) as c:
        r = c.post("/v1/triage", json={"text": "help"})
    assert r.status_code == 422


def test_sse_stream():
    with client_with(FakeProvider([])) as c:
        r = c.post("/v1/chat/stream", json={"message": "hi"})
    assert r.headers["content-type"].startswith("text/event-stream")
    deltas = [json.loads(line[6:])["delta"] for line in r.text.splitlines()
              if line.startswith("data: {\"delta")]
    assert "".join(deltas) == "Hello!"
    assert "event: done" in r.text


def test_budget_keeps_system_and_newest():
    msgs = [Message("system", "sys")] + [Message("user", f"turn {i} " + "word " * 50) for i in range(20)]
    kept = fit_to_budget(msgs, 300)
    assert kept[0].role == "system" and kept[-1].content.startswith("turn 19")
    assert count_message_tokens(kept) <= 300


@pytest.mark.anyio
async def test_openai_compatible_parses_real_response_shape():
    def handler(request: httpx.Request):
        sent = json.loads(request.content)
        assert sent["response_format"]["type"] == "json_schema"
        assert request.url.host == "api.groq.com"
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{\"a\": 1}"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        })

    p = OpenAICompatibleProvider("groq", "k", 10)
    p.client = p.client.with_options(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    res = await p.chat(CallRequest([Message("user", "hi")], "openai/gpt-oss-20b", json_schema={"type": "object"}))
    assert res.text == '{"a": 1}' and res.usage.input_tokens == 11


@pytest.mark.anyio
async def test_openai_compatible_maps_429_to_retryable():
    p = OpenAICompatibleProvider("groq", "k", 10)
    p.client = p.client.with_options(http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(429, json={"error": {"message": "slow down"}}))))
    with pytest.raises(ProviderError) as info:
        await p.chat(CallRequest([Message("user", "hi")], "m"))
    assert info.value.retryable


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_api_key_required_when_configured():
    fake = FakeProvider(["ok", "ok"])
    with client_with(fake) as c:
        app.state.settings = app.state.settings.model_copy(update={"app_api_key": "s3cret"})
        assert c.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 401
        r = c.post("/v1/chat", headers={"X-API-Key": "s3cret"},
                   json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
