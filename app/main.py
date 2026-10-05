# app/main.py
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.providers.base import CallRequest, Message, ProviderError
from app.providers.factory import build_provider
from app.schemas import Triage
from app.security import require_api_key
from app.services.history import InMemoryHistory, PostgresHistory
from app.services.pricing import record_usage
from app.services.prompts import build_triage_messages
from app.services.retry import chat_with_retry
from app.services.structured import StructuredOutputError, generate_structured
from app.services.tokens import fit_to_budget

logging.basicConfig(level=logging.INFO)
SYSTEM_PROMPT = "You are a concise, accurate assistant for the Modular AI Integration Platform."
HISTORY_TOKEN_BUDGET = 6_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    if not hasattr(app.state, "provider"):          # tests inject a fake provider
        app.state.provider = build_provider(settings)
    if not hasattr(app.state, "history"):
        if os.getenv("USE_MEMORY_HISTORY") == "1":
            app.state.history = InMemoryHistory()
        else:
            from psycopg_pool import AsyncConnectionPool

            # Small pool: on Lambda each instance handles one request at a time.
            pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=2, open=False,
                                        check=AsyncConnectionPool.check_connection)
            await pool.open()
            app.state.history = PostgresHistory(pool)
            await app.state.history.create_schema()
    yield


app = FastAPI(title="Modular AI Integration Platform", lifespan=lifespan,
              dependencies=[Depends(require_api_key)])


# ---------- request and response models ----------

class ChatIn(BaseModel):
    messages: list[dict] = Field(..., description="Full history, sent by the client each time")
    temperature: float = Field(0.2, ge=0, le=2)
    max_tokens: int = Field(512, ge=1, le=4096)


class ChatOut(BaseModel):
    reply: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class TurnIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=8_000)


class TriageIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=4_000)


# ---------- endpoints ----------

@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn, request: Request):
    """Stateless: the client owns the history and sends it every time."""
    s = request.app.state.settings
    messages = [Message("system", SYSTEM_PROMPT)] + [
        Message(m["role"], m["content"]) for m in body.messages
    ]
    req = CallRequest(messages=messages, model=s.llm_model,
                      temperature=body.temperature, max_tokens=body.max_tokens)
    try:
        result = await chat_with_retry(request.app.state.provider, req, s.llm_max_retries)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=f"Model provider error: {exc}") from exc
    cost = record_usage(result, feature="chat")
    return ChatOut(reply=result.text, input_tokens=result.usage.input_tokens,
                   output_tokens=result.usage.output_tokens, cost_usd=cost,
                   latency_ms=result.latency_ms)


@app.post("/v1/conversations/{conversation_id}/messages", response_model=ChatOut)
async def converse(conversation_id: str, body: TurnIn, request: Request):
    """Stateful: the server stores the history in PostgreSQL."""
    s = request.app.state.settings
    history = request.app.state.history
    past = await history.load(conversation_id)
    user_msg = Message("user", body.message)
    messages = fit_to_budget([Message("system", SYSTEM_PROMPT), *past, user_msg],
                             HISTORY_TOKEN_BUDGET)
    req = CallRequest(messages=messages, model=s.llm_model)
    try:
        result = await chat_with_retry(request.app.state.provider, req, s.llm_max_retries)
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=f"Model provider error: {exc}") from exc
    await history.append(conversation_id, user_msg)
    await history.append(conversation_id, Message("assistant", result.text))
    cost = record_usage(result, feature="conversation")
    return ChatOut(reply=result.text, input_tokens=result.usage.input_tokens,
                   output_tokens=result.usage.output_tokens, cost_usd=cost,
                   latency_ms=result.latency_ms)


@app.post("/v1/chat/stream")
async def chat_stream(body: TurnIn, request: Request):
    """Server-Sent Events: tokens reach the client as they are generated."""
    s = request.app.state.settings
    req = CallRequest(
        messages=[Message("system", SYSTEM_PROMPT), Message("user", body.message)],
        model=s.llm_model,
    )

    async def events():
        try:
            async with asyncio.timeout(s.llm_timeout_seconds * 2):
                async for piece in request.app.state.provider.stream(req):
                    if await request.is_disconnected():
                        logging.info("Client disconnected; stopping generation")
                        return
                    yield f"data: {json.dumps({'delta': piece})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except TimeoutError:
            yield f"event: error\ndata: {json.dumps({'error': 'timeout'})}\n\n"
        except ProviderError as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/v1/triage", response_model=Triage)
async def triage(body: TriageIn, request: Request):
    """Structured output: free text in, a validated object out."""
    s = request.app.state.settings
    req = CallRequest(
        messages=build_triage_messages("Univaciti Bank", ["billing", "account", "technical", "other"],
                                       body.text),
        model=s.llm_model,
        temperature=0,
        max_tokens=800,   # reasoning models spend part of this budget thinking
    )
    try:
        return await generate_structured(request.app.state.provider, req, Triage)
    except StructuredOutputError as exc:
        raise HTTPException(status_code=422, detail="Model output failed validation") from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=f"Model provider error: {exc}") from exc
