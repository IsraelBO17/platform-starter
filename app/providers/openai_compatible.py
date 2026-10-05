# app/providers/openai_compatible.py
import time
from collections.abc import AsyncIterator

import openai
from openai import AsyncOpenAI

from app.providers.base import CallRequest, CallResult, ProviderError, Usage

# Many providers accept the OpenAI request shape at their own address.
BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": None,  # the SDK's default
    "xai": "https://api.x.ai/v1",
    "ollama": "http://localhost:11434/v1",
}


class OpenAICompatibleProvider:
    def __init__(self, name: str, api_key: str | None, timeout: float):
        if name not in BASE_URLS:
            raise ValueError(f"Unknown OpenAI-compatible provider: {name}")
        self.name = name
        # max_retries=0: we own retry policy ourselves (app/services/retry.py).
        self.client = AsyncOpenAI(
            api_key=api_key or "not-needed-for-ollama",
            base_url=BASE_URLS[name],
            timeout=timeout,
            max_retries=0,
        )

    def _payload(self, request: CallRequest) -> dict:
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        # OpenAI's newer models take max_completion_tokens; the others still use max_tokens.
        payload["max_completion_tokens" if self.name == "openai" else "max_tokens"] = request.max_tokens
        # OpenAI's reasoning models accept only their default sampling settings.
        if self.name == "openai" and request.model.startswith(("o1", "o3", "o4", "gpt-5")):
            del payload["temperature"], payload["top_p"]
        if request.stop:
            payload["stop"] = request.stop
        if request.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": request.json_schema},
            }
        return payload

    async def chat(self, request: CallRequest) -> CallResult:
        started = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(**self._payload(request))
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise ProviderError(str(exc), retryable=True) from exc
        except openai.APIStatusError as exc:
            raise ProviderError(str(exc), retryable=exc.status_code >= 500) from exc

        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return CallResult(
            text=choice.message.content or "",
            model=response.model,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                cached_input_tokens=cached,
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=choice.finish_reason,
        )

    async def stream(self, request: CallRequest) -> AsyncIterator[str]:
        try:
            chunks = await self.client.chat.completions.create(**self._payload(request), stream=True)
            async for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.APIError as exc:
            raise ProviderError(str(exc), retryable=False) from exc
