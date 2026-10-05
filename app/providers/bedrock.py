# app/providers/bedrock.py
import asyncio
import time
from collections.abc import AsyncIterator

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.providers.base import CallRequest, CallResult, ProviderError, Usage

RETRYABLE_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException"}


class BedrockProvider:
    """Calls any chat model on Amazon Bedrock through the Converse API."""

    name = "bedrock"

    def __init__(self, region: str, timeout: float = 30.0):
        # Turn off boto3's own retries so retry policy lives in one place (retry.py),
        # and give every call a read timeout.
        config = Config(retries={"max_attempts": 1, "mode": "standard"},
                        read_timeout=timeout, connect_timeout=5)
        self.client = boto3.client("bedrock-runtime", region_name=region, config=config)

    def _converse_args(self, request: CallRequest) -> dict:
        system = [{"text": m.content} for m in request.messages if m.role == "system"]
        messages = [
            {"role": m.role, "content": [{"text": m.content}]}
            for m in request.messages
            if m.role != "system"
        ]
        config = {
            "maxTokens": request.max_tokens,
            "temperature": request.temperature,
            "topP": request.top_p,
        }
        if request.stop:
            config["stopSequences"] = request.stop
        args = {"modelId": request.model, "messages": messages, "inferenceConfig": config}
        if system:
            args["system"] = system
        return args

    @staticmethod
    def _error(exc: Exception) -> ProviderError:
        if isinstance(exc, ClientError):
            code = exc.response["Error"]["Code"]
            return ProviderError(f"{code}: {exc}", retryable=code in RETRYABLE_CODES)
        return ProviderError(str(exc), retryable=True)  # timeouts, dropped connections

    async def chat(self, request: CallRequest) -> CallResult:
        started = time.perf_counter()
        try:
            # boto3 is synchronous; run it in a worker thread so the event loop stays free.
            response = await asyncio.to_thread(self.client.converse, **self._converse_args(request))
        except (ClientError, BotoCoreError) as exc:
            raise self._error(exc) from exc

        blocks = response["output"]["message"]["content"]
        text = "".join(block.get("text", "") for block in blocks)
        usage = response.get("usage", {})
        cached = usage.get("cacheReadInputTokens", 0)
        return CallResult(
            text=text,
            model=request.model,
            usage=Usage(
                # Converse reports cached tokens separately from inputTokens; add them back.
                input_tokens=usage.get("inputTokens", 0) + cached + usage.get("cacheWriteInputTokens", 0),
                output_tokens=usage.get("outputTokens", 0),
                cached_input_tokens=cached,
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=response.get("stopReason"),
        )

    async def stream(self, request: CallRequest) -> AsyncIterator[str]:
        try:
            response = await asyncio.to_thread(self.client.converse_stream, **self._converse_args(request))
            events = iter(response["stream"])
            while True:
                # Each read from the event stream blocks, so do it in a worker thread too.
                event = await asyncio.to_thread(next, events, None)
                if event is None:
                    break
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                if "text" in delta:
                    yield delta["text"]
        except (ClientError, BotoCoreError) as exc:
            raise self._error(exc) from exc
