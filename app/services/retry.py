# app/services/retry.py
import asyncio
import logging
import random

from app.providers.base import CallRequest, CallResult, LLMProvider, ProviderError

log = logging.getLogger(__name__)


async def chat_with_retry(
    provider: LLMProvider,
    request: CallRequest,
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
) -> CallResult:
    """Retry rate limits and transient failures with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return await provider.chat(request)
        except ProviderError as exc:
            if not exc.retryable or attempt == max_retries:
                raise
            delay = min(max_delay, base_delay * 2**attempt) * random.uniform(0.5, 1.0)
            log.warning("Attempt %d failed (%s); retrying in %.1fs", attempt + 1, exc, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
