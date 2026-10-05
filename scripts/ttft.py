# scripts/ttft.py
import asyncio
import time

from app.config import get_settings
from app.providers.base import CallRequest, Message
from app.providers.factory import build_provider


async def main():
    s = get_settings()
    provider = build_provider(s)
    request = CallRequest([Message("user", "List five uses of a Foundation Model.")], s.llm_model)
    started = time.perf_counter()
    first = None
    async for _ in provider.stream(request):
        first = first or time.perf_counter()
    end = time.perf_counter()
    print(f"time to first token {(first - started) * 1000:.0f} ms, total {(end - started) * 1000:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
