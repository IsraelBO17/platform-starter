# scripts/first_call.py
import asyncio
import sys

from app.config import get_settings
from app.providers.base import CallRequest, Message
from app.providers.factory import build_provider


async def main(temperature: float, max_tokens: int):
    settings = get_settings()
    provider = build_provider(settings)
    request = CallRequest(
        messages=[
            Message("system", "You are a helpful assistant. Answer in one sentence."),
            Message("user", "Give me a name for a Lagos-based food delivery startup."),
        ],
        model=settings.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    result = await provider.chat(request)
    print(f"provider : {provider.name}  model: {result.model}")
    print(f"reply    : {result.text.strip()}")
    print(f"tokens   : in={result.usage.input_tokens} out={result.usage.output_tokens}")
    print(f"latency  : {result.latency_ms:.0f} ms   finish_reason: {result.finish_reason}")


if __name__ == "__main__":
    temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    asyncio.run(main(temperature, max_tokens))
