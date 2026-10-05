# scripts/compare_models.py
"""Send the same prompt to several models and compare latency, tokens and cost."""
import asyncio
import statistics

from app.config import get_settings
from app.providers.base import CallRequest, Message
from app.providers.bedrock import BedrockProvider
from app.providers.factory import build_provider
from app.services.pricing import cost_usd
from app.services.retry import chat_with_retry

PROMPT = [
    Message("system", "Answer in exactly two sentences."),
    Message("user", "Why do software engineers consume Foundation Models instead of training them?"),
]
RUNS = 3


async def measure(label, provider, model):
    latencies, costs, last = [], [], None
    for _ in range(RUNS):
        result = await chat_with_retry(provider, CallRequest(PROMPT, model, max_tokens=150))
        latencies.append(result.latency_ms)
        costs.append(cost_usd(result))
        last = result
    print(f"{label:<28} median {statistics.median(latencies):>7.0f} ms   "
          f"in {last.usage.input_tokens:>4}  out {last.usage.output_tokens:>4}   "
          f"${statistics.mean(costs):.6f}/call")
    print(f"    {last.text.strip()[:110]}...")


async def main():
    s = get_settings()
    primary = build_provider(s)                     # your configured provider (Groq, Gemini, ...)
    bedrock = BedrockProvider(s.aws_region, s.llm_timeout_seconds)
    candidates = [(f"{primary.name} {s.llm_model}", primary, s.llm_model)]
    if s.llm_provider == "groq":
        candidates.append(("groq llama-3.1-8b-instant", primary, "llama-3.1-8b-instant"))
    candidates += [
        ("Bedrock Nova Micro", bedrock, "us.amazon.nova-micro-v1:0"),
        ("Bedrock Llama 3.1 8B", bedrock, "us.meta.llama3-1-8b-instruct-v1:0"),
        ("Bedrock Mistral Small", bedrock, "mistral.mistral-small-2402-v1:0"),
    ]
    for label, provider, model in candidates:
        try:
            await measure(label, provider, model)
        except Exception as exc:  # one failing model should not stop the comparison
            print(f"{label:<28} FAILED: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
