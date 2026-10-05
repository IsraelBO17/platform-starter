# app/services/pricing.py
import logging
from dataclasses import dataclass

from app.providers.base import CallResult

log = logging.getLogger("usage")


@dataclass(frozen=True)
class Price:
    input_per_million: float          # USD per 1M input tokens
    output_per_million: float         # USD per 1M output tokens
    cached_input_per_million: float | None = None


# Illustrative list prices. Look up today's figures on each provider's pricing page
# and keep this table in configuration, not code, once it matters.
PRICES: dict[str, Price] = {
    "openai/gpt-oss-20b": Price(0.075, 0.30),
    "llama-3.1-8b-instant": Price(0.05, 0.08),
    "gemini-2.5-flash": Price(0.30, 2.50, 0.03),
    "us.amazon.nova-micro-v1:0": Price(0.035, 0.14),
}


def cost_usd(result: CallResult) -> float:
    price = PRICES.get(result.model)
    if price is None:
        return 0.0
    u = result.usage
    cached_rate = price.cached_input_per_million or price.input_per_million
    uncached_input = u.input_tokens - u.cached_input_tokens
    return (
        uncached_input * price.input_per_million
        + u.cached_input_tokens * cached_rate
        + u.output_tokens * price.output_per_million
    ) / 1_000_000


def record_usage(result: CallResult, feature: str) -> float:
    """Log one line per model call. Every call, no exceptions."""
    cost = cost_usd(result)
    log.info(
        "feature=%s model=%s in=%d out=%d cached=%d latency_ms=%.0f cost_usd=%.6f",
        feature, result.model, result.usage.input_tokens, result.usage.output_tokens,
        result.usage.cached_input_tokens, result.latency_ms, cost,
    )
    return cost
