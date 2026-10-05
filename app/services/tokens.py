# app/services/tokens.py
from functools import lru_cache

from app.providers.base import Message

PER_MESSAGE_OVERHEAD = 4  # role markers and separators


@lru_cache
def _encoding():
    # o200k_base is OpenAI's tokenizer. Other providers tokenise differently, so treat
    # this as an estimate and trust the provider's usage numbers for billing.
    # tiktoken downloads the vocabulary on first use; if that fails, fall back.
    try:
        import tiktoken
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is None:
        return max(1, len(text) // 4)  # rule of thumb: ~4 characters per English token
    return len(encoding.encode(text))


def count_message_tokens(messages: list[Message]) -> int:
    return sum(count_tokens(m.content) + PER_MESSAGE_OVERHEAD for m in messages)


def fit_to_budget(messages: list[Message], budget: int) -> list[Message]:
    """Keep the system prompt and the newest turns; drop the oldest turns first."""
    system = [m for m in messages if m.role == "system"]
    turns = [m for m in messages if m.role != "system"]
    kept: list[Message] = []
    used = count_message_tokens(system)
    for message in reversed(turns):
        cost = count_message_tokens([message])
        if used + cost > budget:
            break
        kept.insert(0, message)
        used += cost
    return system + kept
