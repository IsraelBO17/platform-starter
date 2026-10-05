# app/providers/base.py
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class CallRequest:
    messages: list[Message]
    model: str
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 512
    stop: list[str] | None = None
    json_schema: dict | None = None   # set this to ask for structured output


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass
class CallResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str | None = None


class ProviderError(Exception):
    """Any failure talking to a model provider."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(Protocol):
    """The one interface the rest of the platform depends on."""

    name: str

    async def chat(self, request: CallRequest) -> CallResult: ...

    def stream(self, request: CallRequest) -> AsyncIterator[str]: ...
