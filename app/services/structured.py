# app/services/structured.py
import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.providers.base import CallRequest, LLMProvider, Message
from app.services.retry import chat_with_retry

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """The model could not produce valid output, even after a repair attempt."""


def _extract_json(text: str) -> str:
    """Strip markdown fences and surrounding chatter some models add."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text


async def generate_structured(
    provider: LLMProvider,
    request: CallRequest,
    schema: type[T],
    use_provider_schema: bool = True,
) -> T:
    """Validate, repair once, then reject."""
    if use_provider_schema:
        request.json_schema = schema.model_json_schema()

    result = await chat_with_retry(provider, request)
    try:
        return schema.model_validate_json(_extract_json(result.text))
    except ValidationError as first_error:
        # Repair: show the model its own output and the exact validation errors.
        repair = CallRequest(
            model=request.model,
            temperature=0,
            max_tokens=request.max_tokens,
            json_schema=request.json_schema,
            messages=request.messages
            + [
                Message("assistant", result.text),
                Message(
                    "user",
                    "Your reply did not match the required JSON schema. Errors:\n"
                    f"{first_error}\n"
                    f"Schema:\n{json.dumps(schema.model_json_schema())}\n"
                    "Reply again with only the corrected JSON object.",
                ),
            ],
        )
        repaired = await chat_with_retry(provider, repair)
        try:
            return schema.model_validate_json(_extract_json(repaired.text))
        except ValidationError as second_error:
            raise StructuredOutputError(str(second_error)) from second_error
