# app/schemas.py
from typing import Literal

from pydantic import BaseModel, Field


class Triage(BaseModel):
    """The contract the model must honour."""
    category: Literal["billing", "account", "technical", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(..., max_length=120)
