# app/services/history.py
from typing import Protocol

from psycopg_pool import AsyncConnectionPool

from app.providers.base import Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id TEXT        NOT NULL,
    role            TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT        NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conv_messages ON conversation_messages (conversation_id, id);
"""


class HistoryStore(Protocol):
    async def load(self, conversation_id: str) -> list[Message]: ...
    async def append(self, conversation_id: str, message: Message) -> None: ...


class PostgresHistory:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_schema(self) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(SCHEMA)

    async def load(self, conversation_id: str) -> list[Message]:
        async with self.pool.connection() as conn:
            rows = await conn.execute(
                "SELECT role, content FROM conversation_messages "
                "WHERE conversation_id = %s ORDER BY id",
                (conversation_id,),
            )
            return [Message(role, content) for role, content in await rows.fetchall()]

    async def append(self, conversation_id: str, message: Message) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO conversation_messages (conversation_id, role, content) "
                "VALUES (%s, %s, %s)",
                (conversation_id, message.role, message.content),
            )


class InMemoryHistory:
    """For tests and for running without a database."""

    def __init__(self):
        self.data: dict[str, list[Message]] = {}

    async def load(self, conversation_id: str) -> list[Message]:
        return list(self.data.get(conversation_id, []))

    async def append(self, conversation_id: str, message: Message) -> None:
        self.data.setdefault(conversation_id, []).append(message)
