"""
Conversation memory — persists chat turns and retrieves recent context.

Wraps the `chat_messages` table so RAG / agent code can ask for
the last N conversation turns without touching SQL directly.

A "turn" = one user message + one assistant message.
`settings().memory_turns` controls how many turns of context to return.
"""

from typing import Any

from app.config import settings
from app.db import connection
from app.observability import trace


# ==========================================
# WRITE
# ==========================================

def _save(
    tenant_id: str,
    conversation_id: str,
    role: str,
    content: str,
) -> None:
    """Internal: insert one message row."""

    if role not in ("user", "assistant"):
        raise ValueError(f"invalid role: {role!r}")
    if not content or not content.strip():
        raise ValueError("message content cannot be empty")

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (tenant_id, conversation_id, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (tenant_id, conversation_id, role, content),
        )


@trace("memory_save_user", tenant_id="system")
def save_user_message(
    tenant_id: str,
    conversation_id: str,
    content: str,
) -> None:
    """Persist one user message."""
    _save(tenant_id, conversation_id, "user", content)


@trace("memory_save_assistant", tenant_id="system")
def save_assistant_message(
    tenant_id: str,
    conversation_id: str,
    content: str,
) -> None:
    """Persist one assistant message."""
    _save(tenant_id, conversation_id, "assistant", content)


# ==========================================
# READ
# ==========================================

@trace("memory_get_recent", tenant_id="system")
def get_recent(
    tenant_id: str,
    conversation_id: str,
    limit_turns: int | None = None,
) -> list[dict[str, Any]]:
    """
    Return the most recent messages, in chronological order.

    Returns [{"role": "user"|"assistant", "content": str, "created_at": datetime}, ...]

    limit_turns defaults to settings().memory_turns. Each turn ~ 2 messages,
    so we fetch limit_turns * 2 messages.
    """

    if limit_turns is None:
        limit_turns = settings().memory_turns

    if limit_turns < 1:
        raise ValueError("limit_turns must be >= 1")

    limit_messages = limit_turns * 2

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM (
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE tenant_id = %s
                  AND conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            (tenant_id, conversation_id, limit_messages),
        ).fetchall()

    return [
        {
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ==========================================
# DELETE
# ==========================================

@trace("memory_clear", tenant_id="system")
def clear_conversation(
    tenant_id: str,
    conversation_id: str,
) -> int:
    """
    Delete all messages in a conversation. Returns number of rows deleted.

    Used by a "clear chat" UI. Not used by RAG.
    """

    with connection() as conn:
        result = conn.execute(
            """
            DELETE FROM chat_messages
            WHERE tenant_id = %s AND conversation_id = %s
            """,
            (tenant_id, conversation_id),
        )
        return result.rowcount