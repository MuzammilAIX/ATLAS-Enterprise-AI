from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import settings


# ==========================================
# DATABASE CONNECTION
# ==========================================

@contextmanager
def connection() -> Iterator[Connection]:
    """
    Create a PostgreSQL connection.

    Uses Neon PostgreSQL through DATABASE_URL.
    Commits on success, rolls back on error, and closes on exit.
    """

    database_url = settings().database_url

    if not database_url or not database_url.strip():
        raise RuntimeError(
            "DATABASE_URL is missing from the .env file"
        )

    with psycopg.connect(
        database_url.strip(),
        row_factory=dict_row,
        connect_timeout=10,
    ) as conn:
        yield conn


# ==========================================
# DATABASE HEALTH CHECK
# ==========================================

def check_database() -> bool:
    """
    Check whether PostgreSQL is accessible.

    Returns False on any connection or query failure, including
    missing configuration.
    """

    try:
        with connection() as conn:
            result = conn.execute(
                "SELECT 1 AS status"
            ).fetchone()

            return result is not None and result["status"] == 1

    except (psycopg.Error, RuntimeError):
        return False


# ==========================================
# AUDIT LOGGING
# ==========================================

def audit(
    conn: Connection,
    actor: str,
    tenant: str,
    action: str,
    detail: dict[str, Any] | None = None,
) -> int:
    """
    Record an application event in audit_events.

    Examples:
        document_uploaded
        document_deleted
        document_processed
        rag_query
        approval_requested
        approval_completed
    """

    if not actor or not tenant or not action:
        raise ValueError(
            "Actor, tenant and action are required"
        )

    result = conn.execute(
        """
        INSERT INTO audit_events (
            actor,
            tenant_id,
            action,
            detail
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (
            actor,
            tenant,
            action,
            Jsonb(detail if detail is not None else {}),
        ),
    )

    row = result.fetchone()
    if row is None:
        raise RuntimeError("Failed to insert audit event")
    return row["id"]


# ==========================================
# DOCUMENT OPERATIONS
# ==========================================

def get_document(
    document_id: str,
    tenant_id: str,
) -> dict | None:
    """
    Retrieve a document belonging to a specific tenant.
    """

    with connection() as conn:
        document = conn.execute(
            """
            SELECT *
            FROM documents
            WHERE id = %s
              AND tenant_id = %s
            """,
            (document_id, tenant_id),
        ).fetchone()

        return document


def list_documents(
    tenant_id: str,
    status: str = "ready",
) -> list[dict]:
    """
    Retrieve documents belonging to a tenant.
    """

    with connection() as conn:
        documents = conn.execute(
            """
            SELECT *
            FROM documents
            WHERE tenant_id = %s
              AND status = %s
            ORDER BY created_at DESC
            """,
            (tenant_id, status),
        ).fetchall()

        return documents


# ==========================================
# CONVERSATION MEMORY
# ==========================================

def get_recent_messages(
    tenant_id: str,
    conversation_id: str,
    limit: int | None = None,
) -> list[dict]:
    """
    Retrieve recent conversation messages.

    MEMORY_TURNS controls the number of conversation turns.
    Each turn normally contains one user and one assistant message.
    """

    if limit is None:
        limit = settings().memory_turns * 2

    if limit < 1:
        raise ValueError("Message limit must be positive")

    with connection() as conn:
        messages = conn.execute(
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
            (tenant_id, conversation_id, limit),
        ).fetchall()

        return messages