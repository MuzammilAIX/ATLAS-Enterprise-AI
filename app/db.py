from contextlib import contextmanager
from typing import Any, Iterator

import atexit
import logging

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.config import settings


_logger = logging.getLogger(__name__)


# ==========================================
# CONNECTION POOL
# ==========================================

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """
    Get (or lazily create) the shared connection pool.

    Pool sizes are tuned for Neon free tier (~10 concurrent connections):
    - min_size=1: keep one connection warm (avoids cold-start delay)
    - max_size=4: never exceed 4 connections (leaves headroom for other clients)
    - timeout=10: fail fast instead of hanging
    - max_idle=60: recycle idle connections to avoid Neon-side timeouts
    - max_lifetime=600: recycle after 10 min (Neon best practice)

    IMPORTANT: open=False + explicit .open() avoids starting worker
    threads inside __init__, which prevents the psycopg_pool
    "cannot join current thread" crash on shutdown.
    """
    global _pool

    if _pool is None:
        database_url = settings().database_url
        if not database_url or not database_url.strip():
            raise RuntimeError("DATABASE_URL is missing from the .env file")

        _pool = ConnectionPool(
            database_url.strip(),
            min_size=1,
            max_size=4,
            timeout=10,
            max_idle=60,
            max_lifetime=600,
            kwargs={"row_factory": dict_row},
            open=False,               # ← don't start workers in __init__
        )
        _pool.open()                  # ← open explicitly (starts workers now)

    return _pool


# ==========================================
# DATABASE CONNECTION
# ==========================================

@contextmanager
def connection() -> Iterator[Connection]:
    """
    Get a connection from the pool.

    Automatically commits on success, rolls back on error,
    and returns the connection to the pool when the `with` block exits.
    """
    pool = _get_pool()
    with pool.connection() as conn:
        yield conn


# ==========================================
# DATABASE HEALTH CHECK
# ==========================================

def check_database() -> bool:
    """Check whether PostgreSQL is accessible."""

    try:
        with connection() as conn:
            result = conn.execute("SELECT 1 AS status").fetchone()
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
    """Record an application event in audit_events."""

    if not actor or not tenant or not action:
        raise ValueError("Actor, tenant and action are required")

    result = conn.execute(
        """
        INSERT INTO audit_events (actor, tenant_id, action, detail)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (actor, tenant, action, Jsonb(detail if detail is not None else {})),
    )

    row = result.fetchone()
    if row is None:
        raise RuntimeError("Failed to insert audit event")
    return row["id"]


# ==========================================
# DOCUMENT OPERATIONS
# ==========================================

def get_document(document_id: str, tenant_id: str) -> dict | None:
    """Retrieve a document belonging to a specific tenant."""

    with connection() as conn:
        return conn.execute(
            """
            SELECT * FROM documents
            WHERE id = %s AND tenant_id = %s
            """,
            (document_id, tenant_id),
        ).fetchone()


def list_documents(tenant_id: str, status: str = "ready") -> list[dict]:
    """Retrieve documents belonging to a tenant."""

    with connection() as conn:
        return conn.execute(
            """
            SELECT * FROM documents
            WHERE tenant_id = %s AND status = %s
            ORDER BY created_at DESC
            """,
            (tenant_id, status),
        ).fetchall()


# ==========================================
# CONVERSATION MEMORY
# ==========================================

def get_recent_messages(
    tenant_id: str,
    conversation_id: str,
    limit: int | None = None,
) -> list[dict]:
    """Retrieve recent conversation messages."""

    if limit is None:
        limit = settings().memory_turns * 2

    if limit < 1:
        raise ValueError("Message limit must be positive")

    with connection() as conn:
        return conn.execute(
            """
            SELECT role, content, created_at
            FROM (
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE tenant_id = %s AND conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
            ) recent
            ORDER BY id ASC
            """,
            (tenant_id, conversation_id, limit),
        ).fetchall()


# ==========================================
# SHUTDOWN HANDLER (safety net)
# ==========================================

def _shutdown_pool():
    """Close the connection pool on process exit."""
    global _pool
    if _pool is not None:
        try:
            _pool.close(timeout=5)
        except Exception as exc:
            _logger.warning("Pool close failed during shutdown: %s", exc)
        finally:
            _pool = None


atexit.register(_shutdown_pool)