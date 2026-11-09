"""
Observability events — write to Postgres for tracing.

Every significant step (embed, pinecone_query, llm_call, ingestion_step)
records a row in `observability_events`. Cheap, queryable, no external
service required.

Usage:
    from app.observability import trace, observe

    @trace("embed")
    def embed(text): ...

    with observe("llm_call", tenant_id="demo", detail={"model": "gpt-4o-mini"}):
        answer = client.chat.completions.create(...)
"""

import sys
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator

from psycopg.types.json import Jsonb

from app.db import connection


# ==========================================
# CORE WRITER
# ==========================================

def _emit(
    event_type: str,
    tenant_id: str,
    actor: str | None = None,
    trace_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """
    Insert one row into observability_events.

    Never raises — observability must not break the caller.
    Prints to stderr if the insert fails, so config errors surface during dev.
    """
    try:
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO observability_events (
                    tenant_id, actor, trace_id, event_type, detail
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    tenant_id,
                    actor,
                    trace_id,
                    event_type,
                    Jsonb(detail or {}),
                ),
            )
    except Exception as exc:
        print(
            f"[observability] failed to emit {event_type!r}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


# ==========================================
# DECORATOR
# ==========================================

def trace(
    event_type: str,
    tenant_id: str = "system",
    actor: str | None = None,
):
    """
    Decorator that records an observability event around a function call.

    Records: event_type, function name, duration_ms, status (ok/error),
    error message (on exception, then re-raises).
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())
            start = time.perf_counter()
            detail: dict[str, Any] = {"function": fn.__name__}

            try:
                result = fn(*args, **kwargs)
                detail["duration_ms"] = int((time.perf_counter() - start) * 1000)
                detail["status"] = "ok"
                _emit(event_type, tenant_id, actor, trace_id, detail)
                return result
            except Exception as exc:
                detail["duration_ms"] = int((time.perf_counter() - start) * 1000)
                detail["status"] = "error"
                detail["error"] = f"{type(exc).__name__}: {exc}"
                _emit(event_type, tenant_id, actor, trace_id, detail)
                raise

        return wrapper
    return decorator


# ==========================================
# CONTEXT MANAGER
# ==========================================

@contextmanager
def observe(
    event_type: str,
    tenant_id: str = "system",
    actor: str | None = None,
    detail: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Context manager for recording an event.

    Yields a mutable `detail` dict — you can attach fields inside the block:

        with observe("pinecone_query", tenant_id="demo") as d:
            result = index.query(...)
            d["top_k"] = 5
            d["matches"] = len(result.matches)
    """
    trace_id = str(uuid.uuid4())
    start = time.perf_counter()
    d: dict[str, Any] = dict(detail or {})

    try:
        yield d
        d["duration_ms"] = int((time.perf_counter() - start) * 1000)
        d["status"] = "ok"
        _emit(event_type, tenant_id, actor, trace_id, d)
    except Exception as exc:
        d["duration_ms"] = int((time.perf_counter() - start) * 1000)
        d["status"] = "error"
        d["error"] = f"{type(exc).__name__}: {exc}"
        _emit(event_type, tenant_id, actor, trace_id, d)
        raise