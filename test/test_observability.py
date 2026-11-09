"""Tests for app.observability — trace decorator + observe context manager."""

import pytest

from app.db import connection
from app.observability import observe, trace


def _count_events(event_type: str, tenant_id: str) -> int:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM observability_events
            WHERE event_type = %s AND tenant_id = %s
            """,
            (event_type, tenant_id),
        ).fetchone()
    return row["n"]


def _latest_event(event_type: str, tenant_id: str) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            """
            SELECT event_type, tenant_id, actor, trace_id, detail
            FROM observability_events
            WHERE event_type = %s AND tenant_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (event_type, tenant_id),
        ).fetchone()
    return row


def test_trace_records_success():
    event_type = "test_trace_success"
    tenant = "test_tenant"

    @trace(event_type, tenant_id=tenant, actor="alice")
    def add(a, b):
        return a + b

    before = _count_events(event_type, tenant)
    assert add(2, 3) == 5
    after = _count_events(event_type, tenant)

    assert after - before == 1

    row = _latest_event(event_type, tenant)
    assert row["actor"] == "alice"
    assert row["detail"]["status"] == "ok"
    assert row["detail"]["function"] == "add"
    assert row["detail"]["duration_ms"] >= 0


def test_trace_records_error_and_reraises():
    event_type = "test_trace_error"
    tenant = "test_tenant"

    @trace(event_type, tenant_id=tenant)
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        boom()

    row = _latest_event(event_type, tenant)
    assert row is not None
    assert row["detail"]["status"] == "error"
    assert "ValueError: kaboom" in row["detail"]["error"]


def test_observe_context_manager_records():
    event_type = "test_observe_ok"
    tenant = "test_tenant"

    with observe(event_type, tenant_id=tenant, actor="bob") as d:
        d["custom_field"] = 42

    row = _latest_event(event_type, tenant)
    assert row is not None
    assert row["actor"] == "bob"
    assert row["detail"]["custom_field"] == 42
    assert row["detail"]["status"] == "ok"


def test_observe_context_manager_records_error():
    event_type = "test_observe_err"
    tenant = "test_tenant"

    with pytest.raises(RuntimeError, match="kaboom"):
        with observe(event_type, tenant_id=tenant):
            raise RuntimeError("kaboom")

    row = _latest_event(event_type, tenant)
    assert row is not None
    assert row["detail"]["status"] == "error"
    assert "RuntimeError: kaboom" in row["detail"]["error"]


def test_observability_swallows_db_errors(monkeypatch):
    from app import observability

    def broken_connection():
        raise RuntimeError("db down")

    monkeypatch.setattr(observability, "connection", broken_connection)

    # Must not raise
    observability._emit(
        event_type="test_broken",
        tenant_id="test_tenant",
        detail={"note": "should be swallowed"},
    )