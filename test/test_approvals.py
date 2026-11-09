"""Tests for app.approvals — full HITL workflow."""

import uuid

import pytest

from app import approvals
from app.db import connection


def _cleanup_tenant(tenant: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM approvals WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM audit_events WHERE tenant_id = %s", (tenant,))


@pytest.fixture
def tenant():
    t = f"test_appr_{uuid.uuid4().hex[:8]}"
    yield t
    _cleanup_tenant(t)


def test_create_returns_uuid(tenant):
    aid = approvals.create(tenant, "alice", "what is the salary?")
    assert isinstance(aid, str)
    uuid.UUID(aid)  # validates as UUID


def test_create_rejects_empty_tenant():
    with pytest.raises(ValueError, match="tenant_id"):
        approvals.create("", "alice", "q")


def test_create_rejects_empty_requester(tenant):
    with pytest.raises(ValueError, match="requester"):
        approvals.create(tenant, "", "q")


def test_create_rejects_empty_question(tenant):
    with pytest.raises(ValueError, match="question"):
        approvals.create(tenant, "alice", "")


def test_list_pending_returns_only_pending(tenant):
    a1 = approvals.create(tenant, "alice", "q1")
    a2 = approvals.create(tenant, "bob", "q2")
    approvals.decide(a2, reviewer="admin", approve=True)

    pending = approvals.list_pending(tenant)
    assert len(pending) == 1
    assert pending[0]["id"] == a1


def test_list_all_returns_everything(tenant):
    approvals.create(tenant, "alice", "q1")
    a2 = approvals.create(tenant, "bob", "q2")
    approvals.decide(a2, reviewer="admin", approve=False)

    all_items = approvals.list_all(tenant)
    assert len(all_items) == 2


def test_list_pending_is_tenant_scoped(tenant):
    other = f"test_appr_{uuid.uuid4().hex[:8]}"
    try:
        approvals.create(tenant, "alice", "mine")
        approvals.create(other, "bob", "not mine")

        assert len(approvals.list_pending(tenant)) == 1
        assert len(approvals.list_pending(other)) == 1
    finally:
        _cleanup_tenant(other)


def test_get_returns_row(tenant):
    aid = approvals.create(tenant, "alice", "what is the salary?")
    row = approvals.get(aid)
    assert row["id"] == aid
    assert row["status"] == "pending"
    assert row["requester"] == "alice"


def test_get_missing_returns_none():
    assert approvals.get(str(uuid.uuid4())) is None


def test_decide_approve_sets_status(tenant):
    aid = approvals.create(tenant, "alice", "q")
    row = approvals.decide(aid, reviewer="admin", approve=True, note="ok")
    assert row["status"] == "approved"
    assert row["reviewer"] == "admin"
    assert row["reviewed_at"] is not None


def test_decide_reject_sets_status(tenant):
    aid = approvals.create(tenant, "alice", "q")
    row = approvals.decide(aid, reviewer="admin", approve=False)
    assert row["status"] == "rejected"


def test_decide_twice_raises(tenant):
    aid = approvals.create(tenant, "alice", "q")
    approvals.decide(aid, reviewer="admin", approve=True)
    with pytest.raises(ValueError, match="already"):
        approvals.decide(aid, reviewer="admin2", approve=False)


def test_decide_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        approvals.decide(str(uuid.uuid4()), reviewer="admin", approve=True)


def test_cancel_works(tenant):
    aid = approvals.create(tenant, "alice", "q")
    row = approvals.cancel(aid, actor="alice")
    assert row["status"] == "cancelled"


def test_cancel_after_decided_raises(tenant):
    aid = approvals.create(tenant, "alice", "q")
    approvals.decide(aid, reviewer="admin", approve=True)
    with pytest.raises(ValueError, match="already"):
        approvals.cancel(aid, actor="alice")


def test_audit_events_recorded(tenant):
    aid = approvals.create(tenant, "alice", "q")
    approvals.decide(aid, reviewer="admin", approve=True)

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT action FROM audit_events
            WHERE tenant_id = %s
              AND action LIKE %s
            ORDER BY id
            """,
            (tenant, "approval_%"),
        ).fetchall()

    actions = [r["action"] for r in rows]
    assert "approval_requested" in actions
    assert "approval_approved" in actions