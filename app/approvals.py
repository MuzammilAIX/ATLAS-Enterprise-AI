"""
Human-in-the-loop (HITL) approvals workflow.

Persists approval requests to the `approvals` table when safe mode flags
a sensitive question. Reviewers can list pending requests and record
approve/reject decisions. Every action is audited.

Status lifecycle:
    pending → approved | rejected | cancelled

Approved requests do NOT auto-execute the original question — they just
record that a human approved it. If you want auto-execution, add a
background worker that re-runs the question after approval.
"""

from typing import Any

from app.db import audit, connection
from app.observability import trace


# ==========================================
# WRITE
# ==========================================

@trace("approval_create", tenant_id="system")
def create(
    tenant_id: str,
    requester: str,
    question: str,
) -> str:
    """
    Insert a pending approval request. Returns the new approval's UUID.

    Called by app/graph.py when safe mode flags a sensitive question.
    """

    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id is required")
    if not requester or not requester.strip():
        raise ValueError("requester is required")
    if not question or not question.strip():
        raise ValueError("question is required")

    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO approvals (tenant_id, requester, question, status)
            VALUES (%s, %s, %s, 'pending')
            RETURNING id
            """,
            (tenant_id, requester, question),
        ).fetchone()
        approval_id = str(row["id"])

        audit(
            conn=conn,
            actor=requester,
            tenant=tenant_id,
            action="approval_requested",
            detail={
                "approval_id": approval_id,
                "question_preview": question[:200],
            },
        )

    return approval_id


@trace("approval_decide", tenant_id="system")
def decide(
    approval_id: str,
    reviewer: str,
    approve: bool,
    note: str | None = None,
) -> dict[str, Any]:
    """
    Record a decision on an approval.

    Raises ValueError if:
      - approval not found
      - approval is not in 'pending' status (can't re-decide)

    Returns the updated row.
    """

    if not approval_id:
        raise ValueError("approval_id is required")
    if not reviewer or not reviewer.strip():
        raise ValueError("reviewer is required")

    new_status = "approved" if approve else "rejected"

    with connection() as conn:
        # Check current status first
        current = conn.execute(
            "SELECT tenant_id, status FROM approvals WHERE id = %s",
            (approval_id,),
        ).fetchone()

        if not current:
            raise ValueError(f"approval {approval_id!r} not found")

        if current["status"] != "pending":
            raise ValueError(
                f"approval {approval_id!r} is already {current['status']!r}, "
                "cannot decide again"
            )

        tenant_id = current["tenant_id"]

        # Update row
        row = conn.execute(
            """
            UPDATE approvals
            SET status = %s,
                reviewer = %s,
                reviewed_at = NOW()
            WHERE id = %s
            RETURNING id, tenant_id, requester, question, status,
                      reviewer, created_at, reviewed_at
            """,
            (new_status, reviewer, approval_id),
        ).fetchone()

        audit(
            conn=conn,
            actor=reviewer,
            tenant=tenant_id,
            action=f"approval_{new_status}",
            detail={
                "approval_id": approval_id,
                "note": note or "",
            },
        )

    return {
        "id": str(row["id"]),
        "tenant_id": row["tenant_id"],
        "requester": row["requester"],
        "question": row["question"],
        "status": row["status"],
        "reviewer": row["reviewer"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "reviewed_at": row["reviewed_at"].isoformat() if row["reviewed_at"] else None,
    }


@trace("approval_cancel", tenant_id="system")
def cancel(approval_id: str, actor: str) -> dict[str, Any]:
    """
    Cancel a pending approval (requester withdrew the question).
    Only works on 'pending' approvals.
    """

    if not approval_id:
        raise ValueError("approval_id is required")

    with connection() as conn:
        current = conn.execute(
            "SELECT tenant_id, status FROM approvals WHERE id = %s",
            (approval_id,),
        ).fetchone()

        if not current:
            raise ValueError(f"approval {approval_id!r} not found")

        if current["status"] != "pending":
            raise ValueError(
                f"approval {approval_id!r} is already {current['status']!r}"
            )

        tenant_id = current["tenant_id"]

        row = conn.execute(
            """
            UPDATE approvals
            SET status = 'cancelled',
                reviewer = %s,
                reviewed_at = NOW()
            WHERE id = %s
            RETURNING id, status
            """,
            (actor, approval_id),
        ).fetchone()

        audit(
            conn=conn,
            actor=actor,
            tenant=tenant_id,
            action="approval_cancelled",
            detail={"approval_id": approval_id},
        )

    return {"id": str(row["id"]), "status": row["status"]}


# ==========================================
# READ
# ==========================================

@trace("approval_list", tenant_id="system")
def list_pending(
    tenant_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List pending approvals for a tenant, newest first."""

    if not tenant_id:
        raise ValueError("tenant_id is required")

    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tenant_id, requester, question, status,
                   reviewer, created_at, reviewed_at
            FROM approvals
            WHERE tenant_id = %s
              AND status = 'pending'
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (tenant_id, limit),
        ).fetchall()

    return [
        {
            "id": str(r["id"]),
            "tenant_id": r["tenant_id"],
            "requester": r["requester"],
            "question": r["question"],
            "status": r["status"],
            "reviewer": r["reviewer"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "reviewed_at": r["reviewed_at"].isoformat() if r["reviewed_at"] else None,
        }
        for r in rows
    ]


@trace("approval_list_all", tenant_id="system")
def list_all(
    tenant_id: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List all approvals for a tenant (any status), newest first."""

    if not tenant_id:
        raise ValueError("tenant_id is required")

    if limit < 1 or limit > 500:
        raise ValueError("limit must be between 1 and 500")

    with connection() as conn:
        rows = conn.execute(
            """
            SELECT id, tenant_id, requester, question, status,
                   reviewer, created_at, reviewed_at
            FROM approvals
            WHERE tenant_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (tenant_id, limit),
        ).fetchall()

    return [
        {
            "id": str(r["id"]),
            "tenant_id": r["tenant_id"],
            "requester": r["requester"],
            "question": r["question"],
            "status": r["status"],
            "reviewer": r["reviewer"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "reviewed_at": r["reviewed_at"].isoformat() if r["reviewed_at"] else None,
        }
        for r in rows
    ]


@trace("approval_get", tenant_id="system")
def get(approval_id: str) -> dict[str, Any] | None:
    """Fetch one approval by ID. Returns None if not found."""

    if not approval_id:
        raise ValueError("approval_id is required")

    with connection() as conn:
        row = conn.execute(
            """
            SELECT id, tenant_id, requester, question, status,
                   reviewer, created_at, reviewed_at
            FROM approvals
            WHERE id = %s
            """,
            (approval_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "id": str(row["id"]),
        "tenant_id": row["tenant_id"],
        "requester": row["requester"],
        "question": row["question"],
        "status": row["status"],
        "reviewer": row["reviewer"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "reviewed_at": row["reviewed_at"].isoformat() if row["reviewed_at"] else None,
    }