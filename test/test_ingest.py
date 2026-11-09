"""
End-to-end ingestion test: S3 upload → documents row → process_document →
Pinecone vectors → RAG query finds them.

This is the full loop. If it passes, the entire ingestion pipeline works.
"""

import uuid

import pytest

from app import memory, rag, vector
from app.aws import delete_object, upload_bytes
from app.db import connection
from worker.ingest import process_document


@pytest.fixture
def ingested_doc():
    """Create a fresh tenant, upload a synthetic doc, and clean up after."""
    tenant = f"test_ingest_{uuid.uuid4().hex[:8]}"
    doc_id = str(uuid.uuid4())
    s3_key = f"test/_ingest/{doc_id}.txt"
    conv = str(uuid.uuid4())

    content = (
        "ATLAS Test Document\n\n"
        "The refund window for enterprise customers is exactly 45 days "
        "from the date of purchase. Refunds are processed within 5 "
        "business days after approval.\n\n"
        "Section 2: Warranty\n\n"
        "All hardware carries a 24-month warranty against manufacturing "
        "defects. Warranty does not cover accidental damage.\n"
    ).encode("utf-8")

    upload_bytes(s3_key, content, "text/plain")

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (id, tenant_id, title, owner_sub, s3_key, status, current_version)
            VALUES (%s, %s, %s, %s, %s, 'pending', 1)
            """,
            (doc_id, tenant, "atlas-test-doc.txt", "test-owner", s3_key),
        )

    yield {
        "tenant": tenant,
        "doc_id": doc_id,
        "s3_key": s3_key,
        "conv": conv,
    }

    # Cleanup
    try:
        delete_object(s3_key)
    except Exception:
        pass
    try:
        vector.delete_by_tenant(tenant)
    except Exception:
        pass
    with connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.execute("DELETE FROM audit_events WHERE tenant_id = %s", (tenant,))


# ==========================================
# INGESTION TESTS
# ==========================================

def test_process_document_succeeds(ingested_doc):
    """process_document transitions pending → ready, embeds, upserts."""

    result = process_document(ingested_doc["doc_id"], actor="test-runner")

    assert result["status"] == "ready"
    assert result["chunks"] >= 1
    assert result["vectors"] >= 1
    assert result["version"] == 2  # bumped from 1

    with connection() as conn:
        row = conn.execute(
            "SELECT status, current_version FROM documents WHERE id = %s",
            (ingested_doc["doc_id"],),
        ).fetchone()
    assert row["status"] == "ready"
    assert row["current_version"] == 2


def test_process_document_creates_pinecone_vectors(ingested_doc):
    """After ingest, vectors exist and are queryable by the tenant."""

    process_document(ingested_doc["doc_id"], actor="test-runner")

    qvec = rag.embed_query("What is the refund window for enterprise customers?")
    matches = vector.query(
        vector=qvec,
        tenant_id=ingested_doc["tenant"],
        top_k=5,
        min_score=0.0,
    )

    assert len(matches) >= 1
    joined = " ".join(m["metadata"].get("text", "") for m in matches)
    assert "45 days" in joined


def test_process_document_records_audit_events(ingested_doc):
    """Both start and complete audit events were recorded."""

    process_document(ingested_doc["doc_id"], actor="test-runner")

    # ✅ FIXED: pass the LIKE pattern as a parameter, not inline with `%`
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT action FROM audit_events
            WHERE tenant_id = %s
              AND action LIKE %s
            ORDER BY id
            """,
            (ingested_doc["tenant"], "document_ingest%"),
        ).fetchall()

    actions = [r["action"] for r in rows]
    assert "document_ingest_started" in actions
    assert "document_ingest_complete" in actions


# ==========================================
# END-TO-END: INGEST → RAG QUERY
# ==========================================

def test_rag_query_finds_ingested_content(ingested_doc):
    """
    The full loop:
      upload → ingest → RAG query returns an answer citing the doc.
    """

    process_document(ingested_doc["doc_id"], actor="test-runner")

    result = rag.query(
        question="What is the refund window for enterprise customers?",
        tenant_id=ingested_doc["tenant"],
        conversation_id=ingested_doc["conv"],
        save_to_memory=False,
    )

    assert len(result["sources"]) >= 1
    found_text = " ".join(
        s["metadata"].get("text", "") for s in result["sources"]
    )
    assert "45 days" in found_text

    answer_low = result["answer"].lower()
    assert (
        "45" in answer_low
        or "forty five" in answer_low
        or "forty-five" in answer_low
    )


# ==========================================
# FAILURE MODES
# ==========================================

def test_process_document_rejects_missing_doc():
    with pytest.raises(ValueError, match="not found"):
        process_document(str(uuid.uuid4()))


def test_process_document_marks_failed_on_empty_s3_object():
    """Empty S3 content → extract fails → status='failed'."""
    tenant = f"test_ingest_{uuid.uuid4().hex[:8]}"
    doc_id = str(uuid.uuid4())
    s3_key = f"test/_ingest/{doc_id}_empty.txt"

    upload_bytes(s3_key, b"", "text/plain")
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO documents
                (id, tenant_id, title, owner_sub, s3_key, status, current_version)
            VALUES (%s, %s, %s, %s, %s, 'pending', 1)
            """,
            (doc_id, tenant, "empty.txt", "test-owner", s3_key),
        )

    try:
        with pytest.raises(RuntimeError, match="no text extracted"):
            process_document(doc_id)

        with connection() as conn:
            row = conn.execute(
                "SELECT status FROM documents WHERE id = %s",
                (doc_id,),
            ).fetchone()
        assert row["status"] == "failed"
    finally:
        try:
            delete_object(s3_key)
        except Exception:
            pass
        with connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
            conn.execute("DELETE FROM audit_events WHERE tenant_id = %s", (tenant,))