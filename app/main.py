"""
ATLAS — FastAPI application.

Endpoints:
    GET  /health                     - liveness + deps
    POST /upload                     - upload doc, trigger background ingest
    POST /query                      - RAG query (fast mode, or safe mode)
    GET  /documents/{tenant_id}      - list docs
    GET  /documents/status/{doc_id}  - poll ingestion status
    POST /feedback                   - record user feedback

Query modes:
    safe_mode=False (default) — linear rag.query, fast, no HITL
    safe_mode=True            — graph.run, HITL approval gate + rewrite

Run:
    uvicorn app.main:app --reload --port 8000
"""

import datetime
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import guardrails, rag
from app.aws import upload_bytes
from app.config import settings
from app.db import audit, connection
from worker.ingest import process_document


app = FastAPI(
    title="ATLAS Enterprise AI",
    version="0.1.0",
    description="Multi-tenant RAG assistant",
)


# ==========================================
# MODELS
# ==========================================

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field(..., min_length=1, max_length=64)
    conversation_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    safe_mode: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    conversation_id: str
    suspicious: bool
    injection_hits: list[str]
    needs_approval: bool = False
    approval_reason: str = ""
    rewritten: bool = False
    steps: list[str] = []


class UploadResponse(BaseModel):
    document_id: str
    s3_key: str
    status: str
    message: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    current_version: int
    title: str


class FeedbackRequest(BaseModel):
    tenant_id: str
    actor: str
    conversation_id: str | None = None
    rating: int = Field(..., ge=-1, le=1)
    note: str | None = None


# ==========================================
# HEALTH
# ==========================================

@app.get("/health")
def health() -> dict[str, Any]:
    from app.db import check_database

    db_ok = check_database()
    result = {
        "status": "ok" if db_ok else "degraded",
        "components": {"postgres": "ok" if db_ok else "down"},
    }
    if not db_ok:
        return JSONResponse(status_code=503, content=result)
    return result


# ==========================================
# UPLOAD
# ==========================================

@app.post("/upload", response_model=UploadResponse, status_code=202)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    owner_sub: str = Form(...),
) -> UploadResponse:
    try:
        tenant_id = guardrails.validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not owner_sub or not owner_sub.strip():
        raise HTTPException(status_code=400, detail="owner_sub is required")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > settings().max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file too large ({len(data)} > {settings().max_upload_bytes} bytes)",
        )

    doc_id = str(uuid.uuid4())
    # ✅ FIX: timezone-aware UTC — datetime.utcnow() is deprecated
    year = datetime.datetime.now(datetime.UTC).year
    safe_filename = (file.filename or "upload.bin").replace("/", "_").replace("\\", "_")
    s3_key = f"{tenant_id}/{year}/{doc_id}-{safe_filename}"

    try:
        upload_bytes(
            key=s3_key,
            data=data,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"upload failed: {exc}")

    try:
        with connection() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (id, tenant_id, title, owner_sub, s3_key, status, current_version)
                VALUES (%s, %s, %s, %s, %s, 'pending', 1)
                """,
                (doc_id, tenant_id, file.filename or "upload", owner_sub, s3_key),
            )
            audit(
                conn=conn,
                actor=owner_sub,
                tenant=tenant_id,
                action="document_uploaded",
                detail={"document_id": doc_id, "s3_key": s3_key, "size": len(data)},
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"db insert failed: {exc}")

    background_tasks.add_task(process_document, doc_id, actor=owner_sub)

    return UploadResponse(
        document_id=doc_id,
        s3_key=s3_key,
        status="pending",
        message="Upload accepted. Processing in background.",
    )


# ==========================================
# QUERY (fast or safe mode)
# ==========================================

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    try:
        if req.safe_mode:
            from app.graph import run as graph_run
            result = graph_run(
                req.question,
                tenant_id=req.tenant_id,
                conversation_id=req.conversation_id,
                top_k=req.top_k,
            )
        else:
            result = rag.query(
                question=req.question,
                tenant_id=req.tenant_id,
                conversation_id=req.conversation_id,
                top_k=req.top_k,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"query failed: {exc}")

    return QueryResponse(**result)


# ==========================================
# DOCUMENTS
# ==========================================

@app.get("/documents/{tenant_id}")
def list_documents(tenant_id: str, status: str | None = None) -> dict[str, Any]:
    try:
        tenant_id = guardrails.validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    sql = """
        SELECT id, tenant_id, title, owner_sub, status, current_version, created_at
        FROM documents
        WHERE tenant_id = %s
    """
    params: list[Any] = [tenant_id]
    if status:
        sql += " AND status = %s"
        params.append(status)
    sql += " ORDER BY created_at DESC LIMIT 100"

    with connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return {"documents": [dict(r) for r in rows], "count": len(rows)}


@app.get("/documents/status/{document_id}", response_model=DocumentStatusResponse)
def document_status(document_id: str) -> DocumentStatusResponse:
    with connection() as conn:
        row = conn.execute(
            "SELECT id, status, current_version, title FROM documents WHERE id = %s",
            (document_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    return DocumentStatusResponse(
        document_id=row["id"],
        status=row["status"],
        current_version=row["current_version"],
        title=row["title"],
    )


# ==========================================
# FEEDBACK
# ==========================================

@app.post("/feedback", status_code=201)
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    try:
        tenant_id = guardrails.validate_tenant_id(req.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    with connection() as conn:
        row = conn.execute(
            """
            INSERT INTO feedback
                (tenant_id, actor, conversation_id, rating, note)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (tenant_id, req.actor, req.conversation_id, req.rating, req.note),
        ).fetchone()

    return {"id": row["id"], "status": "recorded"}