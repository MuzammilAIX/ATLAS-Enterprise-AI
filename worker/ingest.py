"""
Document ingestion pipeline.

Takes a document already in S3, extracts text, chunks it, embeds each chunk,
upserts to Pinecone, and updates the `documents` row's status.

Status lifecycle:
  pending → processing → ready | failed

Usage:
    from worker.ingest import process_document
    result = process_document("some-document-uuid")
"""

from functools import lru_cache
from typing import Any

from app import vector
from app.aws import download_bytes
from app.config import settings
from app.db import audit, connection
from app.ingest.chunker import chunk_text
from app.ingest.extractor import extract_text
from app.observability import trace


# ==========================================
# EMBEDDING
# ==========================================

@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    s = settings()
    return SentenceTransformer(s.embedding_model, device=s.embedding_device)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    s = settings()
    if not texts:
        return []
    vecs = _embedder().encode(
        texts,
        normalize_embeddings=s.normalize_embeddings,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return vecs.tolist()


# ==========================================
# DB HELPERS
# ==========================================

def _set_status(document_id: str, status: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE documents SET status = %s, updated_at = NOW() WHERE id = %s",
            (status, document_id),
        )


def _bump_version(document_id: str) -> int:
    with connection() as conn:
        row = conn.execute(
            """
            UPDATE documents
            SET current_version = current_version + 1, updated_at = NOW()
            WHERE id = %s
            RETURNING current_version
            """,
            (document_id,),
        ).fetchone()
        return row["current_version"] if row else 0


def _audit(action: str, actor: str, tenant: str, detail: dict) -> None:
    """Best-effort audit — never crashes the pipeline."""
    try:
        with connection() as conn:
            audit(conn=conn, actor=actor, tenant=tenant, action=action, detail=detail)
    except Exception:
        pass


# ==========================================
# MAIN PIPELINE
# ==========================================

@trace("ingest_process_document", tenant_id="system")
def process_document(
    document_id: str,
    actor: str = "system",
) -> dict[str, Any]:
    """
    Full ingestion pipeline for one document.

    Expects `documents` row with `s3_key` pointing to a real S3 object.

    Returns:
        {"status": "ready", "chunks": N, "vectors": M, "version": V}

    Raises on catastrophic errors; per-file problems update `documents.status`
    to 'failed' and re-raise so the caller can log/retry.
    """

    # 1. Load document metadata
    with connection() as conn:
        doc = conn.execute(
            """
            SELECT id, tenant_id, s3_key, title, owner_sub, status
            FROM documents
            WHERE id = %s
            """,
            (document_id,),
        ).fetchone()

    if not doc:
        raise ValueError(f"document_id {document_id!r} not found")

    if doc["status"] == "processing":
        raise RuntimeError(f"document {document_id!r} is already processing")

    tenant_id = doc["tenant_id"]
    s3_key = doc["s3_key"]
    title = doc["title"]

    _audit(
        "document_ingest_started",
        actor=actor,
        tenant=tenant_id,
        detail={"document_id": document_id, "s3_key": s3_key},
    )

    try:
        # 2. Mark processing
        _set_status(document_id, "processing")

        # 3. Download from S3
        data = download_bytes(s3_key)

        # 4. Extract text
        extracted = extract_text(data, filename=title)
        if not extracted.strip():
            raise RuntimeError("no text extracted from document")

        # 5. Chunk
        chunks = chunk_text(extracted)
        if not chunks:
            raise RuntimeError("chunking produced no chunks")

        # 6. Embed all chunks in one batch
        vectors_vals = _embed_batch(chunks)

        # 7. Build Pinecone records
        new_version = _bump_version(document_id)
        pinecone_vectors = []
        for i, (text, vec) in enumerate(zip(chunks, vectors_vals)):
            pinecone_vectors.append(
                {
                    "id": f"{document_id}:v{new_version}:{i}",
                    "values": vec,
                    "metadata": {
                        "tenant_id": tenant_id,
                        "document_id": document_id,
                        "chunk_index": i,
                        "version": new_version,
                        "text": text,
                    },
                }
            )

        # 8. Upsert to Pinecone
        up = vector.upsert(pinecone_vectors)

        # 9. Mark ready
        _set_status(document_id, "ready")

        _audit(
            "document_ingest_complete",
            actor=actor,
            tenant=tenant_id,
            detail={
                "document_id": document_id,
                "chunks": len(chunks),
                "vectors": up.get("upserted_count", 0),
                "version": new_version,
            },
        )

        return {
            "status": "ready",
            "chunks": len(chunks),
            "vectors": up.get("upserted_count", 0),
            "version": new_version,
        }

    except Exception as exc:
        try:
            _set_status(document_id, "failed")
            _audit(
                "document_ingest_failed",
                actor=actor,
                tenant=tenant_id,
                detail={
                    "document_id": document_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        except Exception:
            pass
        raise