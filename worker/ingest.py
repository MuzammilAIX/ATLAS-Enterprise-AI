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

import logging
from typing import Any

from app import vector
from app.aws import download_bytes
from app.config import settings
from app.db import audit, connection
from app.ingest.chunker import chunk_text
from app.ingest.extractor import extract_text
from app.observability import trace

# 🔧 FIX: reuse the single OpenAI-based embedder from app.rag so
# document vectors and query vectors live in the same vector space.
from app.rag import _openai_client


logger = logging.getLogger(__name__)


# ==========================================
# EMBEDDING (OpenAI, batched)
# ==========================================

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts with OpenAI.

    OpenAI's embeddings endpoint accepts up to 2048 inputs per request,
    but for reliability with mixed-size chunks we send them in sub-batches
    of EMBED_BATCH_SIZE.
    """
    if not texts:
        return []

    s = settings()
    client = _openai_client()
    batch_size = s.embed_batch_size

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(
            model=s.openai_embedding_model,
            input=batch,
            dimensions=s.embed_dim,
        )
        # Response preserves input order; collect in order.
        for item in response.data:
            all_vectors.append(item.embedding)

    return all_vectors


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

    logger.info(
        f"Starting ingest: document_id={document_id} "
        f"tenant={tenant_id} title={title!r}"
    )

    try:
        # 2. Mark processing
        _set_status(document_id, "processing")

        # 3. Download from S3
        logger.debug(f"Downloading s3_key={s3_key}")
        data = download_bytes(s3_key)
        logger.debug(f"Downloaded {len(data)} bytes")

        # 🔧 FIX: wrap extract_text so failures are logged with traceback
        logger.debug(f"Extracting text from {title!r} ({len(data)} bytes)")
        try:
            extracted = extract_text(data, filename=title)
        except Exception as e:
            logger.error(
                f"Extraction failed for document_id={document_id} "
                f"title={title!r}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

        if not extracted.strip():
            raise RuntimeError(
                f"no text extracted from document {document_id} "
                f"(title={title!r}, {len(data)} bytes) — "
                "likely a scanned/image-only PDF"
            )

        logger.debug(f"Extracted {len(extracted)} chars from {title!r}")

        # 5. Chunk
        chunks = chunk_text(extracted)
        if not chunks:
            raise RuntimeError(
                f"chunking produced no chunks for document {document_id} "
                f"(title={title!r}, extracted_len={len(extracted)})"
            )

        logger.debug(f"Chunked into {len(chunks)} pieces")

        # 6. Embed all chunks in sub-batches via OpenAI
        logger.debug(f"Embedding {len(chunks)} chunks via OpenAI")
        vectors_vals = _embed_batch(chunks)

        if len(vectors_vals) != len(chunks):
            raise RuntimeError(
                f"embedding returned {len(vectors_vals)} vectors for "
                f"{len(chunks)} chunks — mismatch"
            )

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
        logger.debug(f"Upserting {len(pinecone_vectors)} vectors to Pinecone")
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

        logger.info(
            f"Ingest complete: document_id={document_id} "
            f"chunks={len(chunks)} vectors={up.get('upserted_count', 0)} "
            f"version={new_version}"
        )

        return {
            "status": "ready",
            "chunks": len(chunks),
            "vectors": up.get("upserted_count", 0),
            "version": new_version,
        }

    except Exception as exc:
        # 🔧 FIX: log the full traceback before updating status,
        # so you can see what actually broke in the terminal.
        logger.error(
            f"Ingest FAILED: document_id={document_id} "
            f"title={title!r} "
            f"error={type(exc).__name__}: {exc}",
            exc_info=True,
        )

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
            # Even if DB updates fail, don't mask the original error
            logger.exception(
                f"Failed to record failure status for {document_id}"
            )
        raise