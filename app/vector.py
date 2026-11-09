"""
Pinecone vector store operations.

Every vector is scoped to a tenant via metadata (`tenant_id`) so RAG queries
never leak across tenants. Upsert, query, delete-by-id, delete-by-tenant.

The Pinecone index must be dimension=384 (matches embedding_model
BAAI/bge-small-en-v1.5) and metric=cosine. Verified by the smoke test.
"""

from functools import lru_cache
from typing import Any

from pinecone import Pinecone
from pinecone.exceptions.exceptions import NotFoundException

from app.config import settings
from app.observability import trace


@lru_cache(maxsize=1)
def _pc() -> Pinecone:
    """Return a cached Pinecone client."""
    s = settings()
    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in .env")
    return Pinecone(api_key=s.pinecone_api_key)


@lru_cache(maxsize=1)
def index():
    """Return a cached Pinecone Index handle."""

    s = settings()

    if s.pinecone_index_host:
        return _pc().Index(host=s.pinecone_index_host)

    return _pc().Index(s.pinecone_index_name)


# ==========================================
# WRITE
# ==========================================

@trace("pinecone_upsert", tenant_id="system")
def upsert(
    vectors: list[dict[str, Any]],
    namespace: str = "",
) -> dict:
    """
    Upsert a batch of vectors.

    Each vector: {
        "id": str,                       # e.g. "docid:chunk_i"
        "values": list[float],           # 384-dim
        "metadata": {
            "tenant_id": str,            # REQUIRED for multi-tenant safety
            "document_id": str,
            "chunk_index": int,
            "text": str,
            ... any other fields
        }
    }

    Returns: { "upserted_count": int }
    """

    if not vectors:
        return {"upserted_count": 0}

    for v in vectors:
        if "metadata" not in v or "tenant_id" not in v["metadata"]:
            raise ValueError(
                f"vector {v.get('id')!r} has no metadata.tenant_id — "
                "multi-tenant safety requires it"
            )

    result = index().upsert(vectors=vectors, namespace=namespace)
    return {"upserted_count": getattr(result, "upserted_count", 0)}


# ==========================================
# READ
# ==========================================

@trace("pinecone_query", tenant_id="system")
def query(
    vector: list[float],
    tenant_id: str,
    top_k: int = 5,
    namespace: str = "",
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """
    Query by embedding vector, scoped to a tenant.

    Returns a list of dicts:
        [{"id": str, "score": float, "metadata": dict}, ...]

    If min_score is set, results below that score are dropped.
    Empty namespace (default) query against un-namespaced vectors.
    """

    s = settings()
    if min_score is None:
        min_score = s.min_relevance

    result = index().query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter={"tenant_id": {"$eq": tenant_id}},
    )

    matches = []
    for m in result.matches:
        if m.score >= min_score:
            matches.append(
                {
                    "id": m.id,
                    "score": m.score,
                    "metadata": dict(m.metadata or {}),
                }
            )
    return matches


# ==========================================
# DELETE
# ==========================================

@trace("pinecone_delete_ids", tenant_id="system")
def delete_ids(ids: list[str], namespace: str = "") -> None:
    """Delete specific vector IDs."""

    if not ids:
        return

    try:
        index().delete(ids=ids, namespace=namespace)
    except NotFoundException:
        # Namespace doesn't exist → nothing to delete
        return


@trace("pinecone_delete_tenant", tenant_id="system")
def delete_by_tenant(tenant_id: str, namespace: str = "") -> None:
    """
    Delete ALL vectors for a tenant. Irreversible — use with care.

    Safe to call when the tenant has zero vectors: Pinecone returns
    NotFoundException ("Namespace not found") for a tenant that was
    never upserted into, and we treat that as "nothing to delete".
    """

    if not tenant_id:
        raise ValueError("tenant_id is required")

    try:
        index().delete(
            filter={"tenant_id": {"$eq": tenant_id}},
            namespace=namespace,
        )
    except NotFoundException:
        # Namespace never created (tenant has no vectors) — nothing to delete
        return


# ==========================================
# STATS
# ==========================================

def stats() -> dict:
    """Return index statistics (dimension, vector count, namespaces)."""

    s = index().describe_index_stats()
    return {
        "dimension": s.dimension,
        "total_vector_count": s.total_vector_count,
        "namespaces": {k: v.vector_count for k, v in (s.namespaces or {}).items()},
    }