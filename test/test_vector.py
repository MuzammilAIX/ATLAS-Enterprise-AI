"""Tests for app.vector — Pinecone upsert/query/delete against the live index."""

import uuid

import pytest

from app.config import settings
from app.vector import (
    delete_by_tenant,
    delete_ids,
    index,
    query,
    stats,
    upsert,
)


TEST_NS = f"test_{uuid.uuid4().hex[:8]}"
TEST_TENANT = "test_tenant"
OTHER_TENANT = "other_tenant"


@pytest.fixture(scope="module", autouse=True)
def _cleanup():
    yield
    delete_by_tenant(TEST_TENANT, namespace=TEST_NS)
    delete_by_tenant(OTHER_TENANT, namespace=TEST_NS)


def _random_vector(seed: int = 0) -> list[float]:
    import math
    import random
    random.seed(seed)
    v = [random.random() for _ in range(settings().embed_dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _make_vector(
    tenant: str = TEST_TENANT,
    doc_id: str | None = None,
    chunk_i: int = 0,
    seed: int = 0,
    text: str = "sample chunk",
) -> dict:
    return {
        "id": f"{doc_id or uuid.uuid4().hex}:{chunk_i}",
        "values": _random_vector(seed),
        "metadata": {
            "tenant_id": tenant,
            "document_id": doc_id or "doc-" + uuid.uuid4().hex[:8],
            "chunk_index": chunk_i,
            "text": text,
        },
    }


def test_stats_returns_dimension():
    s = stats()
    assert s["dimension"] == settings().embed_dim


def test_upsert_requires_tenant_metadata():
    bad = [{"id": "x", "values": _random_vector(0), "metadata": {}}]
    with pytest.raises(ValueError, match="tenant_id"):
        upsert(bad, namespace=TEST_NS)


def test_upsert_and_query_scoped_to_tenant():
    v = _make_vector(tenant=TEST_TENANT, seed=42, text="the cat sat on the mat")
    upsert([v], namespace=TEST_NS)

    results = query(
        v["values"],
        tenant_id=TEST_TENANT,
        top_k=5,
        namespace=TEST_NS,
        min_score=0.99,
    )

    ids = [r["id"] for r in results]
    assert v["id"] in ids
    top = results[0]
    assert top["metadata"]["tenant_id"] == TEST_TENANT
    assert top["metadata"]["text"] == "the cat sat on the mat"
    assert top["score"] >= 0.99


def test_query_does_not_leak_across_tenants():
    v_a = _make_vector(tenant=TEST_TENANT, seed=11, text="tenant A secret")
    v_b = _make_vector(tenant=OTHER_TENANT, seed=22, text="tenant B secret")
    upsert([v_a, v_b], namespace=TEST_NS)

    results_a = query(
        v_a["values"],
        tenant_id=TEST_TENANT,
        top_k=10,
        namespace=TEST_NS,
        min_score=0.5,
    )
    for r in results_a:
        assert r["metadata"]["tenant_id"] == TEST_TENANT

    results_b = query(
        v_b["values"],
        tenant_id=OTHER_TENANT,
        top_k=10,
        namespace=TEST_NS,
        min_score=0.5,
    )
    for r in results_b:
        assert r["metadata"]["tenant_id"] == OTHER_TENANT

    b_ids = {r["id"] for r in results_b}
    assert v_a["id"] not in b_ids


def test_delete_by_ids_removes_vectors():
    v = _make_vector(tenant=TEST_TENANT, seed=99)
    upsert([v], namespace=TEST_NS)

    results_before = query(
        v["values"], tenant_id=TEST_TENANT, top_k=5,
        namespace=TEST_NS, min_score=0.99,
    )
    assert any(r["id"] == v["id"] for r in results_before)

    delete_ids([v["id"]], namespace=TEST_NS)

    import time
    for _ in range(10):
        results_after = query(
            v["values"], tenant_id=TEST_TENANT, top_k=5,
            namespace=TEST_NS, min_score=0.99,
        )
        if not any(r["id"] == v["id"] for r in results_after):
            break
        time.sleep(0.5)
    else:
        pytest.fail("vector still present after delete")


def test_delete_by_tenant_removes_all():
    t = "purge_tenant"
    vectors = [_make_vector(tenant=t, seed=i, chunk_i=i) for i in range(3)]
    upsert(vectors, namespace=TEST_NS)

    delete_by_tenant(t, namespace=TEST_NS)

    import time
    for _ in range(10):
        results = query(
            vectors[0]["values"], tenant_id=t, top_k=10,
            namespace=TEST_NS, min_score=0.0,
        )
        if not results:
            break
        time.sleep(0.5)
    else:
        pytest.fail("vectors still present after delete_by_tenant")