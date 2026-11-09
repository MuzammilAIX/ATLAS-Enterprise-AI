"""Tests for app.main — FastAPI endpoints via TestClient."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ==========================================
# FRONTEND
# ==========================================

def test_homepage_served(client):
    """Root / serves the HTML frontend."""
    r = client.get("/")
    assert r.status_code == 200
    assert "Atlas" in r.text
    assert "/static/js/app.js" in r.text
    assert "/static/css/styles.css" in r.text
    assert "Enterprise AI" in r.text


def test_static_css_served(client):
    """CSS file is served with the expected content."""
    r = client.get("/static/css/styles.css")
    assert r.status_code == 200
    assert "sidebar" in r.text
    assert "composer" in r.text
    assert "#0b1020" in r.text  # dark background


def test_static_js_served(client):
    """JS file is served with the expected content."""
    r = client.get("/static/js/app.js")
    assert r.status_code == 200
    assert "conversationId" in r.text
    assert "/query" in r.text
    assert "/upload" in r.text


# ==========================================
# HEALTH
# ==========================================

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["components"]["postgres"] == "ok"


# ==========================================
# QUERY — FAST MODE
# ==========================================

def test_query_fast_mode(client):
    r = client.post(
        "/query",
        json={
            "question": "What is the refund window?",
            "tenant_id": "test_api_tenant",
            "conversation_id": str(uuid.uuid4()),
            "top_k": 3,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert isinstance(body["answer"], str)
    assert "sources" in body
    assert body["needs_approval"] is False


# ==========================================
# QUERY — SAFE MODE
# ==========================================

def test_query_safe_mode(client):
    r = client.post(
        "/query",
        json={
            "question": "What is the refund window?",
            "tenant_id": "test_api_tenant",
            "conversation_id": str(uuid.uuid4()),
            "top_k": 3,
            "safe_mode": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "steps" in body
    assert "validate" in body["steps"]


def test_query_safe_mode_hitl(client):
    """Safe mode flags salary question as requiring approval."""
    r = client.post(
        "/query",
        json={
            "question": "What is the salary for level 5?",
            "tenant_id": "test_api_tenant",
            "conversation_id": str(uuid.uuid4()),
            "safe_mode": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["needs_approval"] is True
    assert "salary" in body["approval_reason"]


# ==========================================
# QUERY — VALIDATION
# ==========================================

def test_query_rejects_empty_question(client):
    r = client.post(
        "/query",
        json={
            "question": "",
            "tenant_id": "test_api_tenant",
            "conversation_id": str(uuid.uuid4()),
        },
    )
    assert r.status_code == 422


def test_query_rejects_bad_tenant(client):
    r = client.post(
        "/query",
        json={
            "question": "hi",
            "tenant_id": "../etc/passwd",
            "conversation_id": str(uuid.uuid4()),
        },
    )
    assert r.status_code == 400


# ==========================================
# DOCUMENTS
# ==========================================

def test_document_status_not_found(client):
    r = client.get(f"/documents/status/{uuid.uuid4()}")
    assert r.status_code == 404


def test_list_documents_empty(client):
    r = client.get("/documents/test_api_tenant")
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body
    assert isinstance(body["count"], int)


# ==========================================
# FEEDBACK
# ==========================================

def test_feedback_recorded(client):
    r = client.post(
        "/feedback",
        json={
            "tenant_id": "test_api_tenant",
            "actor": "alice",
            "rating": 1,
            "note": "great answer",
        },
    )
    assert r.status_code == 201
    assert "id" in r.json()


def test_feedback_rejects_invalid_rating(client):
    r = client.post(
        "/feedback",
        json={"tenant_id": "test_api_tenant", "actor": "alice", "rating": 5},
    )
    assert r.status_code == 422


# ==========================================
# UPLOAD
# ==========================================

def test_upload_rejects_empty_file(client):
    r = client.post(
        "/upload",
        files={"file": ("test.txt", b"", "text/plain")},
        data={"tenant_id": "test_api_tenant", "owner_sub": "alice"},
    )
    assert r.status_code == 400


def test_upload_rejects_bad_tenant(client):
    r = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
        data={"tenant_id": "../etc", "owner_sub": "alice"},
    )
    assert r.status_code == 400


def test_upload_accepts_valid_file(client):
    content = b"This is a valid test document with refund policy 45 days."
    r = client.post(
        "/upload",
        files={"file": ("policy.txt", content, "text/plain")},
        data={"tenant_id": "test_api_tenant", "owner_sub": "alice"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "pending"
    assert "document_id" in body
    assert "s3_key" in body