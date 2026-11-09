"""Tests for app.aws — S3 wrapper against the live bucket."""

import uuid

import pytest

from app.aws import (
    client,
    delete_object,
    download_bytes,
    exists,
    presigned_url,
    upload_bytes,
)
from app.config import settings


TEST_PREFIX = "test/_smoke"


def _key(suffix: str = "") -> str:
    """Generate a unique test key that won't collide with real data."""
    return f"{TEST_PREFIX}/{uuid.uuid4().hex}{suffix}"


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """Clear the lru_cache so each test can verify client creation."""
    client.cache_clear()
    yield
    client.cache_clear()


def test_client_is_cached():
    """Same client instance on repeated calls."""
    c1 = client()
    c2 = client()
    assert c1 is c2


def test_client_region_matches_config():
    """Client is configured for the region in .env."""
    c = client()
    assert c.meta.region_name == settings().aws_region


def test_upload_download_roundtrip():
    """Uploaded bytes are byte-identical when downloaded."""
    key = _key(".txt")
    payload = b"hello from atlas test"

    try:
        upload_bytes(key, payload, "text/plain")
        got = download_bytes(key)
        assert got == payload
    finally:
        delete_object(key)


def test_exists_true_after_upload():
    """exists() returns True for an uploaded object."""
    key = _key(".txt")
    try:
        assert exists(key) is False
        upload_bytes(key, b"x", "text/plain")
        assert exists(key) is True
    finally:
        delete_object(key)


def test_delete_removes_object():
    """After delete, exists() is False again."""
    key = _key(".txt")
    upload_bytes(key, b"temp", "text/plain")
    assert exists(key) is True

    delete_object(key)
    assert exists(key) is False


def test_presigned_url_contains_key():
    """Presigned URL includes the object key path."""
    key = _key(".txt")
    upload_bytes(key, b"x", "text/plain")
    try:
        url = presigned_url(key, expires_in=60)
        assert key in url
        assert url.startswith("https://")
    finally:
        delete_object(key)


def test_client_detects_misconfigured_creds(monkeypatch):
    """If only one of the two AWS keys is set, client() raises."""
    from app import aws

    # Clear cached client so the validator runs
    aws.client.cache_clear()

    s = settings()
    original = (s.aws_access_key_id, s.aws_secret_access_key)
    try:
        # Simulate half-configured creds
        s.aws_access_key_id = "AKIA_FAKE"
        s.aws_secret_access_key = ""
        with pytest.raises(ValueError, match="both AWS_ACCESS_KEY_ID"):
            aws.client()
    finally:
        s.aws_access_key_id, s.aws_secret_access_key = original
        aws.client.cache_clear()