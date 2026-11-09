"""Pytest bootstrapping."""

import os
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


_ATLAS_ENV_KEYS = [
    "OPENAI_API_KEY", "OPENAI_MODEL",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET",
    "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBED_DIM", "EMBEDDING_DEVICE",
    "NORMALIZE_EMBEDDINGS",
    "PINECONE_API_KEY", "PINECONE_INDEX_HOST", "PINECONE_INDEX_NAME",
    "PINECONE_CLOUD", "PINECONE_REGION", "PINECONE_READY_TIMEOUT",
    "DATABASE_URL",
    "MIN_RELEVANCE", "MAX_UPLOAD_BYTES",
    "HITL_ENABLED", "MEMORY_TURNS",
    "APP_NAME", "ENVIRONMENT", "LOG_LEVEL",
    "RIUS_API_KEY", "RIUS_SERVICE_NAME", "RIUS_DISABLED", "RIUS_CAPTURE_CONTENT",
]


@pytest.fixture(autouse=True)
def _clean_env_and_cache(monkeypatch):
    """
    Before every test:
      - clear ATLAS env vars from os.environ
      - clear settings() lru_cache so .env is re-read
    After every test, restore.
    """
    # Save and clear
    for key in _ATLAS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    from app.config import settings
    settings.cache_clear()

    yield

    # Reset cache again (in case a test populated it)
    settings.cache_clear()