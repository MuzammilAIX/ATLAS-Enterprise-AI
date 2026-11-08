
import os

import pytest

from app.config import Settings, settings


def test_settings_loads_successfully():
    """Test that Settings object initializes successfully."""
    config = Settings()

    assert config is not None


def test_default_openai_configuration():
    """Test default OpenAI configuration."""

    config = Settings(_env_file=None)

    assert config.openai_api_key == ""
    assert config.openai_model == "gpt-5.5"


def test_default_aws_configuration():
    """Test default AWS configuration."""

    config = Settings(_env_file=None)

    assert config.aws_region == "ap-southeast-2"
    assert config.s3_bucket == ""


def test_default_embedding_configuration():
    """Test default embedding configuration."""

    config = Settings(_env_file=None)

    assert config.embedding_provider == "sentence-transformers"
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.embedding_device == "cpu"
    assert config.embed_dim == 384
    assert config.normalize_embeddings is True


def test_default_pinecone_configuration():
    """Test default Pinecone configuration."""

    config = Settings(_env_file=None)

    assert config.pinecone_index_name == "atlas-enterprise-ai-index"
    assert config.pinecone_cloud == "aws"
    assert config.pinecone_region == "ap-southeast-2"
    assert config.pinecone_ready_timeout == 180


def test_default_rag_configuration():
    """Test default RAG configuration."""

    config = Settings(_env_file=None)

    assert config.min_relevance == 0.25
    assert config.max_upload_bytes == 10 * 1024 * 1024


def test_default_agent_configuration():
    """Test default agent configuration."""

    config = Settings(_env_file=None)

    assert config.hitl_enabled is True
    assert config.memory_turns == 6


def test_environment_variable_override(monkeypatch):
    """Test environment variables override default values."""

    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("EMBED_DIM", "768")

    config = Settings(_env_file=None)

    assert config.openai_model == "gpt-test-model"
    assert config.aws_region == "us-east-1"
    assert config.embed_dim == 768


def test_boolean_environment_variable(monkeypatch):
    """Test boolean environment variable parsing."""

    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("NORMALIZE_EMBEDDINGS", "false")

    config = Settings(_env_file=None)

    assert config.hitl_enabled is False
    assert config.normalize_embeddings is False


def test_settings_cache():
    """Test settings() uses caching."""

    settings.cache_clear()

    first = settings()
    second = settings()

    assert first is second

    settings.cache_clear()
