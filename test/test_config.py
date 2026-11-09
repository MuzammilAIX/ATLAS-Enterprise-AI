import pytest

from app.config import Settings, settings


# ==========================================
# BASIC LOAD
# ==========================================

def test_settings_loads_successfully():
    """Test that Settings object initializes successfully."""
    config = Settings()
    assert config is not None


# ==========================================
# DEFAULT VALUES
# ==========================================

def test_default_application_configuration():
    """Test default application configuration."""

    config = Settings(_env_file=None)

    assert config.app_name == "ATLAS"
    assert config.environment == "development"
    assert config.log_level == "INFO"


def test_default_openai_configuration():
    """Test default OpenAI configuration."""

    config = Settings(_env_file=None)

    assert config.openai_api_key == ""
    assert config.openai_model == "gpt-4o-mini"


def test_default_aws_configuration():
    """Test default AWS configuration."""

    config = Settings(_env_file=None)

    assert config.aws_region == "ap-southeast-2"
    assert config.s3_bucket == ""
    assert config.aws_access_key_id == ""
    assert config.aws_secret_access_key == ""


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
    assert config.pinecone_index_host == ""


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


# ==========================================
# ENVIRONMENT VARIABLE OVERRIDES
# ==========================================

def test_environment_variable_override(monkeypatch):
    """Test environment variables override default values."""

    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    # Keep embedding model/dim consistent so validator passes
    monkeypatch.setenv("EMBED_DIM", "768")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    config = Settings(_env_file=None)

    assert config.openai_model == "gpt-test-model"
    assert config.aws_region == "us-east-1"
    assert config.embed_dim == 768
    assert config.embedding_model == "BAAI/bge-base-en-v1.5"


def test_boolean_environment_variable(monkeypatch):
    """Test boolean environment variable parsing."""

    monkeypatch.setenv("HITL_ENABLED", "false")
    monkeypatch.setenv("NORMALIZE_EMBEDDINGS", "false")

    config = Settings(_env_file=None)

    assert config.hitl_enabled is False
    assert config.normalize_embeddings is False


# ==========================================
# VALIDATION
# ==========================================

def test_embedding_dim_mismatch_rejected(monkeypatch):
    """embed_dim must match the known embedding_model dimension."""

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("EMBED_DIM", "768")

    with pytest.raises(ValueError, match="embed_dim"):
        Settings(_env_file=None)


def test_partial_aws_credentials_rejected(monkeypatch):
    """aws_access_key_id without aws_secret_access_key must fail."""

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA123")

    with pytest.raises(ValueError, match="aws_access_key_id"):
        Settings(_env_file=None)


def test_partial_aws_secret_rejected(monkeypatch):
    """aws_secret_access_key without aws_access_key_id must fail."""

    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    with pytest.raises(ValueError, match="aws_access_key_id"):
        Settings(_env_file=None)


def test_aws_credentials_both_set_accepted(monkeypatch):
    """Both AWS credentials set together is valid."""

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA123")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    config = Settings(_env_file=None)

    assert config.aws_access_key_id == "AKIA123"
    assert config.aws_secret_access_key == "secret"


def test_invalid_pinecone_host_rejected(monkeypatch):
    """pinecone_index_host must be a URL when set."""

    monkeypatch.setenv("PINECONE_INDEX_HOST", "myhost.pinecone.io")

    with pytest.raises(ValueError, match="pinecone_index_host"):
        Settings(_env_file=None)


def test_valid_pinecone_host_accepted(monkeypatch):
    """A properly-formatted pinecone host is accepted."""

    monkeypatch.setenv(
        "PINECONE_INDEX_HOST",
        "https://atlas-index-abc123.svc.aped-1234.pinecone.io",
    )

    config = Settings(_env_file=None)

    assert config.pinecone_index_host.startswith("https://")


def test_invalid_environment_rejected(monkeypatch):
    """environment must be one of the allowed values."""

    monkeypatch.setenv("ENVIRONMENT", "prod")

    with pytest.raises(ValueError, match="environment"):
        Settings(_env_file=None)


def test_min_relevance_out_of_bounds_rejected(monkeypatch):
    """min_relevance must be between 0.0 and 1.0."""

    monkeypatch.setenv("MIN_RELEVANCE", "1.5")

    with pytest.raises(ValueError, match="min_relevance"):
        Settings(_env_file=None)


def test_memory_turns_must_be_positive(monkeypatch):
    """memory_turns must be >= 1."""

    monkeypatch.setenv("MEMORY_TURNS", "0")

    with pytest.raises(ValueError, match="memory_turns"):
        Settings(_env_file=None)


# ==========================================
# CACHING
# ==========================================

def test_settings_cache():
    """Test settings() uses caching."""

    settings.cache_clear()

    first = settings()
    second = settings()

    assert first is second

    settings.cache_clear()