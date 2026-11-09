from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ==========================================
    # Application
    # ==========================================
    app_name: str = "ATLAS"
    environment: str = Field(
        default="development",
        pattern="^(development|staging|production)$",
    )
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )

    # ==========================================
    # OpenAI Configuration
    # ==========================================
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"          # ← FIXED

    # ==========================================
    # AWS Configuration (S3 only)
    # ==========================================
    aws_region: str = "ap-southeast-2"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = ""

    # ==========================================
    # Embedding Configuration
    # ==========================================
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embed_dim: int = 384
    normalize_embeddings: bool = True

    # ==========================================
    # Pinecone Configuration
    # ==========================================
    pinecone_api_key: str = ""
    pinecone_index_host: str = ""
    pinecone_index_name: str = "atlas-enterprise-ai-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "ap-southeast-2"
    pinecone_ready_timeout: int = 180

    # ==========================================
    # PostgreSQL Configuration
    # ==========================================
    database_url: str = ""

    # ==========================================
    # RAG Configuration
    # ==========================================
    min_relevance: float = Field(default=0.25, ge=0.0, le=1.0)
    max_upload_bytes: int = 10 * 1024 * 1024

    # ==========================================
    # Agent Configuration
    # ==========================================
    hitl_enabled: bool = True
    memory_turns: int = Field(default=6, ge=1)

    # ==========================================
    # Validators
    # ==========================================
    @model_validator(mode="after")
    def _check_embedding_dim(self) -> "Settings":
        known = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        expected = known.get(self.embedding_model)
        if expected is not None and expected != self.embed_dim:
            raise ValueError(
                f"embed_dim={self.embed_dim} does not match "
                f"embedding_model='{self.embedding_model}' "
                f"(expected {expected})"
            )
        return self

    @model_validator(mode="after")
    def _check_aws_credentials(self) -> "Settings":
        has_key = bool(self.aws_access_key_id.strip())
        has_secret = bool(self.aws_secret_access_key.strip())
        if has_key != has_secret:
            raise ValueError(
                "aws_access_key_id and aws_secret_access_key must be "
                "set together, or both left empty to use IAM role / default chain"
            )
        return self

    @model_validator(mode="after")
    def _check_pinecone_host(self) -> "Settings":
        if self.pinecone_index_host:
            host = self.pinecone_index_host.strip()
            if not host.startswith(("http://", "https://")):
                raise ValueError(
                    "pinecone_index_host must start with http:// or https://"
                )
        return self


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()