
from functools import lru_cache
from pathlib import Path

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
    # OpenAI Configuration
    # ==========================================
    openai_api_key: str = ""
    openai_model: str = "gpt-5.5"

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
    min_relevance: float = 0.25
    max_upload_bytes: int = 10 * 1024 * 1024

    # ==========================================
    # Agent Configuration
    # ==========================================
    hitl_enabled: bool = True
    memory_turns: int = 6


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
