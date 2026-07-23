"""
Configuration Engine for Invariable Docs.

Loads configuration from `.env` using `pydantic-settings` with type validation
and defaults tailored for local V1 running on MacBook Air M5 (16GB RAM).
"""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Inject .env into os.environ so third-party libraries (like litellm) can find API keys automatically
load_dotenv()

class Settings(BaseSettings):
    """System-wide settings managed via environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # Provider Selection
    # --------------------------------------------------------------------------
    LLM_PROVIDER: Literal["ollama", "openai", "anthropic", "litellm"] = Field(
        default="ollama", description="Active Language Model provider."
    )
    EMBEDDING_PROVIDER: Literal["local_bge", "openai", "cohere"] = Field(
        default="local_bge", description="Active Dense Embedding provider."
    )
    RERANKER_PROVIDER: Literal["local_bge", "cohere", "jina", "none"] = Field(
        default="local_bge", description="Active Cross-Encoder Re-Ranker provider."
    )
    VECTOR_STORE_PROVIDER: Literal["qdrant_local", "qdrant_cloud", "pinecone"] = Field(
        default="qdrant_local", description="Active Vector Database provider."
    )
    OBSERVABILITY_PROVIDER: Literal["local_logging", "langfuse_selfhosted", "langfuse_cloud"] = Field(
        default="local_logging", description="Active Telemetry & Observability provider."
    )

    # --------------------------------------------------------------------------
    # V1 Local Provider Settings (MacBook Air M5 Optimization)
    # --------------------------------------------------------------------------
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama API base URL.")
    OLLAMA_MODEL: str = Field(default="llama3.1:8b-instruct-q4_K_M", description="Local LLM model name.")
    OLLAMA_TEMPERATURE: float = Field(default=0.1, description="LLM generation temperature.")
    OLLAMA_TOP_P: float = Field(default=0.90, description="Nucleus sampling threshold.")
    
    LLM_MODEL_NAME: str = Field(default="groq/llama-3.3-70b-versatile", description="LiteLLM provider model string.")

    LOCAL_EMBEDDING_MODEL: str = Field(default="BAAI/bge-large-en-v1.5", description="Local BGE model repo.")
    LOCAL_EMBEDDING_DIMENSION: int = Field(default=1024, description="Embedding vector dimension.")
    LOCAL_EMBEDDING_DEVICE: str = Field(default="mps", description="PyTorch hardware device (mps/cpu/cuda).")

    LOCAL_RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3", description="Cross-encoder model repo.")
    LOCAL_RERANKER_DEVICE: str = Field(default="mps", description="PyTorch hardware device for re-ranker.")
    RERANKER_TOP_N: int = Field(default=4, description="Chunks passed to LLM context window.")
    RERANKER_SCORE_THRESHOLD: float = Field(default=0.25, description="Minimum relevance score threshold.")

    QDRANT_LOCAL_PATH: str = Field(default="./qdrant_storage", description="Path for Qdrant embedded storage.")
    QDRANT_COLLECTION_NAME: str = Field(default="invariable_docs_v1", description="Default vector collection.")

    # --------------------------------------------------------------------------
    # Chunking & Retrieval Parameters
    # --------------------------------------------------------------------------
    CHUNKING_STRATEGY: Literal["recursive", "semantic"] = Field(default="recursive")
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=64)
    MIN_CHUNK_SIZE: int = Field(default=50)
    SEMANTIC_CHUNK_THRESHOLD: float = Field(default=0.55)

    RETRIEVAL_TOP_K: int = Field(default=15)
    RRF_K: int = Field(default=60)
    HYBRID_DENSE_WEIGHT: float = Field(default=0.7)
    HYBRID_SPARSE_WEIGHT: float = Field(default=0.3)
    BM25_K1: float = Field(default=1.2)
    BM25_B: float = Field(default=0.75)

    # --------------------------------------------------------------------------
    # Cloud API Keys (Optional)
    # --------------------------------------------------------------------------
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

    COHERE_API_KEY: Optional[str] = None
    COHERE_RERANK_MODEL: str = "rerank-v3.5"

    QDRANT_CLOUD_URL: Optional[str] = None
    QDRANT_CLOUD_API_KEY: Optional[str] = None

    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None

    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of global settings."""
    return Settings()
