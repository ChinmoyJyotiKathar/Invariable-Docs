"""
Provider Factory Engine.

Dynamically instantiates provider interfaces based on the system configuration (`config.py`).
Allows seamless plug-and-play swapping between local and cloud providers without changing application logic.
"""

from __future__ import annotations

import logging
from invariable_docs.config import get_settings
from invariable_docs.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseRerankerProvider,
    BaseVectorStoreProvider,
)
from invariable_docs.providers.embeddings.local_bge_provider import LocalBGEProvider
from invariable_docs.providers.llm.ollama_provider import OllamaLLMProvider
from invariable_docs.providers.rerankers.local_bge_reranker import LocalRerankerProvider
from invariable_docs.providers.vector_stores.qdrant_provider import QdrantProvider

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for initializing configured AI providers."""

    @classmethod
    def get_llm_provider(cls) -> BaseLLMProvider:
        """Instantiate the configured LLM provider."""
        settings = get_settings()
        
        if settings.LLM_PROVIDER == "ollama":
            return OllamaLLMProvider(
                model_name=settings.OLLAMA_MODEL,
                host=settings.OLLAMA_BASE_URL,
            )
        elif settings.LLM_PROVIDER == "litellm":
            from invariable_docs.providers.llm.litellm_provider import LiteLLMProvider
            return LiteLLMProvider(
                model_name=settings.LLM_MODEL_NAME,
            )
        else:
            raise NotImplementedError(f"LLM provider '{settings.LLM_PROVIDER}' is not yet implemented.")

    @classmethod
    def get_embedding_provider(cls) -> BaseEmbeddingProvider:
        """Instantiate the configured Dense Embedding provider."""
        settings = get_settings()
        
        if settings.EMBEDDING_PROVIDER == "local_bge":
            return LocalBGEProvider(
                model_name=settings.LOCAL_EMBEDDING_MODEL,
                device=settings.LOCAL_EMBEDDING_DEVICE,
            )
        else:
            raise NotImplementedError(f"Embedding provider '{settings.EMBEDDING_PROVIDER}' is not yet implemented.")

    @classmethod
    def get_reranker_provider(cls) -> BaseRerankerProvider:
        """Instantiate the configured Cross-Encoder Re-ranker provider."""
        settings = get_settings()
        
        if settings.RERANKER_PROVIDER == "local_bge":
            return LocalRerankerProvider(
                model_name=settings.LOCAL_RERANKER_MODEL,
                device=settings.LOCAL_RERANKER_DEVICE,
            )
        elif settings.RERANKER_PROVIDER == "none":
            # Can return a dummy pass-through provider if needed, 
            # or the calling layer checks for None.
            raise ValueError("RERANKER_PROVIDER is set to 'none'. Handle this at the application layer.")
        else:
            raise NotImplementedError(f"Re-ranker provider '{settings.RERANKER_PROVIDER}' is not yet implemented.")

    @classmethod
    def get_vector_store_provider(cls) -> BaseVectorStoreProvider:
        """Instantiate the configured Vector Database provider."""
        settings = get_settings()
        
        if settings.VECTOR_STORE_PROVIDER == "qdrant_local":
            return QdrantProvider(
                path=settings.QDRANT_LOCAL_PATH,
            )
        elif settings.VECTOR_STORE_PROVIDER == "qdrant_cloud":
            if not settings.QDRANT_CLOUD_URL or not settings.QDRANT_CLOUD_API_KEY:
                raise ValueError("Qdrant cloud URL and API key must be set in .env")
            return QdrantProvider(
                url=settings.QDRANT_CLOUD_URL,
                api_key=settings.QDRANT_CLOUD_API_KEY,
            )
        else:
            raise NotImplementedError(f"Vector Store provider '{settings.VECTOR_STORE_PROVIDER}' is not yet implemented.")
