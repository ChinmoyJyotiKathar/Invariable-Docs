"""
Provider Abstraction Layer for Invariable Docs.

All external dependencies (LLMs, Embeddings, Re-Rankers, Vector DBs, Observability)
are accessed strictly through the abstract interfaces defined in this module.
"""

from invariable_docs.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseObservabilityProvider,
    BaseRerankerProvider,
    BaseVectorStoreProvider,
    ChunkMetadata,
    RetrievedChunk,
)
from invariable_docs.providers.llm.ollama_provider import OllamaLLMProvider
from invariable_docs.providers.rerankers.local_bge_reranker import LocalRerankerProvider
from invariable_docs.providers.factory import ProviderFactory

__all__ = [
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    "BaseRerankerProvider",
    "BaseVectorStoreProvider",
    "BaseObservabilityProvider",
    "ChunkMetadata",
    "RetrievedChunk",
    "OllamaLLMProvider",
    "LocalRerankerProvider",
    "ProviderFactory",
]
