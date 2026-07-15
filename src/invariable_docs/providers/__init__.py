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

__all__ = [
    "BaseLLMProvider",
    "BaseEmbeddingProvider",
    "BaseRerankerProvider",
    "BaseVectorStoreProvider",
    "BaseObservabilityProvider",
    "ChunkMetadata",
    "RetrievedChunk",
]
