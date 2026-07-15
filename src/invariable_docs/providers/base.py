"""
Base Protocol definitions for Invariable Docs Provider Abstraction Layer.

This module guarantees that switching between V1 Local (Ollama, BGE-Large, Qdrant Local)
and enterprise cloud (OpenAI, Cohere, Pinecone, Langfuse) requires zero changes to core pipeline logic.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk ingested into the vector store."""
    doc_id: str = Field(..., description="Unique identifier or filename of the source document.")
    page_no: int = Field(..., description="1-indexed page number in the source PDF/document.")
    section_header: Optional[str] = Field(None, description="Section heading under which this chunk falls.")
    doc_date: Optional[str] = Field(None, description="Document creation or publication date.")
    chunk_index: int = Field(..., description="Sequential index of the chunk within the document.")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Additional domain-specific metadata.")


class RetrievedChunk(BaseModel):
    """A single document passage retrieved or re-ranked during search."""
    chunk_id: str = Field(..., description="Unique ID of the chunk.")
    text: str = Field(..., description="Raw text passage.")
    score: float = Field(..., description="Relevance score (cosine, BM25, RRF, or cross-encoder score).")
    metadata: ChunkMetadata = Field(..., description="Associated document metadata.")


# ==============================================================================
# 1. LLM PROVIDER PROTOCOL
# ==============================================================================
class BaseLLMProvider(ABC):
    """Abstract interface for Language Model generation."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.90,
        max_tokens: int = 768,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a response given a prompt and optional system instructions."""
        pass

    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.90,
        max_tokens: int = 768,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Asynchronous generation for high-concurrency API requests."""
        pass


# ==============================================================================
# 2. EMBEDDING PROVIDER PROTOCOL
# ==============================================================================
class BaseEmbeddingProvider(ABC):
    """Abstract interface for text embedding generation."""

    @abstractmethod
    def embed_text(self, text: str, input_type: str = "query") -> List[float]:
        """
        Embed a single text string into a dense vector.
        
        Args:
            text: Text to embed.
            input_type: "query" for user queries or "document" for ingestion.
                        Critical for asymmetric embedding models like BGE and E5.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """Embed a batch of strings into a list of vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the output dimension size of the embedding model."""
        pass


# ==============================================================================
# 3. RE-RANKER PROVIDER PROTOCOL
# ==============================================================================
class BaseRerankerProvider(ABC):
    """Abstract interface for second-stage cross-encoder re-ranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: int = 4,
        score_threshold: float = 0.25,
    ) -> List[RetrievedChunk]:
        """
        Re-score and filter a list of candidate chunks against the query.
        
        Args:
            query: User search query.
            chunks: Candidate chunks retrieved from hybrid search.
            top_n: Maximum number of chunks to return after re-ranking.
            score_threshold: Minimum cross-encoder score required to retain a chunk.
        """
        pass


# ==============================================================================
# 4. VECTOR STORE PROVIDER PROTOCOL
# ==============================================================================
class BaseVectorStoreProvider(ABC):
    """Abstract interface for Vector Database indexing and dense search."""

    @abstractmethod
    def ensure_collection(self, collection_name: str, dimension: int) -> None:
        """Create the collection if it does not already exist."""
        pass

    @abstractmethod
    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[RetrievedChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Insert or update chunks with their dense embeddings and metadata."""
        pass

    @abstractmethod
    def search_dense(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 15,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Perform dense vector similarity search (cosine distance)."""
        pass

    @abstractmethod
    def delete_document(self, collection_name: str, doc_id: str) -> int:
        """Delete all chunks associated with a specific document ID."""
        pass


# ==============================================================================
# 5. OBSERVABILITY & TRACING PROTOCOL
# ==============================================================================
class BaseObservabilityProvider(ABC):
    """Abstract interface for telemetry, tracing, and cost aggregation."""

    @abstractmethod
    def start_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Start a top-level workflow trace."""
        pass

    @abstractmethod
    def log_generation(
        self,
        trace_id: Any,
        prompt: str,
        output: str,
        model: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Log an LLM generation step along with latency and token usage."""
        pass

    @abstractmethod
    def end_trace(self, trace_id: Any, status: str = "SUCCESS", error: Optional[str] = None) -> None:
        """End a trace and flush to the sink."""
        pass
