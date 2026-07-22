"""
Hybrid Retrieval and Fusion Engine.

Orchestrates simultaneous Dense (Vector) and Sparse (BM25) searches,
merges results using Reciprocal Rank Fusion (RRF), and applies 
a second-stage Cross-Encoder Re-ranker for ultimate precision.
"""

from __future__ import annotations

import logging
import concurrent.futures
from typing import Dict, List, Optional
from invariable_docs.providers.base import (
    BaseEmbeddingProvider,
    BaseRerankerProvider,
    BaseVectorStoreProvider,
    RetrievedChunk,
)
from invariable_docs.retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class HybridRetrievalEngine:
    """
    Executes advanced Hybrid RAG Retrieval (Dense + Sparse + RRF + Re-rank).
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store_provider: BaseVectorStoreProvider,
        reranker_provider: Optional[BaseRerankerProvider] = None,
        bm25_index: Optional[BM25Index] = None,
        collection_name: str = "invariable_docs",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store_provider = vector_store_provider
        self.reranker_provider = reranker_provider
        self.bm25_index = bm25_index
        self.collection_name = collection_name
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        final_top_n: int = 4,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_hyde_document: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute the full hybrid retrieval pipeline.
        
        Args:
            query: The user's search query (used for BM25 and Re-ranking).
            top_k: Number of chunks to retrieve per search branch (Dense & Sparse).
            final_top_n: Final number of chunks to return after Cross-Encoder re-ranking.
            metadata_filters: Pre-filtering dict for the vector DB (e.g. {"is_latest": True}).
            use_hyde_document: Optional HyDE document to use instead of the query for dense embedding.
        """
        dense_results: List[RetrievedChunk] = []
        sparse_results: List[RetrievedChunk] = []

        # Target text for Dense embedding (either original query or HyDE doc)
        dense_query_target = use_hyde_document if use_hyde_document else query

        # 1. Execute Dense and Sparse searches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(
                self._execute_dense, dense_query_target, top_k, metadata_filters
            )
            future_sparse = executor.submit(
                self._execute_sparse, query, top_k
            )

            dense_results = future_dense.result()
            sparse_results = future_sparse.result()

        # 2. Merge using Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        
        # If no re-ranker is configured, return the top N from RRF directly
        if not self.reranker_provider:
            logger.debug("No cross-encoder configured. Returning raw RRF results.")
            return fused_results[:final_top_n]

        # 3. Second-stage Cross-Encoder Re-ranking
        logger.info(f"Re-ranking {len(fused_results)} fused candidates using Cross-Encoder...")
        reranked_results = self.reranker_provider.rerank(
            query=query,
            chunks=fused_results,
            top_n=final_top_n,
            score_threshold=0.25,
        )
        
        return reranked_results

    def _execute_dense(
        self, target_text: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[RetrievedChunk]:
        """Execute dense vector search."""
        logger.debug(f"Executing Dense Search for: '{target_text[:50]}...'")
        query_embedding = self.embedding_provider.embed_text(target_text, input_type="query")
        return self.vector_store_provider.search_dense(
            collection_name=self.collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filters=filters,
        )

    def _execute_sparse(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Execute sparse BM25 search."""
        if not self.bm25_index:
            logger.debug("BM25 index not provided. Skipping sparse search.")
            return []
            
        logger.debug(f"Executing Sparse BM25 Search for: '{query}'")
        return self.bm25_index.search(query, top_k=top_k)

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievedChunk],
        sparse_results: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Merge multiple ranked lists using the RRF algorithm.
        Score = Weight * (1 / (rrf_k + Rank))
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # Process Dense rankings
        for rank, chunk in enumerate(dense_results):
            rrf_score = self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + rrf_score
            chunk_map[chunk.chunk_id] = chunk

        # Process Sparse rankings
        for rank, chunk in enumerate(sparse_results):
            rrf_score = self.sparse_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + rrf_score
            chunk_map[chunk.chunk_id] = chunk

        # Sort combined results by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        fused_chunks = []
        for chunk_id in sorted_ids:
            chunk = chunk_map[chunk_id]
            # Override the score with the new RRF score for observability
            new_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=rrf_scores[chunk_id],
                metadata=chunk.metadata,
            )
            fused_chunks.append(new_chunk)

        logger.debug(f"RRF Fusion complete. Combined {len(dense_results)} Dense and {len(sparse_results)} Sparse into {len(fused_chunks)} unique chunks.")
        return fused_chunks
