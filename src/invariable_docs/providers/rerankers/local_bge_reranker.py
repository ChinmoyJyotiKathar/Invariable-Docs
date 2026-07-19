"""
Local BGE Reranker Provider Module.

Implements `BaseRerankerProvider` for cross-encoder re-ranking of retrieved chunks
using `BAAI/bge-reranker-v2-m3` or similar models via the FlagEmbedding library.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from invariable_docs.providers.base import BaseRerankerProvider, RetrievedChunk

logger = logging.getLogger(__name__)


class LocalRerankerProvider(BaseRerankerProvider):
    """
    Local Cross-Encoder Re-ranker via FlagEmbedding.
    
    Scores (query, chunk_text) pairs using a transformer cross-encoder and 
    sorts/filters candidates based on precise relevance scores.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the local cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model hub ID or local path.
            use_fp16: Whether to use half-precision for faster inference (Apple MPS supports FP16).
            device: Target execution device (`mps`, `cpu`, `cuda`). Defaults to `mps` if available.
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16

        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    logger.info("Apple Metal Performance Shaders (MPS) detected for Re-ranker.")
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading re-ranker model '{self.model_name}' on device '{self.device}'...")
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(self.model_name, use_fp16=self.use_fp16, device=self.device)
            logger.info(f"Successfully initialized BGE Re-ranker '{self.model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load re-ranker model '{self.model_name}': {e}", exc_info=True)
            raise

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: int = 4,
        score_threshold: float = 0.25,
    ) -> List[RetrievedChunk]:
        """
        Re-score and filter candidate chunks using the cross-encoder.
        """
        if not chunks:
            return []

        # Construct (query, text) pairs for the cross-encoder
        sentence_pairs = [[query, chunk.text] for chunk in chunks]
        
        # Compute exact relevance scores
        scores = self.reranker.compute_score(sentence_pairs, normalize=True)
        
        # FlagEmbedding might return a single float if there's only 1 pair
        if isinstance(scores, float):
            scores = [scores]

        # Update scores and filter
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            # We create a new chunk object with the updated score to avoid mutating the original
            new_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                metadata=chunk.metadata,
            )
            if new_chunk.score >= score_threshold:
                scored_chunks.append(new_chunk)

        # Sort by score descending and return top_n
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_n]
