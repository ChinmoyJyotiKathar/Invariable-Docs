"""
BM25 Sparse Keyword Search Index.

Provides exact-match keyword search capabilities using the Okapi BM25 algorithm.
Used in parallel with Dense Vector search for Hybrid Retrieval.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi
import tiktoken
from invariable_docs.providers.base import RetrievedChunk

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Stateful BM25 index utilizing `rank_bm25` and `tiktoken`.
    Supports persisting the index to disk for local V1 setups.
    """

    def __init__(
        self,
        persist_path: str = "./qdrant_storage/bm25_index.pkl",
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """
        Initialize the BM25 Index.

        Args:
            persist_path: Local filesystem path to save/load the pickled index.
            k1: BM25 term frequency saturation parameter.
            b: BM25 length normalization parameter.
        """
        self.persist_path = Path(persist_path)
        self.k1 = k1
        self.b = b
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Internal state
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_chunks: List[RetrievedChunk] = []

    def _tokenize(self, text: str) -> List[str]:
        """Convert text into tokens using tiktoken and decode back to strings."""
        if not text:
            return []
        # Convert integer tokens back to string representation for BM25 processing
        tokens = self.encoder.encode(text)
        return [str(token) for token in tokens]

    def build_index(self, chunks: List[RetrievedChunk]) -> None:
        """
        Build the BM25 index from scratch using a corpus of chunks.
        Note: rank_bm25 does not support incremental upserts easily.
        """
        if not chunks:
            logger.warning("Empty chunk list provided to BM25 index builder.")
            return

        logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        self.corpus_chunks = chunks
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index successfully built.")

    def search(self, query: str, top_k: int = 15) -> List[RetrievedChunk]:
        """
        Search the BM25 index for the top_k matching chunks.
        Returns a list of RetrievedChunk objects with BM25 scores attached.
        """
        if not self.bm25 or not self.corpus_chunks:
            logger.warning("BM25 index is empty or not built.")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0.0:
                continue
                
            original_chunk = self.corpus_chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=original_chunk.chunk_id,
                    text=original_chunk.text,
                    score=float(score),
                    metadata=original_chunk.metadata,
                )
            )
            
        return results

    def save(self) -> None:
        """Persist the BM25 index and chunk corpus to disk."""
        if not self.bm25:
            return
            
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "bm25": self.bm25,
            "corpus_chunks": self.corpus_chunks,
            "k1": self.k1,
            "b": self.b,
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"Saved BM25 index to {self.persist_path}")

    def load(self) -> bool:
        """Load the BM25 index from disk if it exists."""
        if not self.persist_path.exists():
            return False
            
        try:
            with open(self.persist_path, "rb") as f:
                state = pickle.load(f)
            self.bm25 = state["bm25"]
            self.corpus_chunks = state["corpus_chunks"]
            logger.info(f"Loaded BM25 index containing {len(self.corpus_chunks)} chunks from disk.")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {self.persist_path}: {e}")
            return False
