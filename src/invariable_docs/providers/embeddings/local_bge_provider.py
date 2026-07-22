"""
Local BGE Embedding Provider with Apple Metal (MPS) Acceleration.

Implements `BaseEmbeddingProvider` to run dense asymmetric text embeddings
locally on Apple Silicon neural engines using `BAAI/bge-large-en-v1.5` or compatible models.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from invariable_docs.providers.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class LocalBGEProvider(BaseEmbeddingProvider):
    """
    Local BGE Bi-Encoder Embedding Provider.
    
    Automatically leverages Apple Silicon `mps` acceleration when available,
    and handles asymmetric instruction prefixing (`Represent this sentence...`) for queries vs documents.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
    ):
        """
        Initialize the local BGE embedding model.
        
        Args:
            model_name: HuggingFace model hub ID or local path.
            device: Target execution device (`mps`, `cpu`, `cuda`). Defaults to `mps` if available.
            normalize_embeddings: Whether to L2-normalize vectors (critical for cosine similarity).
            query_instruction: Instruction prefix required for BGE query inputs.
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction = query_instruction

        # Determine target device
        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    logger.info("Apple Metal Performance Shaders (MPS) detected. Enabling hardware acceleration.")
                elif torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Use get_embedding_dimension to avoid FutureWarning
            self._dim = self.model.get_embedding_dimension()
            logger.info(f"Successfully initialized BGE model '{self.model_name}' (dim: {self._dim}, device: {self.device}).")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}", exc_info=True)
            raise

    @property
    def dimension(self) -> int:
        """Return the output dimension size of the embedding model."""
        return self._dim

    def embed_text(self, text: str, input_type: str = "query") -> List[float]:
        """
        Embed a single text string into a dense vector.
        
        Args:
            text: Text to embed.
            input_type: "query" (applies query instruction prefix) or "document".
            
        Returns:
            1D list of floating point numbers representing the vector embedding.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        processed_text = text.strip()
        if input_type.lower() == "query" and self.query_instruction:
            processed_text = f"{self.query_instruction}{processed_text}"

        embedding = self.model.encode(
            processed_text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """
        Embed a batch of strings into a list of vectors.
        
        Args:
            texts: List of strings to embed.
            input_type: "query" or "document" (defaults to "document" during ingestion).
            
        Returns:
            2D list of floating point vector embeddings.
        """
        if not texts:
            return []

        processed_texts = [t.strip() for t in texts]
        if input_type.lower() == "query" and self.query_instruction:
            processed_texts = [f"{self.query_instruction}{t}" for t in processed_texts]

        embeddings = self.model.encode(
            processed_texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )
        return [vec.tolist() for vec in embeddings]
