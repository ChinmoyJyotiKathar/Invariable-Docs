"""
Chunking Engine Module for Invariable Docs.

Implements both baseline `RecursiveCharacterChunker` and advanced `SemanticChunker`
to transform cleaned document pages into bounded, semantically coherent `RetrievedChunk` payloads.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from invariable_docs.ingestion.cleaner import CleanedPage
from invariable_docs.providers.base import BaseEmbeddingProvider, ChunkMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract interface for document chunking strategies."""

    @abstractmethod
    def chunk_pages(self, pages: List[CleanedPage]) -> List[RetrievedChunk]:
        """Split a list of cleaned document pages into discrete RetrievedChunk instances."""
        pass


class RecursiveCharacterChunker(BaseChunker):
    """
    Splits document text sequentially across natural delimiters (`\\n\\n`, `. `, ` `)
    while enforcing strict token boundary limits and overlap redundancy.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_pages(self, pages: List[CleanedPage]) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        global_chunk_idx = 0

        for page in pages:
            # First, append table text blocks as self-contained chunks if any exist
            for tbl_text in page.tables_text:
                if len(tbl_text.strip()) >= self.min_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            text=tbl_text.strip(),
                            page=page,
                            chunk_index=global_chunk_idx,
                            is_table=True,
                        )
                    )
                    global_chunk_idx += 1

            # Split primary prose text using recursive character splitting
            raw_text = page.text
            if not raw_text:
                continue

            page_chunks = self._split_text_recursive(raw_text, self.separators)
            for chunk_text in page_chunks:
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunks.append(
                        self._create_chunk(
                            text=chunk_text.strip(),
                            page=page,
                            chunk_index=global_chunk_idx,
                        )
                    )
                    global_chunk_idx += 1

        logger.info(f"RecursiveCharacterChunker generated {len(chunks)} chunks across {len(pages)} pages.")
        return chunks

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchy of separators until sizes fit chunk_size."""
        final_chunks: List[str] = []
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            split_len = len(split) + (len(separator) if current_chunk else 0)
            if current_length + split_len > self.chunk_size and current_chunk:
                merged = separator.join(current_chunk)
                final_chunks.append(merged)
                
                # Retain overlap from trailing splits
                overlap_len = 0
                overlap_chunk: List[str] = []
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + len(separator)
                    else:
                        break
                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(split)
            current_length += split_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks

    def _create_chunk(self, text: str, page: CleanedPage, chunk_index: int, is_table: bool = False) -> RetrievedChunk:
        """Helper to construct a validated RetrievedChunk with metadata payload."""
        chunk_id = f"{page.doc_id}_p{page.page_no}_c{chunk_index}"
        meta = ChunkMetadata(
            doc_id=page.doc_id,
            page_no=page.page_no,
            section_header=page.section_header,
            doc_date=page.doc_date,
            chunk_index=chunk_index,
            custom_fields={"is_table": is_table},
        )
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            score=0.0,  # Unscored at ingestion time
            metadata=meta,
        )


class SemanticChunker(BaseChunker):
    """
    Advanced semantic boundary chunker.
    
    Embeds consecutive sentences and splits where cosine similarity drops below
    the `semantic_threshold`, isolating discrete topic transitions.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        semantic_threshold: float = 0.55,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1536,
    ):
        self.embedding_provider = embedding_provider
        self.semantic_threshold = semantic_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_pages(self, pages: List[CleanedPage]) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        global_chunk_idx = 0

        for page in pages:
            # First, append tables as independent blocks
            for tbl_text in page.tables_text:
                if len(tbl_text.strip()) >= self.min_chunk_size:
                    chunks.append(self._build_chunk(tbl_text.strip(), page, global_chunk_idx, is_table=True))
                    global_chunk_idx += 1

            # Split prose text into sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", page.text) if len(s.strip()) > 10]
            if not sentences:
                continue
            if len(sentences) == 1:
                chunks.append(self._build_chunk(sentences[0], page, global_chunk_idx))
                global_chunk_idx += 1
                continue

            # Embed all sentences to evaluate pairwise semantic shifts
            embeddings = self.embedding_provider.embed_batch(sentences, input_type="document")
            
            current_sentences: List[str] = [sentences[0]]
            current_len = len(sentences[0])

            for i in range(len(sentences) - 1):
                vec1 = np.array(embeddings[i])
                vec2 = np.array(embeddings[i + 1])
                cosine_sim = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-9))

                # If cosine similarity drops below threshold or max length exceeded -> split
                if cosine_sim < self.semantic_threshold or (current_len + len(sentences[i + 1])) > self.max_chunk_size:
                    merged_text = " ".join(current_sentences)
                    if len(merged_text) >= self.min_chunk_size:
                        chunks.append(self._build_chunk(merged_text, page, global_chunk_idx))
                        global_chunk_idx += 1
                    current_sentences = [sentences[i + 1]]
                    current_len = len(sentences[i + 1])
                else:
                    current_sentences.append(sentences[i + 1])
                    current_len += len(sentences[i + 1])

            if current_sentences:
                merged_text = " ".join(current_sentences)
                if len(merged_text) >= self.min_chunk_size:
                    chunks.append(self._build_chunk(merged_text, page, global_chunk_idx))
                    global_chunk_idx += 1

        logger.info(f"SemanticChunker generated {len(chunks)} chunks across {len(pages)} pages.")
        return chunks

    def _build_chunk(self, text: str, page: CleanedPage, chunk_index: int, is_table: bool = False) -> RetrievedChunk:
        """Helper to construct RetrievedChunk payload."""
        chunk_id = f"{page.doc_id}_p{page.page_no}_c{chunk_index}"
        meta = ChunkMetadata(
            doc_id=page.doc_id,
            page_no=page.page_no,
            section_header=page.section_header,
            doc_date=page.doc_date,
            chunk_index=chunk_index,
            custom_fields={"is_table": is_table, "chunk_strategy": "semantic"},
        )
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            score=0.0,
            metadata=meta,
        )
