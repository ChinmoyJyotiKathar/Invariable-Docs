"""
Ingestion Pipeline Orchestrator for Invariable Docs.

Connects document parsing, structural cleaning, chunking, dense embedding,
and sparse indexing into an automated end-to-end vector database ingestion workflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union
from invariable_docs.ingestion.parser import DocumentParser
from invariable_docs.ingestion.cleaner import DocumentCleaner
from invariable_docs.ingestion.chunkers import BaseChunker, RecursiveCharacterChunker
from invariable_docs.providers.base import BaseEmbeddingProvider, BaseVectorStoreProvider, RetrievedChunk

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-End ingestion workflow orchestrator.
    
    Transforms raw PDF files into normalized chunks, computes dense embeddings
    and sparse keyword maps, and persists the composite records inside the vector store.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store_provider: BaseVectorStoreProvider,
        chunker: Optional[BaseChunker] = None,
        extract_tables: bool = True,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store_provider = vector_store_provider
        self.parser = DocumentParser(extract_tables=extract_tables)
        self.cleaner = DocumentCleaner()
        self.chunker = chunker or RecursiveCharacterChunker()

    def ingest_file(
        self,
        file_path: Union[Path, str],
        doc_id: Optional[str] = None,
        default_date: Optional[str] = None,
        batch_size: int = 64,
    ) -> int:
        """
        Ingest a single document file from disk into the vector database index.
        
        Args:
            file_path: Path to the target PDF/document.
            doc_id: Canonical document name or ID (defaults to filename).
            default_date: Optional document publication date.
            batch_size: Batch size for embedding and vector store upsertion.
            
        Returns:
            Total number of chunks successfully indexed.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        canonical_id = doc_id or path.name
        logger.info(f"Starting ingestion workflow for document: '{canonical_id}' ({path})")

        # 1. Parse document into pages
        parsed_pages = self.parser.parse_pdf(path)
        logger.debug(f"Parsed {len(parsed_pages)} raw pages from {canonical_id}")

        # 2. Clean and enrich with hierarchical metadata
        cleaned_pages = self.cleaner.clean_pages(parsed_pages, canonical_id, default_date=default_date)

        # 3. Split into bounded chunks
        chunks = self.chunker.chunk_pages(cleaned_pages)
        if not chunks:
            logger.warning(f"No valid chunks produced from document '{canonical_id}'. Ingestion complete.")
            return 0

        # 4. Process embeddings and upserts in batches
        total_indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = [c.text for c in batch]

            logger.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} chunks) using {self.embedding_provider.__class__.__name__}...")
            embeddings = self.embedding_provider.embed_batch(batch_texts, input_type="document")

            logger.info(f"Upserting batch {i // batch_size + 1} into {self.vector_store_provider.__class__.__name__}...")
            self.vector_store_provider.upsert_chunks(batch, embeddings)
            total_indexed += len(batch)

        logger.info(f"Successfully ingested and indexed {total_indexed} chunks from document '{canonical_id}'.")
        return total_indexed

    def ingest_directory(
        self,
        directory_path: Union[Path, str],
        file_extensions: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> int:
        """
        Batch ingest all matching document files inside a directory.
        """
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        extensions = file_extensions or [".pdf", ".txt"]
        files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        logger.info(f"Found {len(files)} matching files in directory '{dir_path}'.")

        total_chunks = 0
        for file in files:
            try:
                chunks_indexed = self.ingest_file(file, batch_size=batch_size)
                total_chunks += chunks_indexed
            except Exception as e:
                logger.error(f"Failed to ingest document '{file.name}': {e}", exc_info=True)

        return total_chunks
