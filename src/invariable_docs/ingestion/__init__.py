"""
Invariable Docs Ingestion Layer.

Provides high-performance multi-modal document parsing, cleaning, semantic chunking,
and vector database ingestion pipelines.
"""

from invariable_docs.ingestion.parser import DocumentParser, ParsedPage
from invariable_docs.ingestion.cleaner import DocumentCleaner, CleanedPage
from invariable_docs.ingestion.chunkers import BaseChunker, RecursiveCharacterChunker, SemanticChunker
from invariable_docs.ingestion.pipeline import IngestionPipeline

__all__ = [
    "DocumentParser",
    "ParsedPage",
    "DocumentCleaner",
    "CleanedPage",
    "BaseChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "IngestionPipeline",
]
