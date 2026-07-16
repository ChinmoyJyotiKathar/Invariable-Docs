"""
Document Parsing Module for Invariable Docs Ingestion Pipeline.

Extracts raw character sequences, page numbers, and table layouts from PDF files
using `PyMuPDF (fitz)` as primary engine and `pdfplumber` as structured table fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ParsedPage(BaseModel):
    """Represents a single parsed page from a document."""
    page_no: int = Field(..., description="1-indexed page number within the source document.")
    text: str = Field(..., description="Raw extracted text passage from the page.")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Structured table data extracted via pdfplumber.")


class DocumentParser:
    """
    High-performance multi-modal document parser.
    
    Uses PyMuPDF (fitz) for rapid text extraction across high-volume corpuses,
    and falls back to pdfplumber when tabular layouts require cell boundary preservation.
    """

    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables

    def parse_pdf(self, file_path: Union[Path, str]) -> List[ParsedPage]:
        """
        Parse a PDF file into a sequence of ParsedPage objects containing text and tables.
        
        Args:
            file_path: Path to the target PDF document.
            
        Returns:
            List of ParsedPage instances.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source document not found at: {path}")

        parsed_pages: List[ParsedPage] = []

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required for document parsing. Install via `pip install pymupdf`.")

        doc = fitz.open(path)
        logger.info(f"Opened PDF document '{path.name}' with {len(doc)} pages.")

        for page_idx in range(len(doc)):
            page_no = page_idx + 1
            page = doc.load_page(page_idx)
            raw_text = page.get_text("text").strip()

            tables: List[Dict[str, Any]] = []
            if self.extract_tables:
                tables = self._extract_tables_from_page(path, page_no)

            parsed_pages.append(
                ParsedPage(
                    page_no=page_no,
                    text=raw_text,
                    tables=tables,
                )
            )

        doc.close()
        return parsed_pages

    def _extract_tables_from_page(self, file_path: Path, page_no: int) -> List[Dict[str, Any]]:
        """Extract structured table data using pdfplumber for a specific 1-indexed page."""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed; skipping structured table extraction.")
            return []

        extracted_tables: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_no <= len(pdf.pages):
                    page = pdf.pages[page_no - 1]
                    raw_tables = page.extract_tables()
                    for table_idx, raw_table in enumerate(raw_tables):
                        if not raw_table or len(raw_table) < 2:
                            continue
                        headers = [str(col).strip() if col else f"col_{i}" for i, col in enumerate(raw_table[0])]
                        rows = []
                        for row in raw_table[1:]:
                            rows.append({headers[i]: (str(cell).strip() if cell else "") for i, cell in enumerate(row) if i < len(headers)})
                        
                        extracted_tables.append({
                            "table_id": f"table_p{page_no}_{table_idx + 1}",
                            "headers": headers,
                            "rows": rows,
                        })
        except Exception as e:
            logger.debug(f"Could not extract tables from page {page_no} of {file_path.name}: {e}")

        return extracted_tables
