"""
Document Cleaner and Metadata Enrichment Module.

Normalizes raw OCR/PDF text artifacts, cleans control characters,
and detects structural hierarchy (section headers, dates) for metadata tagging.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from invariable_docs.ingestion.parser import ParsedPage

logger = logging.getLogger(__name__)


class CleanedPage(BaseModel):
    """Enriched page payload containing normalized text and detected metadata hierarchy."""
    page_no: int = Field(..., description="1-indexed page number.")
    text: str = Field(..., description="Normalized text with clean spacing and encoding.")
    section_header: Optional[str] = Field(None, description="Detected section heading for this page.")
    doc_id: str = Field(..., description="Canonical source document identifier.")
    doc_date: Optional[str] = Field(None, description="Detected or configured document date.")
    tables_text: List[str] = Field(default_factory=list, description="Serialized table markdown strings.")


class DocumentCleaner:
    """
    Cleans structural noise and identifies metadata hierarchy from raw parsed pages.
    """

    # Common patterns for section headings in SEC filings and technical documentation
    HEADER_PATTERNS = [
        re.compile(r"^(Item\s+\d+[A-Z]?\.\s+[A-Z0-9\s,\-\.\'\(\)]+)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^([0-9]+\.[0-9]*\s+[A-Z][A-Z0-9\s,\-\.\'\(\)]+)", re.MULTILINE),
        re.compile(r"^([A-Z][A-Z\s]{4,60})$", re.MULTILINE),
    ]

    DATE_PATTERN = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b",
        re.IGNORECASE,
    )

    def clean_pages(
        self,
        pages: List[ParsedPage],
        doc_id: str,
        default_date: Optional[str] = None,
    ) -> List[CleanedPage]:
        """
        Normalize text and extract hierarchical metadata for each parsed page.
        
        Args:
            pages: List of raw ParsedPage objects from DocumentParser.
            doc_id: Canonical document identifier (e.g., filename).
            default_date: Optional default creation/publication date string.
            
        Returns:
            List of CleanedPage instances enriched with section headers.
        """
        cleaned_pages: List[CleanedPage] = []
        current_section: Optional[str] = None
        detected_date: Optional[str] = default_date

        for page in pages:
            normalized_text = self._normalize_text(page.text)
            
            # Attempt date extraction if not yet found
            if not detected_date and normalized_text:
                date_match = self.DATE_PATTERN.search(normalized_text[:1000])
                if date_match:
                    detected_date = date_match.group(0).strip()

            # Detect section heading updates
            new_section = self._detect_section_header(normalized_text)
            if new_section:
                current_section = new_section

            # Convert structured tables into readable markdown blocks for chunking
            tables_text: List[str] = []
            for tbl in page.tables:
                markdown_tbl = self._table_to_markdown(tbl)
                if markdown_tbl:
                    tables_text.append(markdown_tbl)

            cleaned_pages.append(
                CleanedPage(
                    page_no=page.page_no,
                    text=normalized_text,
                    section_header=current_section,
                    doc_id=doc_id,
                    doc_date=detected_date,
                    tables_text=tables_text,
                )
            )

        return cleaned_pages

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace, remove null bytes, and clean non-standard PDF ligatures."""
        if not text:
            return ""

        # Replace non-breaking spaces and ligatures
        text = text.replace("\u00a0", " ").replace("\ufb01", "fi").replace("\ufb02", "fl")
        # Remove null control characters
        text = text.replace("\x00", "")
        # Collapse excessive multiple line breaks (more than 3 into 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse trailing multiple spaces on lines
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _detect_section_header(self, text: str) -> Optional[str]:
        """Scan top of page text for major section headings."""
        lines = text.split("\n")[:10]  # Check first 10 lines of page
        for line in lines:
            line_str = line.strip()
            if not line_str or len(line_str) < 4 or len(line_str) > 100:
                continue
            for pattern in self.HEADER_PATTERNS:
                match = pattern.match(line_str)
                if match:
                    return match.group(1).strip()
        return None

    def _table_to_markdown(self, table_dict: Dict[str, Any]) -> str:
        """Serialize a structured table dictionary into a clean markdown string."""
        headers = table_dict.get("headers", [])
        rows = table_dict.get("rows", [])
        if not headers or not rows:
            return ""

        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        row_lines = []
        for row in rows:
            cells = [str(row.get(h, "")).strip() for h in headers]
            row_lines.append("| " + " | ".join(cells) + " |")

        return "\n".join([f"[Table: {table_dict.get('table_id', 'tbl')}]", header_line, sep_line] + row_lines)
