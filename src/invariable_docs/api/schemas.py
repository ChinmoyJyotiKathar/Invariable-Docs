"""
Pydantic Schemas for the FastAPI REST interface.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Ingestion Schemas
# -----------------------------------------------------------------------------
class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the local PDF file to ingest.")
    collection_name: str = Field(default="invariable_docs", description="Target vector store collection.")
    extract_tables: bool = Field(default=True, description="Whether to attempt tabular data extraction.")


class IngestResponse(BaseModel):
    status: str = Field(..., description="Status of the ingestion process.")
    doc_id: str = Field(..., description="The unique document ID generated for this file.")
    chunks_processed: int = Field(..., description="Number of chunks successfully indexed.")
    duration_sec: float = Field(..., description="Time taken to process and index the document.")


# -----------------------------------------------------------------------------
# Query & Retrieval Schemas
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="User's natural language question.")
    top_k: int = Field(default=15, description="Number of initial chunks to retrieve per Dense/Sparse branch.")
    final_top_n: int = Field(default=4, description="Final number of chunks to return after Re-ranking.")
    use_hyde: bool = Field(default=True, description="Whether to use Hypothetical Document Embeddings for dense search.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters to apply to the vector store.")


class ChunkSchema(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The grounded generated answer.")
    retrieved_chunks: List[ChunkSchema] = Field(..., description="The final re-ranked context chunks used.")
    latency_sec: float = Field(..., description="Total latency for the retrieval and generation pipeline.")


# -----------------------------------------------------------------------------
# Evaluation Schemas
# -----------------------------------------------------------------------------
class EvalRequest(BaseModel):
    dataset_path: str = Field(
        default="src/invariable_docs/eval/golden_dataset.json",
        description="Path to the golden dataset."
    )


class EvalResponse(BaseModel):
    status: str
    passed: bool
    aggregate_scores: Dict[str, float]
    report_path: Optional[str] = None
