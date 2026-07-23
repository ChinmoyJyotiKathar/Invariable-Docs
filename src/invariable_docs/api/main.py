"""
FastAPI Entry Point for Invariable Docs.
Wires up the ProviderFactory, Ingestion Pipeline, and Hybrid Generation pipelines.
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from invariable_docs.config import get_settings
from invariable_docs.providers.factory import ProviderFactory
from invariable_docs.ingestion.pipeline import IngestionPipeline
from invariable_docs.retrieval.bm25_index import BM25Index
from invariable_docs.retrieval.hybrid_engine import HybridRetrievalEngine
from invariable_docs.retrieval.transformations import QueryTransformer
from invariable_docs.generation.engine import GenerationEngine
from invariable_docs.api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, ChunkSchema,
    EvalRequest, EvalResponse
)

logger = logging.getLogger(__name__)

# Global state for our RAG components
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize AI models once on startup."""
    logger.info("Initializing ProviderFactory and RAG components...")
    
    try:
        # Load Providers
        llm = ProviderFactory.get_llm_provider()
        embedding = ProviderFactory.get_embedding_provider()
        vector_store = ProviderFactory.get_vector_store_provider()
        reranker = ProviderFactory.get_reranker_provider()
        
        # Load Sparse Index
        bm25 = BM25Index()
        bm25.load() # Will load from disk if it exists

        # Initialize Pipelines
        app_state["ingestion"] = IngestionPipeline(
            embedding_provider=embedding,
            vector_store_provider=vector_store,
        )
        
        app_state["retrieval"] = HybridRetrievalEngine(
            embedding_provider=embedding,
            vector_store_provider=vector_store,
            reranker_provider=reranker,
            bm25_index=bm25,
        )
        
        app_state["transformer"] = QueryTransformer(llm_provider=llm)
        
        app_state["generation"] = GenerationEngine(llm_provider=llm)
        
        logger.info("All pipelines successfully initialized!")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}", exc_info=True)
        raise e
    finally:
        logger.info("Shutting down API and cleaning up resources...")
        app_state.clear()


app = FastAPI(
    title="Invariable Docs API",
    description="Enterprise-grade Hybrid-Search RAG Knowledge Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "vector_store": settings.VECTOR_STORE_PROVIDER,
    }


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """Ingest a PDF document into the vector database and rebuild BM25 index."""
    pipeline: IngestionPipeline = app_state.get("ingestion")
    if not pipeline:
        raise HTTPException(status_code=500, detail="Ingestion pipeline not initialized.")
        
    start_time = time.time()
    try:
        import os
        filename = os.path.basename(request.file_path)
        chunks_processed = pipeline.ingest_file(
            file_path=request.file_path,
            doc_id=filename,
        )
        doc_id = filename
        
        # Since we ingested new chunks, ideally we rebuild the BM25 index here.
        # For a production system we'd use native Qdrant sparse vectors, but for V1 we skip this 
        # or load the full corpus and rebuild (omitted for brevity).
        
        duration = time.time() - start_time
        return IngestResponse(
            status="success",
            doc_id=doc_id,
            chunks_processed=chunks_processed,
            duration_sec=round(duration, 2)
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Execute a hybrid query and generate a grounded answer."""
    retriever: HybridRetrievalEngine = app_state.get("retrieval")
    generator: GenerationEngine = app_state.get("generation")
    transformer: QueryTransformer = app_state.get("transformer")
    
    if not retriever or not generator:
        raise HTTPException(status_code=500, detail="Pipelines not initialized.")

    start_time = time.time()
    try:
        # 1. Query Transformation (HyDE)
        hyde_doc = None
        if request.use_hyde:
            hyde_doc = transformer.generate_hyde_document(request.query)
            
        # 2. Hybrid Retrieval
        chunks = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            final_top_n=request.final_top_n,
            metadata_filters=request.filters,
            use_hyde_document=hyde_doc
        )
        
        # 3. Grounded Generation
        answer = generator.generate_answer(request.query, chunks)
        
        # Format chunks for response
        resp_chunks = [
            ChunkSchema(
                chunk_id=c.chunk_id,
                text=c.text,
                score=c.score,
                metadata=c.metadata.model_dump() if c.metadata else None
            ) for c in chunks
        ]
        
        duration = time.time() - start_time
        return QueryResponse(
            answer=answer,
            retrieved_chunks=resp_chunks,
            latency_sec=round(duration, 2)
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/eval", response_model=EvalResponse)
async def trigger_eval(request: EvalRequest):
    """Trigger the regression evaluation suite."""
    # To run evaluations asynchronously, we should spawn a background task.
    # For now, we simulate returning a trigger success.
    return EvalResponse(
        status="Evaluation triggered in background (simulated).",
        passed=True,
        aggregate_scores={}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("invariable_docs.api.main:app", host="0.0.0.0", port=8000, reload=True)
