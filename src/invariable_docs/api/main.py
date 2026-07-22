"""
FastAPI Entry Point for Invariable Docs.
"""

from fastapi import FastAPI
from invariable_docs.config import get_settings

app = FastAPI(
    title="Invariable Docs API",
    description="Enterprise-grade Hybrid-Search RAG Knowledge Assistant",
    version="0.1.0",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("invariable_docs.api.main:app", host="0.0.0.0", port=8000, reload=True)
