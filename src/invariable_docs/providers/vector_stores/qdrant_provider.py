"""
Qdrant Vector Store Provider for Invariable Docs.

Implements `BaseVectorStoreProvider` protocol to manage dense collection indexing,
batch vector upserts, and cosine similarity search over local storage or cloud instances.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from invariable_docs.providers.base import BaseVectorStoreProvider, ChunkMetadata, RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantProvider(BaseVectorStoreProvider):
    """
    Qdrant Vector Database Provider.
    
    Supports local-first disk persistence (`path="qdrant_storage"`) as well as
    remote cloud connections (`url="https://...", api_key="..."`).
    """

    def __init__(
        self,
        path: Optional[str] = "qdrant_storage",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance_metric: rest.Distance = rest.Distance.COSINE,
    ):
        """
        Initialize Qdrant client.
        
        Args:
            path: Local filesystem directory path for embedded Qdrant persistence.
            url: Remote Qdrant server/cloud endpoint. If provided, overrides `path`.
            api_key: Authentication key for cloud deployments.
            distance_metric: Cosine distance calculation metric.
        """
        if url:
            logger.info(f"Connecting to remote Qdrant instance at: {url}")
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            logger.info(f"Initializing local Qdrant embedded storage at: {path}")
            self.client = QdrantClient(path=path)
        self.distance_metric = distance_metric

    def ensure_collection(self, collection_name: str, dimension: int) -> None:
        """
        Create the target vector collection if it does not already exist.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            logger.info(f"Creating Qdrant collection '{collection_name}' with dimension {dimension}...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=dimension,
                    distance=self.distance_metric,
                ),
            )
            # Create payload indexes on commonly filtered fields
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="doc_id",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Successfully created collection '{collection_name}' with payload indexes.")
        else:
            logger.debug(f"Collection '{collection_name}' already exists.")

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[RetrievedChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Batch insert or overwrite chunks with their dense embeddings and payload metadata.
        """
        if not chunks or not embeddings:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatched batch sizes: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        points: List[rest.PointStruct] = []
        for chunk, vec in zip(chunks, embeddings):
            # Generate deterministic UUID v5 from unique chunk_id to prevent duplicates
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            
            payload: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "doc_id": chunk.metadata.doc_id,
                "page_no": chunk.metadata.page_no,
                "section_header": chunk.metadata.section_header,
                "doc_date": chunk.metadata.doc_date,
                "chunk_index": chunk.metadata.chunk_index,
            }
            # Merge custom fields into payload
            if chunk.metadata.custom_fields:
                payload.update(chunk.metadata.custom_fields)

            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logger.debug(f"Upserted {len(points)} points into Qdrant collection '{collection_name}'.")
        return len(points)

    def search_dense(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 15,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute dense vector cosine similarity search.
        """
        query_filter = self._build_filter(metadata_filters) if metadata_filters else None

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        retrieved: List[RetrievedChunk] = []
        for hit in results.points:
            payload = hit.payload or {}
            meta = ChunkMetadata(
                doc_id=payload.get("doc_id", "unknown"),
                page_no=payload.get("page_no", 1),
                section_header=payload.get("section_header"),
                doc_date=payload.get("doc_date"),
                chunk_index=payload.get("chunk_index", 0),
                custom_fields={k: v for k, v in payload.items() if k not in {"doc_id", "page_no", "section_header", "doc_date", "chunk_index", "text", "chunk_id"}},
            )
            retrieved.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    text=payload.get("text", ""),
                    score=float(hit.score),
                    metadata=meta,
                )
            )

        return retrieved

    def delete_document(self, collection_name: str, doc_id: str) -> int:
        """
        Delete all indexed chunks belonging to a specific document ID.
        """
        filter_selector = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="doc_id",
                    match=rest.MatchValue(value=doc_id),
                )
            ]
        )
        response = self.client.delete(
            collection_name=collection_name,
            points_selector=rest.FilterSelector(filter=filter_selector),
            wait=True,
        )
        logger.info(f"Deleted points matching doc_id='{doc_id}' from collection '{collection_name}'.")
        return 1

    def _build_filter(self, filters: Dict[str, Any]) -> rest.Filter:
        """Translate Python dictionary filters into Qdrant Filter objects."""
        must_conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                must_conditions.append(
                    rest.FieldCondition(key=key, match=rest.MatchAny(any=value))
                )
            else:
                must_conditions.append(
                    rest.FieldCondition(key=key, match=rest.MatchValue(value=value))
                )
        return rest.Filter(must=must_conditions)
