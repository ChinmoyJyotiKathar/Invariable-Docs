"""
Query Transformations for advanced RAG.

Implements Hypothetical Document Embeddings (HyDE), Multi-Query Expansion,
and Step-Back prompting to improve hybrid retrieval recall.
"""

from __future__ import annotations

import logging
from typing import List
from invariable_docs.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class QueryTransformer:
    """
    Applies LLM-driven transformations to user queries to boost retrieval performance.
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider

    def generate_hyde_document(self, query: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE).
        Generates a fake hypothetical document that answers the query.
        This document can then be embedded and searched for in the vector database
        to find structurally similar real documents.
        """
        system_prompt = (
            "You are an expert AI assistant. Given a user query, write a hypothetical "
            "document snippet or paragraph that perfectly answers the query. "
            "Do not include pleasantries. Write it in the tone of an official document."
        )
        logger.debug(f"Generating HyDE document for query: '{query}'")
        hypothetical_doc = self.llm.generate(
            prompt=query,
            system_prompt=system_prompt,
            temperature=0.3,  # Slight creativity for document generation
            max_tokens=256,
        )
        return hypothetical_doc

    def expand_multi_query(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Multi-Query Expansion.
        Generates multiple distinct variations of the original query to overcome
        vocabulary mismatch and capture different semantic angles.
        """
        system_prompt = (
            f"You are an AI search assistant. Generate exactly {num_variations} distinct "
            "search queries related to the user's original query. These queries should use "
            "different synonyms or target different angles of the same underlying intent. "
            "Return ONLY the queries, one per line, without numbers or bullets."
        )
        logger.debug(f"Expanding query into {num_variations} variations: '{query}'")
        response = self.llm.generate(
            prompt=query,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=128,
        )
        
        # Parse the raw response into a list of queries
        variations = [line.strip("- *1234567890.") for line in response.split("\n") if line.strip()]
        
        # Always include the original query
        results = [query]
        for v in variations:
            if v and v.lower() != query.lower():
                results.append(v)
                
        return results[:num_variations + 1]

    def generate_step_back_query(self, query: str) -> str:
        """
        Step-Back Prompting.
        Abstracts a highly specific query into a broader, more general question
        to retrieve foundational context.
        """
        system_prompt = (
            "You are an expert at abstracting specific questions. Given a detailed user query, "
            "write a 'step-back' query that asks the broader, more general foundational question "
            "that encompasses the specific query. "
            "Return ONLY the step-back query."
        )
        logger.debug(f"Generating step-back query for: '{query}'")
        step_back = self.llm.generate(
            prompt=query,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=64,
        )
        return step_back.strip()
