"""
Grounded Generation Engine.

Orchestrates the final LLM text generation step in the RAG pipeline.
Injects context, strictly configures the LLM for deterministic outputs,
and verifies citations in the final answer.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple
from invariable_docs.providers.base import BaseLLMProvider, RetrievedChunk
from invariable_docs.generation.prompts import SYSTEM_PROMPT_TEMPLATE, build_final_prompt

logger = logging.getLogger(__name__)


class GenerationEngine:
    """
    Executes the generation phase of the RAG pipeline, ensuring strict
    context adherence and deterministic generation parameters.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        top_p: float = 0.90,
    ):
        """
        Initialize the Generation Engine.
        
        Args:
            llm_provider: The configured LLM provider instance.
            temperature: Low temperature for highly deterministic/factual generation.
            top_p: Nucleus sampling threshold.
        """
        self.llm = llm_provider
        self.temperature = temperature
        self.top_p = top_p
        
        # Exact string expected for abstentions
        self.abstention_string = "The provided documents do not contain information about this topic."

    def generate_answer(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        """
        Generate a grounded answer for the user's query based ONLY on the provided chunks.
        """
        if not context_chunks:
            logger.info("No context chunks provided to generator. Returning strict abstention.")
            return self.abstention_string
            
        logger.debug(f"Generating answer for query: '{query}' using {len(context_chunks)} chunks.")
        
        final_prompt = build_final_prompt(user_query=query, chunks=context_chunks)
        
        raw_response = self.llm.generate(
            prompt=final_prompt,
            system_prompt=SYSTEM_PROMPT_TEMPLATE,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1024,
        )
        
        # Optional post-generation verification step
        is_valid, error_msg = self.verify_citations(raw_response, context_chunks)
        if not is_valid:
            logger.warning(f"Citation verification failed: {error_msg}")
            # In an enterprise system, we might re-prompt the LLM here to fix it,
            # or gracefully degrade the answer. For now, we return it but log the warning.
            
        return raw_response

    def verify_citations(self, response: str, provided_chunks: List[RetrievedChunk]) -> Tuple[bool, str]:
        """
        Post-generation verification helper to ensure that any citations 
        made by the LLM actually map to the provided chunks.
        
        Expected citation format: [Source: {doc_id}, p. {page_no}]
        """
        if self.abstention_string in response:
            # If the model abstained, it correctly shouldn't have citations.
            return True, "Abstained"
            
        # Regex to find all instances of [Source: ..., p. ...]
        pattern = r"\[Source:\s*([^,]+),\s*p\.\s*([^\]]+)\]"
        citations_found = re.findall(pattern, response)
        
        if not citations_found:
            return False, "No valid citations found in the response."
            
        valid_doc_ids = {chunk.metadata.doc_id for chunk in provided_chunks if chunk.metadata}
        valid_pages = {str(chunk.metadata.page_no) for chunk in provided_chunks if chunk.metadata}
        
        for doc_id, page_no in citations_found:
            doc_id = doc_id.strip()
            page_no = page_no.strip()
            if doc_id not in valid_doc_ids:
                return False, f"Hallucinated citation source: {doc_id}"
            if page_no not in valid_pages:
                return False, f"Hallucinated page number: {page_no} for document {doc_id}"
                
        return True, "All citations valid"
