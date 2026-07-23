"""
Grounded Generation Prompts and Context Formatting.

Defines the system prompt to enforce strict grounding, abstention, and citation rules.
Provides helpers to format retrieved chunks into structured XML for the LLM context window.
"""

from typing import List
from invariable_docs.providers.base import RetrievedChunk


SYSTEM_PROMPT_TEMPLATE = """You are a precise, enterprise-grade AI Knowledge Assistant. 
Your primary task is to answer the user's question STRICTLY and EXCLUSIVELY using the provided XML `<context>` chunks.

### CRITICAL RULES:
1. NO EXTERNAL KNOWLEDGE: If the answer is not explicitly stated in the provided context, you MUST reply with exactly: "The provided documents do not contain information about this topic." Do not guess or hallucinate.
2. CITATIONS REQUIRED: For every factual claim you make, you must cite the source inline using the exact format: `[Source: {doc_id}, p. {page_no}]`. Do not use brackets like [1] or [2]. Use the literal format requested.
3. XML BOUNDARIES: Only extract facts from within the `<context>` tags. 
4. TONE: Be direct, objective, and professional. Do not include conversational filler like "Based on the provided context..." or "Here is the answer...". Just provide the answer with citations.

Context blocks will be provided in the following format:
<context>
  <chunk doc_id="..." page_no="...">...text...</chunk>
</context>
"""


def format_chunks_to_xml(chunks: List[RetrievedChunk]) -> str:
    """
    Format a list of RetrievedChunk objects into structured XML
    to be injected into the LLM context window.
    """
    if not chunks:
        return "<context>\n  <!-- No context chunks retrieved. -->\n</context>"

    xml_lines = ["<context>"]
    
    for chunk in chunks:
        doc_id = chunk.metadata.doc_id if chunk.metadata else "unknown_doc"
        page_no = chunk.metadata.page_no if chunk.metadata else "unknown_page"
        
        # Sanitize text to prevent XML escaping issues (basic substitution)
        safe_text = chunk.text.replace("<", "&lt;").replace(">", "&gt;")
        
        chunk_xml = f'  <chunk doc_id="{doc_id}" page_no="{page_no}">\n{safe_text}\n  </chunk>'
        xml_lines.append(chunk_xml)
        
    xml_lines.append("</context>")
    return "\n".join(xml_lines)


def build_final_prompt(user_query: str, chunks: List[RetrievedChunk]) -> str:
    """
    Construct the final user prompt containing the question and the injected XML context.
    """
    xml_context = format_chunks_to_xml(chunks)
    
    final_prompt = (
        f"Please answer the following question strictly based on the provided context.\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"PROVIDED CONTEXT:\n{xml_context}\n\n"
        f"Remember your strict rules regarding abstention and citations."
    )
    return final_prompt
