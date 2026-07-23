"""
LiteLLM Provider Implementation.

Uses the 'litellm' package to provide a unified interface to over 100+ cloud LLMs
including Groq, HuggingFace, OpenAI, Anthropic, and Cohere.
"""

from typing import Dict, Any, Optional
import litellm
import logging

from invariable_docs.providers.base import BaseLLMProvider

# Disable verbose litellm logging by default
litellm.set_verbose = False
logger = logging.getLogger(__name__)

class LiteLLMProvider(BaseLLMProvider):
    """
    LLM provider that delegates to LiteLLM for cloud API routing.
    """

    def __init__(self, model_name: str = "groq/llama3-8b-8192"):
        """
        Initialize the LiteLLM Provider.
        
        Args:
            model_name: The target model (e.g. 'groq/llama3-8b-8192' or 'huggingface/meta-llama/Meta-Llama-3-8B-Instruct')
        """
        self.model_name = model_name
        logger.info(f"Initialized LiteLLM Provider routing to model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using the configured cloud LLM API.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LiteLLM generation failed: {e}")
            raise e

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.90,
        max_tokens: int = 768,
        stop_sequences: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Asynchronously generate text using the configured cloud LLM API.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LiteLLM async generation failed: {e}")
            raise e
