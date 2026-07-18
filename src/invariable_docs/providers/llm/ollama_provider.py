"""
Ollama LLM Provider Module.

Implements `BaseLLMProvider` to run generative language models locally
using the Ollama engine, optimized for Apple Silicon (Metal) inference.
"""

from __future__ import annotations

import logging
from typing import List, Optional
import ollama
from invariable_docs.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaLLMProvider(BaseLLMProvider):
    """
    Local LLM generation via Ollama.
    
    Supports synchronous and asynchronous chat completions with configurable
    generation parameters (`temperature`, `top_p`, `max_tokens`).
    """

    def __init__(self, model_name: str = "llama3.1", host: str = "http://localhost:11434"):
        """
        Initialize the Ollama provider.
        
        Args:
            model_name: The Ollama model tag (e.g., 'llama3.1', 'mistral', 'phi3').
            host: URL of the running Ollama server.
        """
        self.model_name = model_name
        self.host = host
        # Create a dedicated async client for the `agenerate` method
        self.async_client = ollama.AsyncClient(host=self.host)
        
        # Instantiate a sync client bound to the specific host
        self.client = ollama.Client(host=self.host)
        logger.info(f"Initialized OllamaLLMProvider (model: '{self.model_name}', host: '{self.host}')")

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> List[dict]:
        """Construct the OpenAI-style chat message history."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_options(self, temperature: float, top_p: float, max_tokens: int, stop_sequences: Optional[List[str]]) -> dict:
        """Construct the Ollama-specific generation parameters."""
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,  # Ollama uses num_predict for max output tokens
        }
        if stop_sequences:
            options["stop"] = stop_sequences
        return options

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.90,
        max_tokens: int = 768,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a response synchronously using Ollama."""
        messages = self._build_messages(prompt, system_prompt)
        options = self._build_options(temperature, top_p, max_tokens, stop_sequences)
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
            )
            return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama synchronous generation failed: {e}", exc_info=True)
            raise

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.90,
        max_tokens: int = 768,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a response asynchronously using Ollama AsyncClient."""
        messages = self._build_messages(prompt, system_prompt)
        options = self._build_options(temperature, top_p, max_tokens, stop_sequences)
        
        try:
            response = await self.async_client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
            )
            return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama asynchronous generation failed: {e}", exc_info=True)
            raise
