from __future__ import annotations

import logging
import os

from src.providers.base import BaseLLMProvider, LLMProviderError
from src.providers.gemini import GeminiProvider
from src.providers.mock import MockLLMProvider
from src.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


def create_llm_provider() -> BaseLLMProvider:
    """Create an LLM provider based on explicit configuration.

    Priority:
    1. MOCK_LLM=1 or legacy MOCK_GPT=1 -> MockLLMProvider
    2. LLM_PROVIDER=gemini -> GeminiProvider
    3. LLM_PROVIDER=openai -> OpenAIProvider
    4. Legacy compatibility: if OPENAI_API_KEY or OPENAI_MODEL is set without LLM_PROVIDER -> OpenAIProvider
    5. Missing LLM_PROVIDER -> raise safe LLMProviderError
    """
    # 1. Mock Priority
    if os.getenv("MOCK_LLM") == "1" or os.getenv("MOCK_GPT") == "1":
        logger.info("Using MockLLMProvider (MOCK_LLM/MOCK_GPT active)")
        return MockLLMProvider()

    provider = os.getenv("LLM_PROVIDER")

    if provider:
        provider = provider.strip().lower()
        if provider == "gemini":
            logger.info("Using GeminiProvider (LLM_PROVIDER=gemini)")
            return GeminiProvider()
        elif provider == "openai":
            logger.info("Using OpenAIProvider (LLM_PROVIDER=openai)")
            return OpenAIProvider()
        elif provider == "mock":
            logger.info("Using MockLLMProvider (LLM_PROVIDER=mock)")
            return MockLLMProvider()
        else:
            raise LLMProviderError(
                f"Unsupported LLM_PROVIDER '{provider}'. Supported providers are 'gemini' and 'openai'."
            )

    # Legacy backward compatibility check for existing tests
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MODEL"):
        logger.info("Falling back to legacy OpenAIProvider from existing environment variables")
        return OpenAIProvider()

    raise LLMProviderError(
        "LLM_PROVIDER is not configured. Please set LLM_PROVIDER='gemini' or LLM_PROVIDER='openai'."
    )
