from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProviderError(RuntimeError):
    """Safe, user-facing error raised for expected LLM provider failures."""


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers in ai-agent-bridge."""

    @abstractmethod
    async def generate_text(self, question: str, context: str | None = None) -> str:
        """Generate text from the LLM provider given a question and optional context."""
        ...
