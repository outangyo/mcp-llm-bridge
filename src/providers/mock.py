from __future__ import annotations

from src.providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM Provider for offline testing and verification."""

    def __init__(self, model: str = "mock-model") -> None:
        self.model = model
        self.last_question: str | None = None
        self.last_context: str | None = None

    async def generate_text(self, question: str, context: str | None = None) -> str:
        self.last_question = question
        self.last_context = context

        # If context is explicitly provided
        if context is not None and context.strip():
            return f"Mock GPT: {context.strip()}"

        # If question was a legacy combined prompt containing Context:
        if "Context:" in question:
            context_part = question.split("Context:\n")[1].split("\n\nQuestion:")[0].strip()
            return f"Mock GPT: {context_part}"

        return "Mock GPT response"
