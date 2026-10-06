from __future__ import annotations

import logging
import sys
from typing import Optional

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from src.providers import (
    BaseLLMProvider,
    LLMProviderError,
    create_llm_provider,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger("ai_agent_bridge")

mcp = MCPServer("ai-agent-bridge")

_configured_provider: BaseLLMProvider | None = None


def get_llm_provider() -> BaseLLMProvider:
    """Return the active LLM provider, creating one if not explicitly injected."""
    global _configured_provider
    if _configured_provider is not None:
        return _configured_provider
    return create_llm_provider()


def set_llm_provider(provider: BaseLLMProvider | None) -> None:
    """Inject a custom or mock LLM provider for testing."""
    global _configured_provider
    _configured_provider = provider


class _LazyProviderProxy(BaseLLMProvider):
    """Proxy delegating dynamically to the active provider."""

    async def generate_text(self, question: str, context: str | None = None) -> str:
        provider = get_llm_provider()
        return await provider.generate_text(question=question, context=context)


llm_provider = _LazyProviderProxy()

# Backward compatibility aliases
openai_client = llm_provider
set_openai_client = set_llm_provider
create_openai_client = get_llm_provider
OpenAIClientError = LLMProviderError


@mcp.tool()
async def ask_gpt(
    question: str,
    context: Optional[str] = None,
) -> str:
    """Ask an OpenAI GPT model a question, optionally with additional context."""

    question = question.strip()

    if not question:
        raise ToolError("question must not be empty.")

    try:
        provider = get_llm_provider()
        return await provider.generate_text(question=question, context=context)

    except LLMProviderError as exc:
        logger.warning("ask_gpt failed: %s", exc)
        raise ToolError(str(exc)) from None

    except Exception:
        logger.exception("Unexpected error while executing ask_gpt")
        raise


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
