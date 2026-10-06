import asyncio
import pytest

from src.providers.base import BaseLLMProvider, LLMProviderError
from src.providers.factory import create_llm_provider
from src.providers.gemini import GeminiProvider
from src.providers.mock import MockLLMProvider
from src.providers.openai import OpenAIProvider
from src.server import ask_gpt, get_llm_provider, set_llm_provider


def test_factory_mock_priority_with_mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    provider = create_llm_provider()
    assert isinstance(provider, MockLLMProvider)


def test_factory_mock_priority_with_legacy_mock_gpt(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setenv("MOCK_GPT", "1")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    provider = create_llm_provider()
    assert isinstance(provider, MockLLMProvider)


def test_factory_explicit_gemini(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.delenv("MOCK_GPT", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    provider = create_llm_provider()
    assert isinstance(provider, GeminiProvider)


def test_factory_explicit_openai(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.delenv("MOCK_GPT", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    provider = create_llm_provider()
    assert isinstance(provider, OpenAIProvider)


def test_factory_explicit_mock(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.delenv("MOCK_GPT", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    provider = create_llm_provider()
    assert isinstance(provider, MockLLMProvider)


def test_factory_unsupported_provider_raises(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.delenv("MOCK_GPT", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "unsupported-provider")

    with pytest.raises(LLMProviderError, match="Unsupported LLM_PROVIDER"):
        create_llm_provider()


def test_factory_missing_provider_raises_safe_error(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.delenv("MOCK_GPT", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(LLMProviderError, match="LLM_PROVIDER is not configured"):
        create_llm_provider()


def test_mock_llm_provider_basic_invocation():
    mock = MockLLMProvider()
    result = asyncio.run(mock.generate_text("Hello"))
    assert result == "Mock GPT response"


def test_mock_llm_provider_context_passing():
    mock = MockLLMProvider()
    result = asyncio.run(mock.generate_text("Summarize", context="Important context"))
    assert "Important context" in result


def test_server_dependency_injection():
    class CustomTestProvider(BaseLLMProvider):
        async def generate_text(self, question: str, context: str | None = None) -> str:
            return f"Custom response to: {question}"

    set_llm_provider(CustomTestProvider())
    result = asyncio.run(ask_gpt(question="ping"))
    assert result == "Custom response to: ping"

    # Reset provider injection
    set_llm_provider(None)
