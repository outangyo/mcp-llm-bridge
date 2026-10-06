import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors

from src.providers.gemini import GeminiProvider, GeminiProviderError


class FakeModels:
    def __init__(self, response=None, side_effect=None):
        self.response = response
        self.side_effect = side_effect
        self.last_kwargs = None

    async def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        if self.side_effect:
            raise self.side_effect
        return self.response


class FakeAio:
    def __init__(self, models):
        self.models = models


class FakeGenAIClient:
    def __init__(self, response=None, side_effect=None):
        self.models = FakeModels(response=response, side_effect=side_effect)
        self.aio = FakeAio(self.models)


def test_generate_text_returns_model_output_without_context():
    fake_response = SimpleNamespace(text="Hello from Gemini")
    fake_client = FakeGenAIClient(response=fake_response)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    result = asyncio.run(provider.generate_text("Hello"))

    assert result == "Hello from Gemini"
    assert fake_client.models.last_kwargs == {
        "model": "gemini-2.5-flash",
        "contents": "Hello",
    }


def test_generate_text_with_context():
    fake_response = SimpleNamespace(text="Summary based on context")
    fake_client = FakeGenAIClient(response=fake_response)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    result = asyncio.run(
        provider.generate_text(
            question="What is this?",
            context="System architecture document",
        )
    )

    assert result == "Summary based on context"
    assert fake_client.models.last_kwargs == {
        "model": "gemini-2.5-flash",
        "contents": "Context:\nSystem architecture document\n\nQuestion:\nWhat is this?",
    }


def test_model_resolution_from_llm_model_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-pro")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    fake_response = SimpleNamespace(text="Response")
    fake_client = FakeGenAIClient(response=fake_response)

    provider = GeminiProvider(api_client=fake_client)
    result = asyncio.run(provider.generate_text("Hi"))

    assert result == "Response"
    assert fake_client.models.last_kwargs["model"] == "gemini-2.5-pro"


def test_model_resolution_fallback_to_gemini_model_env(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    fake_response = SimpleNamespace(text="Response")
    fake_client = FakeGenAIClient(response=fake_response)

    provider = GeminiProvider(api_client=fake_client)
    result = asyncio.run(provider.generate_text("Hi"))

    assert result == "Response"
    assert fake_client.models.last_kwargs["model"] == "gemini-2.5-flash"


def test_missing_model_raises(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    provider = GeminiProvider()

    with pytest.raises(GeminiProviderError, match="GEMINI_MODEL is not configured"):
        asyncio.run(provider.generate_text("Hello"))


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = GeminiProvider()

    with pytest.raises(GeminiProviderError, match="GEMINI_API_KEY is not configured"):
        asyncio.run(provider.generate_text("Hello"))


def test_empty_response_raises():
    fake_response = SimpleNamespace(text="")
    fake_client = FakeGenAIClient(response=fake_response)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    with pytest.raises(GeminiProviderError, match="empty response"):
        asyncio.run(provider.generate_text("Hello"))


def test_none_text_response_raises():
    fake_response = SimpleNamespace(text=None)
    fake_client = FakeGenAIClient(response=fake_response)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    with pytest.raises(GeminiProviderError, match="empty response"):
        asyncio.run(provider.generate_text("Hello"))


def test_auth_error_mapping():
    side_effect = errors.APIError(401, {"error": {"message": "API_KEY_INVALID"}})
    fake_client = FakeGenAIClient(side_effect=side_effect)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    with pytest.raises(GeminiProviderError, match="authentication failed"):
        asyncio.run(provider.generate_text("Hello"))


def test_permission_denied_error_mapping():
    side_effect = errors.APIError(403, {"error": {"message": "PERMISSION_DENIED"}})
    fake_client = FakeGenAIClient(side_effect=side_effect)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    with pytest.raises(GeminiProviderError, match="authentication failed"):
        asyncio.run(provider.generate_text("Hello"))


def test_rate_limit_error_mapping():
    side_effect = errors.APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
    fake_client = FakeGenAIClient(side_effect=side_effect)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    with pytest.raises(GeminiProviderError, match="rate limit or quota reached"):
        asyncio.run(provider.generate_text("Hello"))


def test_server_error_mapping():
    side_effect = errors.APIError(503, {"error": {"message": "Service Unavailable"}})
    fake_client = FakeGenAIClient(side_effect=side_effect)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    with pytest.raises(GeminiProviderError, match="server error HTTP 503"):
        asyncio.run(provider.generate_text("Hello"))


def test_timeout_error_mapping():
    side_effect = TimeoutError("Connection timed out")
    fake_client = FakeGenAIClient(side_effect=side_effect)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    with pytest.raises(GeminiProviderError, match="timed out"):
        asyncio.run(provider.generate_text("Hello"))


def test_connection_error_mapping():
    side_effect = ConnectionError("Connection refused")
    fake_client = FakeGenAIClient(side_effect=side_effect)

    provider = GeminiProvider(
        api_client=fake_client,
        model="gemini-2.5-flash",
    )

    with pytest.raises(GeminiProviderError, match="Unable to connect to the Gemini API"):
        asyncio.run(provider.generate_text("Hello"))
