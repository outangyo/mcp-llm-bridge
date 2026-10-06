import asyncio
from types import SimpleNamespace

import pytest

from src.openai_client import OpenAIClient, OpenAIClientError


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.response


class FakeOpenAIClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def test_generate_text_returns_model_output():
    fake_response = SimpleNamespace(
        output_text="Hello from GPT"
    )

    fake_client = FakeOpenAIClient(fake_response)

    client = OpenAIClient(
        api_client=fake_client,
        model="test-model",
    )

    result = asyncio.run(
        client.generate_text("Hello")
    )

    assert result == "Hello from GPT"
    assert fake_client.responses.last_kwargs == {
        "model": "test-model",
        "input": "Hello",
    }


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = OpenAIClient(
        model="test-model",
    )

    with pytest.raises(
        OpenAIClientError,
        match="OPENAI_API_KEY is not configured",
    ):
        asyncio.run(
            client.generate_text("Hello")
        )


def test_missing_model_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    client = OpenAIClient()

    with pytest.raises(
        OpenAIClientError,
        match="OPENAI_MODEL is not configured",
    ):
        asyncio.run(
            client.generate_text("Hello")
        )


def test_empty_model_response_raises():
    fake_response = SimpleNamespace(
        output_text=""
    )

    fake_client = FakeOpenAIClient(fake_response)

    client = OpenAIClient(
        api_client=fake_client,
        model="test-model",
    )

    with pytest.raises(
        OpenAIClientError,
        match="empty response",
    ):
        asyncio.run(
            client.generate_text("Hello")
        )
