from __future__ import annotations

"""Backward compatibility shim for OpenAIClient."""
from src.providers.openai import OpenAIClient, OpenAIClientError, OpenAIProvider

__all__ = ["OpenAIClient", "OpenAIClientError", "OpenAIProvider"]
