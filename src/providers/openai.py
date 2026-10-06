from __future__ import annotations

import logging
import os
from typing import Any

import openai
from openai import AsyncOpenAI

from src.providers.base import BaseLLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class OpenAIClientError(LLMProviderError):
    """Safe, user-facing error raised for expected OpenAI failures."""


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_client: Any | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = api_client
        self._model = model
        self._timeout = timeout

    async def generate_text(self, question: str, context: str | None = None) -> str:
        model = self._model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")

        if not model:
            raise OpenAIClientError(
                "OPENAI_MODEL is not configured."
            )

        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise OpenAIClientError(
                    "OPENAI_API_KEY is not configured."
                )

            self._client = AsyncOpenAI(
                api_key=api_key,
                timeout=self._timeout,
            )

        # Assemble prompt within provider contract
        if context is not None and context.strip():
            prompt = f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}"
        else:
            prompt = question.strip()

        try:
            response = await self._client.responses.create(
                model=model,
                input=prompt,
            )

            output_text = response.output_text

            if not output_text or not output_text.strip():
                raise OpenAIClientError(
                    "OpenAI returned an empty response."
                )

            return output_text.strip()

        except openai.AuthenticationError as exc:
            logger.warning("OpenAI authentication failed: %s", exc)
            raise OpenAIClientError(
                "OpenAI authentication failed. Check OPENAI_API_KEY."
            ) from None

        except openai.RateLimitError as exc:
            logger.warning("OpenAI rate limit reached: %s", exc)
            raise OpenAIClientError(
                "OpenAI rate limit reached. Please retry later."
            ) from None

        except openai.APITimeoutError as exc:
            logger.warning("OpenAI request timed out: %s", exc)
            raise OpenAIClientError(
                "OpenAI request timed out."
            ) from None

        except openai.APIConnectionError as exc:
            logger.warning("OpenAI connection failed: %s", exc)
            raise OpenAIClientError(
                "Unable to connect to the OpenAI API."
            ) from None

        except openai.APIStatusError as exc:
            logger.warning(
                "OpenAI API returned status %s. request_id=%s",
                exc.status_code,
                exc.request_id,
            )
            raise OpenAIClientError(
                f"OpenAI API returned HTTP {exc.status_code}."
            ) from None

        except openai.APIError as exc:
            logger.warning("OpenAI API error: %s", exc)
            raise OpenAIClientError(
                "OpenAI API request failed."
            ) from None


# Backward compatibility alias
OpenAIClient = OpenAIProvider
