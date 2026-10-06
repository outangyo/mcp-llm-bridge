from __future__ import annotations

import logging
import os
from typing import Any

from google import genai
from google.genai import errors

from src.providers.base import BaseLLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class GeminiProviderError(LLMProviderError):
    """Safe, user-facing error raised for expected Gemini failures."""


class GeminiProvider(BaseLLMProvider):
    """LLM Provider for Google Gemini using the official google-genai SDK."""

    def __init__(
        self,
        api_client: Any | None = None,
        model: str | None = None,
    ) -> None:
        self._client = api_client
        self._model = model

    async def generate_text(self, question: str, context: str | None = None) -> str:
        model = self._model or os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL")

        if not model:
            raise GeminiProviderError(
                "GEMINI_MODEL is not configured."
            )

        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise GeminiProviderError(
                    "GEMINI_API_KEY is not configured."
                )

            self._client = genai.Client(api_key=api_key)

        # Assemble prompt within provider contract
        if context is not None and context.strip():
            prompt = f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}"
        else:
            prompt = question.strip()

        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=prompt,
            )

            output_text = getattr(response, "text", None)

            if not output_text or not output_text.strip():
                raise GeminiProviderError(
                    "Gemini returned an empty response."
                )

            return output_text.strip()

        except errors.APIError as exc:
            err_code = getattr(exc, "code", None)
            err_str = str(exc)

            if err_code in (401, 403) or "API_KEY_INVALID" in err_str or "PERMISSION_DENIED" in err_str:
                logger.warning("Gemini authentication failed: %s", exc)
                raise GeminiProviderError(
                    "Gemini authentication failed. Check GEMINI_API_KEY."
                ) from None

            if err_code == 429 or "RESOURCE_EXHAUSTED" in err_str:
                logger.warning("Gemini rate limit/quota reached: %s", exc)
                raise GeminiProviderError(
                    "Gemini rate limit or quota reached. Please retry later."
                ) from None

            if err_code and err_code >= 500:
                logger.warning("Gemini server error (%s): %s", err_code, exc)
                raise GeminiProviderError(
                    f"Gemini API returned server error HTTP {err_code}."
                ) from None

            logger.warning("Gemini API error: %s", exc)
            raise GeminiProviderError(
                f"Gemini API request failed: {err_str}"
            ) from None

        except GeminiProviderError:
            raise

        except Exception as exc:
            err_type = type(exc).__name__
            if "Timeout" in err_type:
                logger.warning("Gemini request timed out: %s", exc)
                raise GeminiProviderError("Gemini request timed out.") from None
            if "Connect" in err_type or "Network" in err_type:
                logger.warning("Gemini connection failed: %s", exc)
                raise GeminiProviderError("Unable to connect to the Gemini API.") from None

            # Unexpected errors are logged and re-raised for server layer
            logger.exception("Unexpected error in GeminiProvider: %s", exc)
            raise
