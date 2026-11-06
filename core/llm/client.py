"""
core.llm.client
~~~~~~~~~~~~~~~

OpenRouter-based LLM client used to drive the **agent under test**.

The user may supply their own API key and model via constructor arguments or
the ``AGENT_API_KEY`` / ``AGENT_MODEL`` environment variables.
When neither is set the client falls back to a free OpenRouter model so the
framework works out-of-the-box without any configuration.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free default — works without an API key on OpenRouter
DEFAULT_CLIENT_MODEL = "mistralai/mistral-7b-instruct:free"

_RETRY_BACKOFF_BASE = 2


class LLMClient:
    """
    OpenRouter-based client that drives the **agent under test**.

    Parameters
    ----------
    api_key:
        OpenRouter API key for the agent model.
        Falls back to the ``AGENT_API_KEY`` environment variable.
        When *neither* is set the client uses a free OpenRouter model.
    model:
        OpenRouter model identifier.
        Falls back to ``AGENT_MODEL`` env var, then :data:`DEFAULT_CLIENT_MODEL`.
    system_prompt:
        System prompt injected as the first message in every request.
        This is where the user defines the agent's personality / role.
        Defaults to a generic helpful assistant prompt.
    temperature:
        Sampling temperature forwarded to the model (0.0 – 2.0).
    max_retries:
        Maximum number of HTTP retry attempts before giving up.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> None:
        self.api_key: Optional[str] = (
            api_key or os.getenv("AGENT_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        )
        self.model: str = (
            model or os.getenv("AGENT_MODEL") or DEFAULT_CLIENT_MODEL
        )
        self.system_prompt: str = system_prompt or (
            "You are a helpful, professional AI assistant. "
            "Answer concisely and accurately."
        )
        self.temperature: float = temperature
        self.max_retries: int = max_retries

        if not self.api_key:
            logger.info(
                "No AGENT_API_KEY found — using free model '%s'.", self.model
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to the model and return the response text.

        Retries up to :attr:`max_retries` times with exponential back-off.

        Parameters
        ----------
        prompt:
            The user-turn message to send.

        Returns
        -------
        str
            The model's response, stripped of leading/trailing whitespace.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._call(messages)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call(self, messages: list[dict]) -> str:
        """Execute the HTTP request with retry logic."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    OPENROUTER_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"OpenRouter HTTP {response.status_code}: {response.text[:300]}"
                    )

                return response.json()["choices"][0]["message"]["content"].strip()

            except Exception as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "LLMClient retry %d/%d after %.0fs: %s",
                    attempt, self.max_retries, wait, exc,
                )
                time.sleep(wait)

        logger.error("LLMClient exhausted retries. Last error: %s", last_exc)
        return "[LLMClient: generation failed after retries]"
