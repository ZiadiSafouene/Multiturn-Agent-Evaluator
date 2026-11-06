"""
core.llm.tester
~~~~~~~~~~~~~~~

OpenRouter-based LLM tester used to generate **scenarios** and **judge**
agent responses.

The user may supply their own API key and model via constructor arguments or
the ``TESTER_API_KEY`` / ``TESTER_MODEL`` environment variables.
When neither is set the tester falls back to a free OpenRouter model.
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

# Free default tester model — works without an API key on OpenRouter
DEFAULT_TESTER_MODEL = "mistralai/mistral-7b-instruct:free"

_RETRY_BACKOFF_BASE = 2


class LLMTester:
    """
    OpenRouter-based tester used for **scenario generation** and **LLM judging**.

    Parameters
    ----------
    api_key:
        OpenRouter API key for the tester model.
        Falls back to the ``TESTER_API_KEY`` environment variable, then
        ``OPENROUTER_API_KEY``.
        When *neither* is set the tester uses a free OpenRouter model.
    model:
        OpenRouter model identifier.
        Falls back to ``TESTER_MODEL`` env var, then :data:`DEFAULT_TESTER_MODEL`.
    temperature:
        Sampling temperature forwarded to the model (0.0 – 2.0).
        Kept low by default to produce consistent structured outputs.
    max_retries:
        Maximum number of HTTP retry attempts before giving up.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        self.api_key: Optional[str] = (
            api_key or os.getenv("TESTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        )
        self.model: str = (
            model or os.getenv("TESTER_MODEL") or DEFAULT_TESTER_MODEL
        )
        self.temperature: float = temperature
        self.max_retries: int = max_retries

        if not self.api_key:
            logger.info(
                "No TESTER_API_KEY found — using free model '%s'.", self.model
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: str = "You are a strict JSON generator.") -> str:
        """
        Send *prompt* to the tester model and return the response text.

        Retries up to :attr:`max_retries` times with exponential back-off.

        Parameters
        ----------
        prompt:
            The instruction / task prompt.
        system:
            System message override (defaults to strict JSON mode).

        Returns
        -------
        str
            The model's response, stripped of leading/trailing whitespace.
        """
        messages = [
            {"role": "system", "content": system},
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
                    "LLMTester retry %d/%d after %.0fs: %s",
                    attempt, self.max_retries, wait, exc,
                )
                time.sleep(wait)

        logger.error("LLMTester exhausted retries. Last error: %s", last_exc)
        # Minimal stub — prevents total pipeline failure
        return (
            '[{"scenario_id":"stub","title":"Stub","description":"stub",'
            '"goal":"stub","domain":"stub","slots":[],"states":[],'
            '"transitions":[],"turns":[]}]'
        )
