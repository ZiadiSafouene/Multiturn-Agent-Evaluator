from __future__ import annotations
import os
import time
import logging
from typing import Optional
import requests
import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """
    OpenRouter-based LLM Client.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY") ,
        model: str = "inclusionai/ling-2.6-flash:free",
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        if not self.api_key:
            logger.warning("No OPENROUTER_API_KEY — using stub mode")

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return self._stub_generate(prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a natural language generator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }

        last_exc = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code != 200:
                    raise RuntimeError(response.text)

                data = response.json()

                content = data["choices"][0]["message"]["content"]

                return content.strip()

            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(f"LLM retry {attempt}: {exc}")
                time.sleep(wait)

        logger.error(f"LLM failed: {last_exc}")
        return self._stub_generate(prompt)

    # ------------------------------------------------------------------
    # STUB (alternative if the model is not reachable)
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_generate(prompt: str) -> str:
        return "Client Not Reachable :)"