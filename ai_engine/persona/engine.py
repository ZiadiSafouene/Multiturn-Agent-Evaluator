"""
persona/engine.py

LLM-based persona transformation engine.
Rewrites text while preserving intent and slot values.
"""

from __future__ import annotations
from typing import Optional

from ai_engine.utils.models import Persona, Dialect
from ai_engine.llm.tester import LLMTester


class PersonaEngine:
    def __init__(self, llm: LLMTester):
        self.llm = llm

    # ─────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────

    def apply(
        self,
        text: str,
        persona: Persona,
        dialect: Optional[Dialect] = None,
    ) -> str:
        prompt = self._build_prompt(text, persona, dialect)

        output = self.llm.generate(prompt)

        cleaned = self._clean_output(output)

        # Safety fallback
        if not cleaned:
            return text

        return cleaned

    # ─────────────────────────────────────────────────────
    # Prompt
    # ─────────────────────────────────────────────────────

    def _build_prompt(
        self,
        text: str,
        persona: Persona,
        dialect: Optional[Dialect],
    ) -> str:
        return f"""
You are a rewriting engine.

TASK:
Rewrite the given text to match a specific USER PERSONA and DIALECT.

STRICT RULES (VERY IMPORTANT):
- DO NOT change the meaning
- DO NOT remove or alter any factual information
- DO NOT remove numbers, IDs, or slot values
- DO NOT add new information
- DO NOT change intent
- ONLY modify tone, phrasing, and style

PERSONA:
{persona.value}

PERSONA BEHAVIOR:
- cooperative: clear, polite, helpful
- confused: unsure, asks clarification, hesitant
- impatient: short, direct, slightly rushed
- angry: frustrated, slightly aggressive but not abusive

DIALECT:
{dialect.value if dialect else "none"}

DIALECT RULE:
- If dialect is "none", keep neutral English
- Otherwise, slightly adapt phrasing (DO NOT translate unless explicitly needed later)

INPUT TEXT:
"{text}"

OUTPUT:
Return ONLY the rewritten text.
No explanations.
"""

    # ─────────────────────────────────────────────────────
    # Output cleaning
    # ─────────────────────────────────────────────────────

    def _clean_output(self, output: str) -> str:
        if not output:
            return ""

        # Remove quotes if model wraps response
        output = output.strip()

        if output.startswith('"') and output.endswith('"'):
            output = output[1:-1]

        # Remove accidental prefixes
        for prefix in ["Output:", "Rewritten:", "Text:"]:
            if output.lower().startswith(prefix.lower()):
                output = output[len(prefix):].strip()

        return output