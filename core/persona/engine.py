"""
core.persona.engine
~~~~~~~~~~~~~~~~~~~

LLM-based persona transformation engine.

Rewrites user messages to match a target persona style and language
while preserving all intent, slot values, and factual information.
"""

from __future__ import annotations

from core.utils.models import Language, Persona
from core.llm.tester import LLMTester
from core.llm.prompts import build_persona_rewrite_prompt


class PersonaEngine:
    """
    Rewrites text to match a specified persona style and conversation language.

    Parameters
    ----------
    llm:
        An :class:`~core.llm.tester.LLMTester` instance used to perform
        the rewriting.
    """

    def __init__(self, llm: LLMTester) -> None:
        self.llm = llm

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def apply(
        self,
        text: str,
        persona: Persona,
        language: Language = Language.ENGLISH,
    ) -> str:
        """
        Rewrite *text* to match *persona* style in *language*.

        Parameters
        ----------
        text:
            The original message to rewrite.
        persona:
            The persona archetype to apply.
        language:
            The target conversation language (English, Arabic, or French).

        Returns
        -------
        str
            The rewritten message, or the original *text* if rewriting fails.
        """
        prompt = build_persona_rewrite_prompt(text, persona, language)
        output = self.llm.generate(prompt, system="You are a text rewriting engine.")
        cleaned = self._clean_output(output)
        return cleaned if cleaned else text

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_output(output: str) -> str:
        """Strip wrapper quotes and common prefixes from LLM output."""
        if not output:
            return ""

        output = output.strip()

        # Remove surrounding quotes if the model wrapped the response
        if output.startswith('"') and output.endswith('"'):
            output = output[1:-1]

        # Remove common prefixes that models sometimes prepend
        for prefix in ("Output:", "Rewritten:", "Text:", "Réponse:", "الرد:"):
            if output.lower().startswith(prefix.lower()):
                output = output[len(prefix):].strip()

        return output
