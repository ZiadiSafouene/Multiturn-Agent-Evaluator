"""
core.scenario.generator
~~~~~~~~~~~~~~~~~~~~~~~

Generates structured scenarios from high-level natural language descriptions
using the OpenRouter LLM tester.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from core.llm.prompts import build_scenario_generation_prompt
from core.llm.tester import LLMTester
from core.scenario.parser import parse_scenario
from core.scenario.validator import validate_scenario
from core.utils.models import Scenario

logger = logging.getLogger(__name__)


def clean_llm_output(text: str) -> str:
    """Strip markdown code fence blocks and superfluous wrapping."""
    text = text.strip()

    if text.startswith("```"):
        text = text.split("```", 1)[-1]
        text = text.rsplit("```", 1)[0]

    if text.lower().startswith("json"):
        text = text[4:].strip()

    return text.strip()


def extract_json_block(text: str) -> str:
    """Extract array substring delimited by [ and ]."""
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    # Check if single object returned
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        return f"[{text[obj_start : obj_end + 1]}]"

    return text


def generate_scenarios_from_description(
    description: str,
    llm: Optional[LLMTester] = None,
) -> Tuple[List[Scenario], List[str]]:
    """
    Generate validated scenarios using an LLMTester.

    Parameters
    ----------
    description:
        Natural language description of the agent's domain and intents.
    llm:
        Optional :class:`~core.llm.tester.LLMTester` instance. If omitted,
        a default tester is instantiated.

    Returns
    -------
    Tuple[List[Scenario], List[str]]
        A tuple of (valid_scenarios, error_messages).
    """
    tester = llm or LLMTester()
    prompt = build_scenario_generation_prompt(description)
    raw_output = tester.generate(prompt)

    cleaned = clean_llm_output(raw_output)
    cleaned = extract_json_block(cleaned)

    try:
        scenarios_json = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode scenarios JSON: %s\nRaw:\n%s", exc, cleaned)
        raise ValueError(f"Invalid JSON returned by tester model:\n{cleaned}") from exc

    if isinstance(scenarios_json, dict):
        scenarios_json = [scenarios_json]

    scenarios: List[Scenario] = []
    errors: List[str] = []

    for sc in scenarios_json:
        try:
            scenario = parse_scenario(sc)
            validation = validate_scenario(scenario)

            if validation.valid:
                scenarios.append(scenario)
            else:
                errors.extend(validation.errors)

        except Exception as e:
            errors.append(str(e))

    return scenarios, errors
