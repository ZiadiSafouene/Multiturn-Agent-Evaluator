from __future__ import annotations
from typing import Any, Dict
import json

from ai_engine.scenario.parser import parse_scenario
from ai_engine.utils.models import Scenario


def scenario_from_dict(data: Dict[str, Any]) -> Scenario:
    return parse_scenario(data)


def scenario_from_json(json_str: str) -> Scenario:
    return parse_scenario(json.loads(json_str))


def scenario_from_file(path: str) -> Scenario:
    with open(path, "r", encoding="utf-8") as f:
        return parse_scenario(json.load(f))
    

from ai_engine.llm.client import LLMClient
from ai_engine.llm.prompts import build_scenario_generation_prompt
from ai_engine.scenario.parser import parse_scenario
from ai_engine.scenario.validator import validate_scenario

import json
def clean_llm_output(text: str) -> str:
    text = text.strip()

    # remove markdown code blocks
    if text.startswith("```"):
        text = text.split("```", 1)[-1]
        text = text.rsplit("```", 1)[0]

    # remove "json" prefix if present
    if text.startswith("json"):
        text = text[len("json"):].strip()

    return text.strip()

def extract_json_block(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:
        return text[start:end+1]

    return text

def generate_scenarios_from_description(description: str):
    llm = LLMClient()

    prompt = build_scenario_generation_prompt(description)
    raw_output = llm.generate(prompt)

    cleaned = clean_llm_output(raw_output)
    cleaned = extract_json_block(cleaned)

    try:
        scenarios_json = json.loads(cleaned)
    except Exception:
        raise ValueError(f"Invalid JSON after cleaning:\n{cleaned}")

    scenarios = []
    errors = []

    for sc in scenarios_json:
        try:
            scenario = parse_scenario(sc)
            validation = validate_scenario(scenario)

            if validation.valid:
                scenarios.append(scenario)
            else:
                errors.append(validation.errors)

        except Exception as e:
            errors.append(str(e))

    return scenarios, errors