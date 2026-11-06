"""
core.scenario.storage
~~~~~~~~~~~~~~~~~~~~~

Serialization and deserialization helpers for Scenario objects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, List

from core.scenario.parser import parse_scenario
from core.utils.models import Scenario


def save_scenarios_to_json(
    scenarios: List[Scenario],
    path: str,
    indent: int = 2,
) -> None:
    """Save a list of Scenario objects to a JSON file."""
    serializable = [_to_dict(s) for s in scenarios]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=indent, ensure_ascii=False)


def load_scenarios_from_json(path: str) -> List[Scenario]:
    """Load scenarios from a JSON file and parse them into Scenario dataclasses."""
    with open(path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    if isinstance(raw_list, dict):
        raw_list = [raw_list]

    return [parse_scenario(item) for item in raw_list]


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and lists into serializable dictionaries."""
    if is_dataclass(obj):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
