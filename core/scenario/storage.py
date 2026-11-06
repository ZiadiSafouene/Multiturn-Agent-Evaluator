from __future__ import annotations
import json
from typing import List
from dataclasses import asdict, is_dataclass

from ai_engine.utils.models import Scenario


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def save_scenarios_to_json(
    scenarios: List[Scenario],
    path: str,
    indent: int = 2
) -> None:
    """
    Save a list of Scenario objects to a JSON file.
    """

    serializable = [_to_dict(s) for s in scenarios]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=indent, ensure_ascii=False)


def load_scenarios_from_json(
    path: str,
):
    """
    Load scenarios from JSON file (raw dicts).
    You can then pass them to your parser.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def _to_dict(obj):
    """
    Recursively convert dataclasses / objects to dict.
    """

    if is_dataclass(obj):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}

    elif isinstance(obj, list):
        return [_to_dict(i) for i in obj]

    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}

    else:
        return obj