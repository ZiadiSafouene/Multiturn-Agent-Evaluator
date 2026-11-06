"""
core.scenario.parser
~~~~~~~~~~~~~~~~~~~~

Parsing utilities to construct domain dataclasses from raw dictionaries or JSON.
"""

from __future__ import annotations

from typing import Any, Dict

from core.utils.models import (
    Scenario,
    SlotDefinition,
    State,
    StateTransition,
    TurnTemplate,
)


def parse_scenario(data: Dict[str, Any]) -> Scenario:
    """
    Parse a raw dictionary into a typed :class:`~core.utils.models.Scenario` instance.

    Parameters
    ----------
    data:
        Dictionary parsed from JSON scenario schema.

    Returns
    -------
    Scenario
        A validated Scenario dataclass instance.
    """
    slots = [_parse_slot(s) for s in data.get("slots", [])]
    states = [_parse_state(s) for s in data.get("states", [])]
    transitions = [_parse_transition(t) for t in data.get("transitions", [])]
    turns = [_parse_turn(t) for t in data.get("turns", [])]

    return Scenario(
        scenario_id=data.get("scenario_id", "unknown"),
        title=data.get("title", ""),
        description=data.get("description", ""),
        goal=data.get("goal", ""),
        domain=data.get("domain", ""),
        slots=slots,
        states=states,
        transitions=transitions,
        turns=turns,
        edge_cases=data.get("edge_cases", []),
        policies=data.get("policies", []),
        metadata=data.get("metadata", {}),
    )


def _parse_slot(data: Dict[str, Any]) -> SlotDefinition:
    aliases = data.get("aliases") or data.get("arabic_aliases") or []
    return SlotDefinition(
        name=data["name"],
        slot_type=data.get("slot_type", "string"),
        required=data.get("required", True),
        pattern=data.get("pattern"),
        minvalue=data.get("minvalue"),
        maxvalue=data.get("maxvalue"),
        valid_values=data.get("valid_values", []),
        description=data.get("description", ""),
        aliases=aliases,
    )


def _parse_state(data: Dict[str, Any]) -> State:
    return State(
        state_id=data["state_id"],
        required_slots=data.get("required_slots", []),
        expected_actions=data.get("expected_actions", []),
        next_states=data.get("next_states", []),
        policy_checks=data.get("policy_checks", []),
        is_terminal=data.get("is_terminal", False),
    )


def _parse_transition(data: Dict[str, Any]) -> StateTransition:
    return StateTransition(
        from_state=data["from_state"],
        to_state=data["to_state"],
        trigger_intent=data["trigger_intent"],
        conditions=data.get("conditions", {}),
    )


def _parse_turn(data: Dict[str, Any]) -> TurnTemplate:
    return TurnTemplate(
        state=data["state"],
        expected_intent=data["expected_intent"],
        agent_action=data.get("agent_action", "ask"),
        slots_to_fill=data.get("slots_to_fill", []),
        role=data.get("role", "user"),
        intent=data.get("intent", data.get("expected_intent", "")),
    )
