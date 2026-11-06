"""Unit tests for scenario parsing, validation, and state machine."""

import pytest
from core.scenario.parser import parse_scenario
from core.scenario.validator import validate_scenario
from core.scenario.state_machine import ScenarioStateMachine
from core.scenario.storage import load_scenarios_from_json
from core.utils.models import Scenario


@pytest.fixture
def sample_scenario_dict():
    return {
        "scenario_id": "TEST_001",
        "title": "Balance Check",
        "description": "User checks balance",
        "goal": "Retrieve account balance",
        "domain": "banking",
        "slots": [
            {"name": "account_id", "slot_type": "string", "required": True, "aliases": ["numéro de compte", "رقم الحساب"]},
            {"name": "pin", "slot_type": "number", "required": True, "aliases": ["code"]},
        ],
        "states": [
            {
                "state_id": "collect_account",
                "required_slots": ["account_id"],
                "next_states": ["verify_pin"],
                "is_terminal": False,
            },
            {
                "state_id": "verify_pin",
                "required_slots": ["pin"],
                "next_states": ["show_balance"],
                "is_terminal": False,
            },
            {
                "state_id": "show_balance",
                "required_slots": [],
                "next_states": [],
                "is_terminal": True,
            },
        ],
        "transitions": [
            {
                "from_state": "collect_account",
                "to_state": "verify_pin",
                "trigger_intent": "provide_account",
                "conditions": {},
            },
            {
                "from_state": "verify_pin",
                "to_state": "show_balance",
                "trigger_intent": "provide_pin",
                "conditions": {},
            },
        ],
        "turns": [
            {
                "state": "collect_account",
                "expected_intent": "provide_account",
                "agent_action": "ask",
                "role": "user",
                "intent": "provide_account",
            }
        ],
    }


def test_parse_scenario(sample_scenario_dict):
    scenario = parse_scenario(sample_scenario_dict)
    assert scenario.scenario_id == "TEST_001"
    assert len(scenario.slots) == 2
    assert len(scenario.states) == 3
    assert len(scenario.transitions) == 2
    assert scenario.slots[0].aliases == ["numéro de compte", "رقم الحساب"]


def test_validate_scenario_success(sample_scenario_dict):
    scenario = parse_scenario(sample_scenario_dict)
    result = validate_scenario(scenario)
    assert result.valid is True
    assert len(result.errors) == 0


def test_validate_scenario_broken_transition(sample_scenario_dict):
    sample_scenario_dict["transitions"].append({
        "from_state": "non_existent",
        "to_state": "verify_pin",
        "trigger_intent": "bad",
    })
    scenario = parse_scenario(sample_scenario_dict)
    result = validate_scenario(scenario)
    assert result.valid is False
    assert any("unknown source state" in err for err in result.errors)


def test_state_machine_execution(sample_scenario_dict):
    scenario = parse_scenario(sample_scenario_dict)
    sm = ScenarioStateMachine(scenario)

    assert sm.current_state_id == "collect_account"
    assert sm.is_terminal is False

    # Missing account_id, cannot transition
    next_s = sm.step({})
    assert next_s == "collect_account"

    # Provide account_id, transitions to verify_pin
    next_s = sm.step({"account_id": "ACC123"})
    assert next_s == "verify_pin"

    # Provide pin, transitions to show_balance (terminal)
    next_s = sm.step({"account_id": "ACC123", "pin": "1234"})
    assert next_s == "show_balance"
    assert sm.is_terminal is True


def test_load_sample_json():
    scenarios = load_scenarios_from_json("test_scenarios.json")
    assert len(scenarios) == 3
    for s in scenarios:
        validation = validate_scenario(s)
        assert validation.valid is True
