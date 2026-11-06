"""Unit tests for deterministic evaluation metrics and scoring."""

import pytest
from core.evaluation.engine import evaluate_deterministic
from core.scenario.storage import load_scenarios_from_json
from core.utils.models import Language, Mutation, MutationType, Persona, TestCase


@pytest.fixture
def banking_scenario():
    scenarios = load_scenarios_from_json("test_scenarios.json")
    return scenarios[0]


def test_deterministic_evaluation_happy_path(banking_scenario):
    test_case = TestCase(
        test_id="TC_01",
        scenario_id=banking_scenario.scenario_id,
        mutation=None,
        language=Language.ENGLISH,
        persona=Persona.COOPERATIVE,
        description="Happy path",
    )

    simulation_result = {
        "history": [
            {"turn": 0, "speaker": "agent", "state": "collect_account", "message": "Give account id"},
            {"turn": 0, "speaker": "user", "state": "collect_account", "message": "account_id: 1234"},
            {"turn": 1, "speaker": "agent", "state": "send_otp", "message": "OTP sent"},
            {"turn": 2, "speaker": "agent", "state": "verify_otp", "message": "Give otp"},
            {"turn": 2, "speaker": "user", "state": "verify_otp", "message": "otp_code: 9999"},
            {"turn": 3, "speaker": "agent", "state": "display_balance", "message": "Balance is 1000"},
        ],
        "summary": {
            "final_state": "display_balance",
            "termination_reason": "terminal_state",
            "memory_final": {"account_id": "1234", "otp_code": "9999"},
        },
    }

    eval_result = evaluate_deterministic(simulation_result, banking_scenario, test_case)
    assert eval_result.task_completed is True
    assert eval_result.slot_accuracy == 1.0
    assert len(eval_result.missing_slots) == 0
    assert len(eval_result.state_errors) == 0
    assert eval_result.score >= 80.0


def test_deterministic_evaluation_missing_slots(banking_scenario):
    test_case = TestCase(
        test_id="TC_02",
        scenario_id=banking_scenario.scenario_id,
        mutation=Mutation(
            type=MutationType.SLOT_MISSING,
            target="otp_code",
            turn_index=1,
            description="User does not provide OTP",
            expected_behavior="Agent re-asks",
        ),
        language=Language.FRENCH,
        persona=Persona.CONFUSED,
        description="Missing slot test",
    )

    simulation_result = {
        "history": [
            {"turn": 0, "speaker": "agent", "state": "collect_account", "message": "Give account id"},
            {"turn": 0, "speaker": "user", "state": "collect_account", "message": "account_id: 1234"},
            {"turn": 1, "speaker": "agent", "state": "verify_otp", "message": "Give otp"},
            {"turn": 1, "speaker": "user", "state": "verify_otp", "message": "I don't know"},
        ],
        "summary": {
            "final_state": "verify_otp",
            "termination_reason": "no_progress",
            "memory_final": {"account_id": "1234"},
        },
    }

    eval_result = evaluate_deterministic(simulation_result, banking_scenario, test_case)
    assert eval_result.task_completed is False
    assert "otp_code" in eval_result.missing_slots
    assert eval_result.score < 70.0
