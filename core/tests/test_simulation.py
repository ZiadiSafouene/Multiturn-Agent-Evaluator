"""Unit tests for multi-turn simulation and multilingual slot extraction."""

import pytest
from core.scenario.storage import load_scenarios_from_json
from core.simulation.simulator import extract_slots_from_text, resolve_next_state, run_simulation
from core.utils.models import Language, Persona, TestCase


class MockLLM:
    """Mock LLM that returns deterministic responses based on prompt keywords."""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, system: str = "") -> str:
        self.call_count += 1
        prompt_lower = prompt.lower()

        # If agent prompt asking for details
        if "generate only the agent" in prompt_lower:
            if "collect_account" in prompt_lower:
                return "Please provide your account_id."
            if "verify_otp" in prompt_lower:
                return "Please enter your otp_code."
            return "Your balance is 5000 USD."

        # If simulated user prompt
        if "generate only the user" in prompt_lower:
            if "collect_account" in prompt_lower:
                return "My account_id is ACC_9988."
            if "verify_otp" in prompt_lower:
                return "My otp_code is 4455."
            return "Thank you, goodbye."

        return "OK"


@pytest.fixture
def banking_scenario():
    scenarios = load_scenarios_from_json("test_scenarios.json")
    return scenarios[0]


def test_slot_extraction_english(banking_scenario):
    memory = {}
    text = "Here is my account_id: ACC_12345"
    slots = extract_slots_from_text(text, banking_scenario, memory)
    assert slots.get("account_id") == "ACC_12345"


def test_slot_extraction_french(banking_scenario):
    memory = {}
    text = "mon account_id est FR98765"
    slots = extract_slots_from_text(text, banking_scenario, memory)
    assert slots.get("account_id") == "FR98765"


def test_slot_extraction_arabic(banking_scenario):
    memory = {}
    text = "account_id: 11223344"
    slots = extract_slots_from_text(text, banking_scenario, memory)
    assert slots.get("account_id") == "11223344"


def test_resolve_next_state_gating(banking_scenario):
    # Missing required slot account_id
    memory = {}
    next_s = resolve_next_state("collect_account", memory, banking_scenario)
    assert next_s == "collect_account"

    # With account_id provided and account_valid=True condition
    memory = {"account_id": "ACC_1", "account_valid": True}
    next_s = resolve_next_state("collect_account", memory, banking_scenario)
    assert next_s == "send_otp"


def test_run_simulation_with_mock_llm(banking_scenario):
    mock_agent = MockLLM()
    mock_tester = MockLLM()

    test_case = TestCase(
        test_id="TC_MOCK_01",
        scenario_id=banking_scenario.scenario_id,
        mutation=None,
        language=Language.ENGLISH,
        persona=Persona.COOPERATIVE,
        description="Mock test nominal",
        initial_slots={"account_valid": True, "otp_valid": True},
    )

    result = run_simulation(
        scenario=banking_scenario,
        test_case=test_case,
        llm_client=mock_agent,
        tester_client=mock_tester,
        max_turns=6,
        use_llm_judge=False,
    )

    assert "history" in result
    assert "summary" in result
    assert result["summary"]["test_id"] == "TC_MOCK_01"
    assert result["summary"]["language"] == "english"
    assert len(result["history"]) > 0

    evaluation = result["summary"]["evaluation"]
    assert "deterministic" in evaluation
    assert evaluation["deterministic"]["score"] >= 70.0
