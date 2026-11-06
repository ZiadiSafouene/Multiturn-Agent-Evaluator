"""Unit tests for AgentEvaluator and end-to-end evaluation pipeline."""

import pytest
from core.orchestrator import AgentEvaluator, EvaluatorConfig
from core.utils.models import Language, Persona


class MockAgentClient:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        prompt_l = prompt.lower()
        if "collect_account" in prompt_l:
            return "Please provide your account_id."
        if "verify_otp" in prompt_l:
            return "Please enter your otp_code."
        if "gather_transfer_details" in prompt_l:
            return "Please provide transfer details."
        if "initiate_report" in prompt_l:
            return "Please provide fraud report details."
        if "verify_identity" in prompt_l:
            return "Please provide security answer."
        return "Transaction completed."


class MockTesterClient:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        prompt_l = prompt.lower()
        if "collect_account" in prompt_l or "account_id" in prompt_l:
            return "account_id: ACC_TEST_01"
        if "otp" in prompt_l:
            return "otp_code: 12345"
        if "gather_transfer_details" in prompt_l or "transfer" in prompt_l:
            return "source_account: SA_01, destination_account: DA_02, transfer_amount: 500"
        if "fraud" in prompt_l or "initiate_report" in prompt_l:
            return "account_number: ACC_99, transaction_id: TX_101"
        if "security_answer" in prompt_l:
            return "security_answer: Blue"
        return "Thank you."


def test_agent_evaluator_end_to_end(monkeypatch, tmp_path):
    config = EvaluatorConfig(
        languages=[Language.ENGLISH],
        personas=[Persona.COOPERATIVE],
        test_cases_per_scenario=1,
        max_turns=5,
        use_llm_judge=False,
        output_dir=str(tmp_path),
    )

    evaluator = AgentEvaluator(config)
    # Monkeypatch clients with mock instances
    evaluator.agent_client = MockAgentClient()
    evaluator.tester_client = MockTesterClient()

    results = evaluator.evaluate("test_scenarios.json")

    assert results.total_tests == 3
    assert results.passed_tests >= 2
    assert results.average_score >= 70.0
    assert "english" in results.scores_by_language

    # Verify JSON report export
    saved_report = tmp_path / "evaluation_report.json"
    assert saved_report.exists()
