"""
orchestrator.py
~~~~~~~~~~~~~~~

Root-level convenience entry point exposing AgentEvaluator and CLI evaluation flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.orchestrator import AgentEvaluator, EvaluatorConfig, EvaluationSuiteResult
from core.utils.models import Language, Persona

__all__ = [
    "AgentEvaluator",
    "EvaluatorConfig",
    "EvaluationSuiteResult",
    "Language",
    "Persona",
]


def run_evaluation_quickstart(
    scenarios_path: str = "test_scenarios.json",
    agent_api_key: str | None = None,
    agent_model: str | None = None,
    agent_system_prompt: str | None = None,
    tester_api_key: str | None = None,
    tester_model: str | None = None,
    use_llm_judge: bool = False,
) -> EvaluationSuiteResult:
    """Convenience helper to run an evaluation with a few keyword arguments."""
    config = EvaluatorConfig(
        agent_api_key=agent_api_key,
        agent_model=agent_model or "mistralai/mistral-7b-instruct:free",
        agent_system_prompt=agent_system_prompt or "You are a helpful and polite banking agent.",
        tester_api_key=tester_api_key,
        tester_model=tester_model or "mistralai/mistral-7b-instruct:free",
        use_llm_judge=use_llm_judge,
        test_cases_per_scenario=2,
    )
    evaluator = AgentEvaluator(config)
    return evaluator.evaluate(scenarios_path)


if __name__ == "__main__":
    from core.cli import app
    app()
