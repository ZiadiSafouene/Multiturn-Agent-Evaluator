"""
Example runnable script demonstrating programmatic usage of AgentEvaluator.

Can be run via:
    python core/tests/example_run.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure repository root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.orchestrator import AgentEvaluator, EvaluatorConfig
from core.utils.models import Language, Persona


def main():
    print("=" * 60)
    print("   AgentEval: Multi-turn Multilingual Agent Evaluator")
    print("=" * 60)

    # User specifies their own agent model & API key, or relies on defaults (free model)
    config = EvaluatorConfig(
        agent_api_key=os.getenv("AGENT_API_KEY"),
        agent_model="mistralai/mistral-7b-instruct:free",
        agent_system_prompt="You are a professional banking assistant. Guide users politely.",
        tester_api_key=os.getenv("TESTER_API_KEY"),
        tester_model="mistralai/mistral-7b-instruct:free",
        languages=[Language.ENGLISH, Language.ARABIC, Language.FRENCH],
        personas=[Persona.COOPERATIVE, Persona.CONFUSED, Persona.ANGRY],
        test_cases_per_scenario=1,
        max_turns=6,
        use_llm_judge=False,
    )

    evaluator = AgentEvaluator(config)
    print("Running evaluation suite on test_scenarios.json...")
    results = evaluator.evaluate("test_scenarios.json")

    print("\n--- RESULTS ---")
    print(f"Total Tests : {results.total_tests}")
    print(f"Passed      : {results.passed_tests}")
    print(f"Failed      : {results.failed_tests}")
    print(f"Average     : {results.average_score:.1f}/100")
    print(f"Languages   : {results.scores_by_language}")


if __name__ == "__main__":
    main()

