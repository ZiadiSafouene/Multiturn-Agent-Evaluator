"""
core.orchestrator
~~~~~~~~~~~~~~~~~

High-level orchestration pipeline for multi-turn AI agent evaluation.

Coordinates:
1. Agent & Tester LLM client instantiation (with user OpenRouter API keys or free models)
2. Scenario loading or synthetic scenario generation
3. Multilingual test case synthesis with adversarial mutations and persona archetypes
4. Multi-turn conversational simulation
5. Dual-layer evaluation (deterministic rules + LLM-as-a-judge)
6. Comprehensive report aggregation and artifact export
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.evaluation.engine import evaluate_simulation
from core.llm.client import DEFAULT_CLIENT_MODEL, LLMClient
from core.llm.tester import DEFAULT_TESTER_MODEL, LLMTester
from core.mutation.generator import generate_test_cases
from core.scenario.generator import generate_scenarios_from_description
from core.scenario.storage import load_scenarios_from_json
from core.simulation.simulator import run_simulation
from core.utils.models import (
    EvaluationReport,
    EvaluationStatus,
    Language,
    Persona,
    Scenario,
    TestCase,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluatorConfig:
    """
    Configuration parameters for the agent evaluation pipeline.

    Attributes
    ----------
    agent_api_key:
        OpenRouter API key for the agent under test.
        Falls back to ``AGENT_API_KEY`` or ``OPENROUTER_API_KEY`` env vars.
    agent_model:
        OpenRouter model identifier for the agent under test.
        Defaults to free OpenRouter model :data:`DEFAULT_CLIENT_MODEL`.
    agent_system_prompt:
        System prompt defining the agent's identity, role, and domain rules.
    tester_api_key:
        OpenRouter API key for the tester/judge model.
        Falls back to ``TESTER_API_KEY`` or ``OPENROUTER_API_KEY`` env vars.
    tester_model:
        OpenRouter model identifier for user simulation & judging.
        Defaults to free OpenRouter model :data:`DEFAULT_TESTER_MODEL`.
    languages:
        List of conversation languages to evaluate (English, Arabic, French).
    personas:
        List of user persona archetypes to simulate.
    test_cases_per_scenario:
        Number of distinct test cases synthesized per scenario.
    max_turns:
        Maximum turn limit per simulated conversation.
    use_llm_judge:
        Whether to invoke qualitative LLM judging alongside deterministic checks.
    output_dir:
        Directory path where JSON reports and transcripts will be written.
    """

    agent_api_key: Optional[str] = None
    agent_model: str = DEFAULT_CLIENT_MODEL
    agent_system_prompt: Optional[str] = None

    tester_api_key: Optional[str] = None
    tester_model: str = DEFAULT_TESTER_MODEL

    languages: List[Language] = field(
        default_factory=lambda: [Language.ENGLISH, Language.ARABIC, Language.FRENCH]
    )
    personas: List[Persona] = field(
        default_factory=lambda: [
            Persona.COOPERATIVE,
            Persona.CONFUSED,
            Persona.IMPATIENT,
            Persona.ANGRY,
        ]
    )

    test_cases_per_scenario: int = 3
    max_turns: int = 8
    use_llm_judge: bool = True
    output_dir: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated Results Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationSuiteResult:
    """Aggregated metrics across an entire evaluation run."""

    total_tests: int = 0
    passed_tests: int = 0
    partial_tests: int = 0
    failed_tests: int = 0
    average_score: float = 0.0

    scores_by_language: Dict[str, float] = field(default_factory=dict)
    scores_by_persona: Dict[str, float] = field(default_factory=dict)
    scores_by_scenario: Dict[str, float] = field(default_factory=dict)

    reports: List[EvaluationReport] = field(default_factory=list)
    simulations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "partial_tests": self.partial_tests,
                "failed_tests": self.failed_tests,
                "pass_rate_pct": round(
                    (self.passed_tests / self.total_tests * 100) if self.total_tests else 0,
                    1,
                ),
                "average_score": self.average_score,
            },
            "breakdown": {
                "by_language": self.scores_by_language,
                "by_persona": self.scores_by_persona,
                "by_scenario": self.scores_by_scenario,
            },
            "reports": [r.to_dict() for r in self.reports],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Engine
# ─────────────────────────────────────────────────────────────────────────────

class AgentEvaluator:
    """
    Main entry point for evaluating multi-turn conversational agents.

    Parameters
    ----------
    config:
        Optional :class:`EvaluatorConfig`. If omitted, default configuration
        using free models is applied.
    """

    def __init__(self, config: Optional[EvaluatorConfig] = None) -> None:
        self.config = config or EvaluatorConfig()

        # Initialize LLM client for agent under test
        self.agent_client = LLMClient(
            api_key=self.config.agent_api_key,
            model=self.config.agent_model,
            system_prompt=self.config.agent_system_prompt,
        )

        # Initialize LLM tester for user simulation and qualitative judging
        self.tester_client = LLMTester(
            api_key=self.config.tester_api_key,
            model=self.config.tester_model,
        )

    def evaluate(
        self,
        scenarios: Union[List[Scenario], str, Path],
    ) -> EvaluationSuiteResult:
        """
        Execute evaluation against provided scenarios or scenario file.

        Parameters
        ----------
        scenarios:
            Can be:
            - A list of :class:`~core.utils.models.Scenario` objects.
            - A path string or Path object pointing to a JSON scenario file.
            - A natural-language description from which scenarios will be generated.

        Returns
        -------
        EvaluationSuiteResult
            Complete suite results with breakdowns and individual reports.
        """
        scenario_list = self._resolve_scenarios(scenarios)
        if not scenario_list:
            raise ValueError("No valid scenarios resolved for evaluation.")

        logger.info(
            "Starting evaluation across %d scenarios in [%s]...",
            len(scenario_list),
            ", ".join(l.value for l in self.config.languages),
        )

        suite_result = EvaluationSuiteResult()
        all_scores: List[float] = []

        lang_scores: Dict[str, List[float]] = {l.value: [] for l in self.config.languages}
        persona_scores: Dict[str, List[float]] = {p.value: [] for p in self.config.personas}
        scenario_scores: Dict[str, List[float]] = {}

        for scenario in scenario_list:
            scenario_scores[scenario.scenario_id] = []

            # 1. Generate test cases with mutations, personas, and languages
            test_cases = generate_test_cases(
                scenario=scenario,
                num_cases=self.config.test_cases_per_scenario,
                languages=self.config.languages,
                personas=self.config.personas,
            )

            for test_case in test_cases:
                # 2. Run multi-turn simulation
                sim_result = run_simulation(
                    scenario=scenario,
                    test_case=test_case,
                    llm_client=self.agent_client,
                    tester_client=self.tester_client,
                    max_turns=self.config.max_turns,
                    use_llm_judge=self.config.use_llm_judge,
                )

                report: EvaluationReport = sim_result["evaluation_report"]
                suite_result.reports.append(report)
                suite_result.simulations.append(sim_result)

                # Record statistics
                suite_result.total_tests += 1
                if report.status == EvaluationStatus.PASS:
                    suite_result.passed_tests += 1
                elif report.status == EvaluationStatus.PARTIAL:
                    suite_result.partial_tests += 1
                else:
                    suite_result.failed_tests += 1

                all_scores.append(report.final_score)
                lang_scores.setdefault(test_case.language.value, []).append(report.final_score)
                persona_scores.setdefault(test_case.persona.value, []).append(report.final_score)
                scenario_scores[scenario.scenario_id].append(report.final_score)

        # Aggregate averages
        suite_result.average_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

        suite_result.scores_by_language = {
            lang: round(sum(scores) / len(scores), 2)
            for lang, scores in lang_scores.items() if scores
        }
        suite_result.scores_by_persona = {
            persona: round(sum(scores) / len(scores), 2)
            for persona, scores in persona_scores.items() if scores
        }
        suite_result.scores_by_scenario = {
            sc_id: round(sum(scores) / len(scores), 2)
            for sc_id, scores in scenario_scores.items() if scores
        }

        # Export if configured
        if self.config.output_dir:
            self._save_results(suite_result)

        return suite_result

    def _resolve_scenarios(self, source: Union[List[Scenario], str, Path]) -> List[Scenario]:
        """Resolve scenarios from file path, natural language description, or list."""
        if isinstance(source, list) and all(isinstance(s, Scenario) for s in source):
            return source

        path_candidate = Path(source) if isinstance(source, (str, Path)) else None
        if path_candidate and path_candidate.exists() and path_candidate.is_file():
            logger.info("Loading scenarios from JSON file: %s", path_candidate)
            return load_scenarios_from_json(str(path_candidate))

        if isinstance(source, str):
            logger.info("Synthesizing scenarios from text description...")
            scenarios, errors = generate_scenarios_from_description(
                description=source,
                llm=self.tester_client,
            )
            if not scenarios:
                raise ValueError(f"Failed to generate scenarios from description: {errors}")
            return scenarios

        raise TypeError(f"Unsupported scenario input type: {type(source)}")

    def _save_results(self, result: EvaluationSuiteResult) -> None:
        """Save results and simulation traces to disk."""
        if not self.config.output_dir:
            return
        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        report_file = out_path / "evaluation_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Evaluation report saved to: %s", report_file)
