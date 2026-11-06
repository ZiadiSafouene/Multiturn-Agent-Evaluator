"""
AgentEval (core)
~~~~~~~~~~~~~~~~

A professional multi-turn conversational AI agent evaluation framework.
Supports English, Arabic, and French with adversarial mutations, personas,
OpenRouter integration, and deterministic + LLM-judge metrics.
"""

from core.__version__ import __version__
from core.orchestrator import AgentEvaluator, EvaluatorConfig, EvaluationSuiteResult
from core.llm.client import LLMClient
from core.llm.tester import LLMTester
from core.utils.models import (
    DeterministicResult,
    EvaluationReport,
    EvaluationStatus,
    Language,
    LLMJudgeResult,
    Mutation,
    MutationType,
    Persona,
    Scenario,
    TestCase,
)

__all__ = [
    "__version__",
    "AgentEvaluator",
    "EvaluatorConfig",
    "EvaluationSuiteResult",
    "LLMClient",
    "LLMTester",
    "Scenario",
    "TestCase",
    "Mutation",
    "MutationType",
    "Language",
    "Persona",
    "DeterministicResult",
    "LLMJudgeResult",
    "EvaluationReport",
    "EvaluationStatus",
]
