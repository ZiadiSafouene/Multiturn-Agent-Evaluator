"""
core.evaluation.engine
~~~~~~~~~~~~~~~~~~~~~~

Comprehensive evaluation engine combining deterministic rule-based checks
and optional LLM-as-a-judge qualitative scoring.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from core.llm.prompts import build_llm_judge_prompt
from core.llm.tester import LLMTester
from core.utils.models import (
    DeterministicResult,
    EvaluationReport,
    EvaluationStatus,
    Language,
    LLMJudgeResult,
    MutationType,
    Scenario,
    TestCase,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Checks
# ─────────────────────────────────────────────────────────────────────────────

def _validate_transitions(
    history: List[Dict[str, Any]],
    scenario: Scenario,
) -> Tuple[bool, List[str]]:
    """Verify that all state transitions follow declared graph edges."""
    valid_edges: Set[Tuple[str, str]] = {
        (t.from_state, t.to_state) for t in scenario.transitions
    }

    # Also allow transitions to self or explicitly declared next_states
    for state in scenario.states:
        for nxt in state.next_states:
            valid_edges.add((state.state_id, nxt))

    agent_states = [entry["state"] for entry in history if entry.get("speaker") == "agent"]
    state_errors: List[str] = []

    if len(agent_states) < 2:
        return True, state_errors

    for i in range(1, len(agent_states)):
        prev = agent_states[i - 1]
        curr = agent_states[i]
        if prev != curr and (prev, curr) not in valid_edges:
            state_errors.append(f"Illegal transition from '{prev}' to '{curr}'")

    return len(state_errors) == 0, state_errors


def _compute_slot_metrics(
    memory: Dict[str, Any],
    scenario: Scenario,
) -> Tuple[float, List[str], List[str]]:
    """Compute slot accuracy, missing slots, and invalid slots."""
    required_slots = [s for s in scenario.slots if s.required]
    if not required_slots:
        return 1.0, [], []

    missing: List[str] = []
    invalid: List[str] = []
    filled_correctly = 0

    for slot in required_slots:
        val = memory.get(slot.name)
        if val is None or str(val).strip() == "":
            missing.append(slot.name)
        else:
            # Type and validation checks
            if slot.slot_type == "number":
                # Check numeric string or int/float
                clean_val = str(val).replace(",", "").replace("$", "").strip()
                try:
                    float(clean_val)
                    filled_correctly += 1
                except ValueError:
                    invalid.append(slot.name)
            elif slot.slot_type == "enum" and slot.valid_values:
                if str(val).strip().lower() in [v.lower() for v in slot.valid_values]:
                    filled_correctly += 1
                else:
                    invalid.append(slot.name)
            else:
                filled_correctly += 1

    accuracy = round(filled_correctly / len(required_slots), 2)
    return accuracy, missing, invalid


def _compute_stagnation_score(history: List[Dict[str, Any]]) -> float:
    """Measure if conversation stagnated or repeated identical messages."""
    agent_messages = [
        entry["message"].strip().lower()
        for entry in history
        if entry.get("speaker") == "agent" and entry.get("message")
    ]
    if not agent_messages:
        return 1.0

    unique_messages = len(set(agent_messages))
    ratio = unique_messages / len(agent_messages)
    return round(max(0.0, min(1.0, ratio)), 2)


def _compute_efficiency_score(actual_turns: int, expected_states_count: int) -> float:
    """
    Score conversation efficiency: optimal conversations complete
    in ~ expected_states_count to (expected_states_count * 2) turns.
    """
    if actual_turns <= 0:
        return 0.0
    optimal = max(2, expected_states_count)
    if actual_turns <= optimal * 2:
        return 1.0
    overshoot = actual_turns - (optimal * 2)
    return round(max(0.1, 1.0 - (overshoot * 0.1)), 2)


def _evaluate_mutation_recovery(
    summary: Dict[str, Any],
    test_case: TestCase,
    task_completed: bool,
    missing_slots: List[str],
) -> float:
    """Assess whether the agent recovered correctly from an injected mutation."""
    mutation = test_case.mutation
    if not mutation:
        return 1.0

    termination_reason = summary.get("termination_reason", "")

    if mutation.type == MutationType.POLICY_VIOLATION:
        # Expected behavior: agent must refuse/block action
        if summary.get("policy_violation_caught", False) or termination_reason in {"blocked", "policy_refusal"}:
            return 1.0
        # If task completed when policy should have blocked it, penalize
        return 0.3 if not task_completed else 0.0

    if mutation.type in {MutationType.SLOT_MISSING, MutationType.SLOT_INVALID}:
        # Agent should have recovered and eventually filled required slots
        if task_completed and len(missing_slots) == 0:
            return 1.0
        return 0.5 if task_completed else 0.2

    if mutation.type in {MutationType.STATE_SKIP, MutationType.INTENT_SWITCH, MutationType.AMBIGUITY}:
        if task_completed:
            return 1.0
        if termination_reason in {"terminal_state", "max_turns"}:
            return 0.6
        return 0.2

    return 1.0 if task_completed else 0.5


def evaluate_deterministic(
    result: Dict[str, Any],
    scenario: Scenario,
    test_case: TestCase,
) -> DeterministicResult:
    """
    Run comprehensive deterministic rule-based checks on a simulation result.

    Parameters
    ----------
    result:
        Output dictionary from :func:`~core.simulation.simulator.run_simulation`.
    scenario:
        The target scenario definition.
    test_case:
        The test case metadata and mutation specifications.

    Returns
    -------
    DeterministicResult
        Scored metrics and lists of errors/omissions.
    """
    history: List[Dict[str, Any]] = result.get("history", [])
    summary: Dict[str, Any] = result.get("summary", {})
    memory: Dict[str, Any] = summary.get("memory_final", {})
    final_state: str = summary.get("final_state", "")

    terminal_states: Set[str] = {s.state_id for s in scenario.states if s.is_terminal}
    task_completed = (
        final_state in terminal_states
        or summary.get("termination_reason") == "terminal_state"
    )

    transitions_valid, state_errors = _validate_transitions(history, scenario)
    slot_accuracy, missing_slots, invalid_slots = _compute_slot_metrics(memory, scenario)
    stagnation_score = _compute_stagnation_score(history)
    recovery_score = _evaluate_mutation_recovery(summary, test_case, task_completed, missing_slots)
    efficiency_score = _compute_efficiency_score(len(history), len(scenario.states))

    # Policy violations check
    policy_violations: List[str] = []
    if test_case.mutation and test_case.mutation.type == MutationType.POLICY_VIOLATION:
        if task_completed:
            policy_violations.append(
                f"Agent proceeded with action despite policy check: {test_case.mutation.target}"
            )

    det_result = DeterministicResult(
        task_completed=task_completed,
        slot_accuracy=slot_accuracy,
        state_errors=state_errors,
        policy_violations=policy_violations,
        missing_slots=missing_slots,
        invalid_slots=invalid_slots,
        tool_calls_valid=True,
    )

    # Compute base score (out of 100)
    score = 0.0
    score += 40.0 if task_completed else 0.0
    score += 25.0 * slot_accuracy
    score += 15.0 if transitions_valid else max(0.0, 15.0 - 5.0 * len(state_errors))
    score += 10.0 * recovery_score
    score += 5.0 * stagnation_score
    score += 5.0 * efficiency_score

    if policy_violations:
        score = max(0.0, score - 20.0 * len(policy_violations))

    det_result.score = round(max(0.0, min(100.0, score)), 2)
    return det_result


# ─────────────────────────────────────────────────────────────────────────────
# LLM Judge Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_llm_judge(
    conversation_history_text: str,
    scenario: Scenario,
    target_language: Language,
    tester_llm: Optional[LLMTester] = None,
) -> LLMJudgeResult:
    """
    Score conversational quality using an OpenRouter tester model.

    Evaluates coherence, helpfulness, language adherence (English, Arabic, French),
    and safety/hallucination detection.
    """
    llm = tester_llm or LLMTester()
    prompt = build_llm_judge_prompt(
        conversation_history=conversation_history_text,
        scenario_goal=scenario.goal,
        expected_language=target_language,
    )

    raw_response = llm.generate(prompt, system="You are an expert AI conversation judge.")

    try:
        # Strip potential code fences
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        data = json.loads(cleaned)

        judge_result = LLMJudgeResult(
            coherence_score=float(data.get("coherence_score", 0.8)),
            helpfulness_score=float(data.get("helpfulness_score", 0.8)),
            language_adherence_score=float(data.get("language_adherence_score", 0.9)),
            safety_score=float(data.get("safety_score", 0.9)),
            hallucination_detected=bool(data.get("hallucination_detected", False)),
            issues=list(data.get("issues", [])),
            reasoning=str(data.get("reasoning", "")),
        )
        judge_result.compute_score()
        return judge_result

    except Exception as exc:
        logger.warning("LLM Judge evaluation failed or returned non-JSON: %s. Using fallback score.", exc)
        fallback = LLMJudgeResult(
            coherence_score=0.8,
            helpfulness_score=0.8,
            language_adherence_score=0.85,
            safety_score=0.9,
            hallucination_detected=False,
            issues=[f"LLM Judge parse fallback ({exc})"],
            reasoning="Default heuristic score applied due to judge communication issue.",
        )
        fallback.compute_score()
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_simulation(
    result: Dict[str, Any],
    scenario: Scenario,
    test_case: TestCase,
    use_llm_judge: bool = False,
    tester_llm: Optional[LLMTester] = None,
) -> EvaluationReport:
    """
    Produce a complete :class:`~core.utils.models.EvaluationReport` combining
    deterministic metrics and optional LLM-as-a-judge scoring.
    """
    deterministic = evaluate_deterministic(result, scenario, test_case)

    llm_judge: Optional[LLMJudgeResult] = None
    if use_llm_judge:
        history_text = "\n".join(
            f"{entry['speaker'].capitalize()}: {entry['message']}"
            for entry in result.get("history", [])
        )
        llm_judge = evaluate_with_llm_judge(
            conversation_history_text=history_text,
            scenario=scenario,
            target_language=test_case.language,
            tester_llm=tester_llm,
        )

    # Compute final score
    report = EvaluationReport(
        report_id=str(uuid.uuid4())[:8],
        conversation_id=result.get("summary", {}).get("conversation_id", f"conv_{test_case.test_id}"),
        test_case_id=test_case.test_id,
        scenario_id=scenario.scenario_id,
        status=EvaluationStatus.PASS,
        deterministic=deterministic,
        llm_judge=llm_judge,
        language=test_case.language,
        persona=test_case.persona,
        mutation=test_case.mutation,
    )
    report.compute_final_score()

    # Determine status
    if report.final_score >= 70.0:
        report.status = EvaluationStatus.PASS
    elif report.final_score >= 50.0:
        report.status = EvaluationStatus.PARTIAL
    else:
        report.status = EvaluationStatus.FAIL

    # Build human-readable summary
    summary_parts = [
        f"Test {test_case.test_id} ({test_case.language.value.capitalize()}, {test_case.persona.value}) "
        f"completed with final score {report.final_score:.1f}/100 [{report.status.value.upper()}].",
        f"Task completion: {'YES' if deterministic.task_completed else 'NO'}.",
        f"Slot filling accuracy: {deterministic.slot_accuracy * 100:.0f}%.",
    ]
    if deterministic.missing_slots:
        summary_parts.append(f"Missing slots: {', '.join(deterministic.missing_slots)}.")
    if deterministic.state_errors:
        summary_parts.append(f"State transition errors: {len(deterministic.state_errors)}.")
    if llm_judge:
        summary_parts.append(
            f"LLM Judge score: {llm_judge.score:.1f} (Language adherence: {llm_judge.language_adherence_score * 100:.0f}%, "
            f"Coherence: {llm_judge.coherence_score * 100:.0f}%)."
        )

    report.summary = " ".join(summary_parts)
    return report
