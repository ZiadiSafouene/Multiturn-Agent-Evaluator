"""
core.simulation.simulator
~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-turn conversation simulator driving an automated dialogue between
the agent under test and a simulated persona user across English, Arabic, and French.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from core.evaluation.engine import evaluate_simulation
from core.llm.client import LLMClient
from core.llm.prompts import (
    build_agent_prompt,
    build_user_prompt,
    get_language_instruction,
)
from core.llm.tester import LLMTester
from core.persona.engine import PersonaEngine
from core.utils.models import Language, Scenario, TestCase

logger = logging.getLogger(__name__)

MAX_TURNS = 8
STAGNATION_WINDOW = 3
NO_PROGRESS_LIMIT = 3


# ─────────────────────────────────────────────────────────────────────────────
# Multilingual Slot Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def extract_slots_from_text(
    text: str,
    scenario: Scenario,
    memory: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract slot values from user utterance using key-value heuristics,
    declared aliases across English, Arabic, and French, and single-slot fallbacks.

    Parameters
    ----------
    text:
        User message string.
    scenario:
        Active :class:`~core.utils.models.Scenario`.
    memory:
        Currently populated slot memory.

    Returns
    -------
    Dict[str, Any]
        Newly discovered slot name -> value mappings.
    """
    extracted: Dict[str, Any] = {}
    expected_map: Dict[str, str] = {}

    for slot in scenario.slots:
        norm = _normalize_name(slot.name)
        expected_map[norm] = slot.name
        for alias in slot.aliases:
            expected_map[_normalize_name(alias)] = slot.name

    # 1. Key-value captures: "key: value" or "key = value" or "key est value"
    kv_pattern = r"([a-zA-Z\u0600-\u06FF_][a-zA-Z\u0600-\u06FF0-9_ ]{1,40})\s*[:=]\s*([^\n,;]+)"
    for key, value in re.findall(kv_pattern, text):
        norm_key = _normalize_name(key)
        if norm_key in expected_map:
            extracted[expected_map[norm_key]] = value.strip()

    # 2. English / French / Arabic verbal anchors:
    # "account is 12345", "mon compte est 12345", "رقم الحساب هو 12345"
    for norm_key, slot_name in expected_map.items():
        if slot_name in extracted or slot_name in memory:
            continue
        anchor_pattern = (
            rf"\b{re.escape(norm_key).replace('_', r'[_\s]+')}\b"
            rf"(?:\s*(?:is|est|هو|هي|\:|\=))?\s+([^\n,;]+)"
        )
        match = re.search(anchor_pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Clean trailing punctuation
            candidate = candidate.rstrip(".!?,")
            if candidate:
                extracted[slot_name] = candidate

    # 3. Numeric capture for lone numbers if exactly one number slot is missing
    missing_num_slots = [
        s.name for s in scenario.slots
        if s.required and s.slot_type == "number"
        and s.name not in memory and s.name not in extracted
    ]
    if len(missing_num_slots) == 1:
        num_match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if num_match:
            extracted[missing_num_slots[0]] = num_match.group(1)

    # 4. Fallback: if exactly one required slot is missing and user utterance is short,
    # treat the whole message as the slot value.
    missing_required = [
        s.name for s in scenario.slots
        if s.required and s.name not in memory and s.name not in extracted
    ]
    if len(missing_required) == 1 and text.strip() and len(text.split()) <= 6:
        clean_text = text.strip().rstrip(".!?")
        if not any(k in clean_text.lower() for k in ["hello", "bonjour", "مرحبا", "hi", "help"]):
            extracted[missing_required[0]] = clean_text

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# State Resolution & Transition Mechanics
# ─────────────────────────────────────────────────────────────────────────────

def _value_matches_condition(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return value in expected
    if isinstance(expected, bool):
        if isinstance(value, str):
            low = value.strip().lower()
            actual = low in {"true", "1", "yes", "oui", "نعم"}
        else:
            actual = bool(value)
        return actual is expected
    return str(value).strip().lower() == str(expected).strip().lower()


def resolve_next_state(current_state: str, memory: Dict[str, Any], scenario: Scenario) -> str:
    state_map = {s.state_id: s for s in scenario.states}
    state = state_map.get(current_state)
    if not state or state.is_terminal:
        return current_state

    # Prerequisite: required slots for the current state must be filled
    if not all(slot in memory and str(memory[slot]).strip() for slot in state.required_slots):
        return current_state

    # Check conditional transitions first
    transitions = [t for t in scenario.transitions if t.from_state == current_state]
    for transition in transitions:
        conditions = transition.conditions or {}
        if conditions and all(_value_matches_condition(memory.get(k), exp) for k, exp in conditions.items()):
            return transition.to_state

    # Check unconditional transitions
    for transition in transitions:
        if not transition.conditions:
            return transition.to_state

    # Fallback to next_states definition
    if state.next_states:
        return state.next_states[0]

    return current_state


def is_stuck(history: List[Dict[str, Any]], window: int = STAGNATION_WINDOW) -> bool:
    """Detect conversational stagnation across a sliding window."""
    if len(history) < window * 2:
        return False
    recent_agent = [
        entry["message"].strip()
        for entry in history[-window * 2 :]
        if entry["speaker"] == "agent"
    ]
    if len(recent_agent) >= window and len(set(recent_agent)) == 1:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    scenario: Scenario,
    test_case: TestCase,
    llm_client: Optional[LLMClient] = None,
    tester_client: Optional[LLMTester] = None,
    build_agent_prompt_fn: Optional[Callable] = None,
    build_user_prompt_fn: Optional[Callable] = None,
    max_turns: int = MAX_TURNS,
    use_llm_judge: bool = False,
) -> Dict[str, Any]:
    """
    Execute a multi-turn simulation between an agent client and a user tester.

    Parameters
    ----------
    scenario:
        The target scenario definition.
    test_case:
        The test case specifying persona, mutation, and language (English, Arabic, French).
    llm_client:
        The client driving the agent under test (defaults to free OpenRouter model).
    tester_client:
        The tester client driving the simulated user and judge (defaults to free OpenRouter model).
    build_agent_prompt_fn:
        Optional prompt builder override for the agent.
    build_user_prompt_fn:
        Optional prompt builder override for the simulated user.
    max_turns:
        Maximum turn limit before forced termination.
    use_llm_judge:
        Whether to run qualitative LLM judging during evaluation.

    Returns
    -------
    Dict[str, Any]
        Complete simulation output with full turn history, memory, and evaluation report.
    """
    agent = llm_client or LLMClient()
    tester = tester_client or LLMTester()
    persona_engine = PersonaEngine(tester)

    agent_prompt_builder = build_agent_prompt_fn or build_agent_prompt
    user_prompt_builder = build_user_prompt_fn or build_user_prompt

    state_map = {s.state_id: s for s in scenario.states}
    current_state = scenario.states[0].state_id if scenario.states else "start"
    memory: Dict[str, Any] = dict(test_case.initial_slots or {})
    history: List[Dict[str, Any]] = []
    termination_reason = "max_turns"

    no_progress_counter = 0
    mutation_turns_used = 0

    for turn_index in range(max_turns):
        state = state_map.get(current_state)
        if not state:
            termination_reason = "state_not_found"
            break

        # ─── 1. Agent Turn ────────────────────────────────────────────────
        agent_prompt = agent_prompt_builder(
            scenario=scenario,
            state=state,
            memory=memory,
            history=history,
        )
        agent_msg = agent.generate(agent_prompt).strip()

        history.append(
            {
                "turn": turn_index,
                "state": current_state,
                "speaker": "agent",
                "message": agent_msg,
                "memory_snapshot": dict(memory),
                "mutation_applied": False,
            }
        )

        if state.is_terminal:
            termination_reason = "terminal_state"
            break

        # ─── 2. User Turn (Simulated) ────────────────────────────────────
        mutation = test_case.mutation
        apply_mutation = bool(
            mutation
            and mutation.turn_index <= turn_index < (mutation.turn_index + 2)
            and mutation_turns_used < 2
        )
        if apply_mutation:
            mutation_turns_used += 1

        # Base user prompt
        user_prompt = user_prompt_builder(
            scenario=scenario,
            state=state,
            memory=memory,
            history=history,
            mutation=mutation if apply_mutation else None,
            persona=test_case.persona,
        )
        # Add target language instruction
        lang_instruction = get_language_instruction(test_case.language)
        user_prompt += f"\n\nLANGUAGE INSTRUCTION:\n{lang_instruction}"

        raw_user_msg = tester.generate(
            user_prompt,
            system=f"You are a simulated user speaking in {test_case.language.value} with a {test_case.persona.value} persona.",
        ).strip()

        # If persona engine rewrite is desired for non-English or style alignment
        if test_case.language != Language.ENGLISH:
            user_msg = persona_engine.apply(
                raw_user_msg,
                persona=test_case.persona,
                language=test_case.language,
            )
        else:
            user_msg = raw_user_msg

        # Extract slots from user message
        memory_before = dict(memory)
        required_for_state = list(state.required_slots)
        extracted = extract_slots_from_text(user_msg, scenario=scenario, memory=memory)
        memory.update(extracted)

        history.append(
            {
                "turn": turn_index,
                "state": current_state,
                "speaker": "user",
                "message": user_msg,
                "memory_snapshot": dict(memory),
                "mutation_applied": apply_mutation,
            }
        )

        # ─── 3. State Progression & Stagnation Detection ─────────────────
        next_state = resolve_next_state(
            current_state=current_state,
            memory=memory,
            scenario=scenario,
        )
        transitioned = next_state != current_state
        memory_changed = memory != memory_before
        current_state = next_state

        required_missing = [
            slot for slot in required_for_state
            if slot not in memory or not str(memory[slot]).strip()
        ]

        if not transitioned and not memory_changed and required_missing:
            no_progress_counter += 1
        else:
            no_progress_counter = 0

        if no_progress_counter >= NO_PROGRESS_LIMIT:
            termination_reason = "no_progress"
            break

        if is_stuck(history, window=STAGNATION_WINDOW):
            termination_reason = "stagnation"
            break

    result: Dict[str, Any] = {
        "history": history,
        "summary": {
            "conversation_id": f"conv_{test_case.test_id}",
            "scenario_id": scenario.scenario_id,
            "test_id": test_case.test_id,
            "language": test_case.language.value,
            "persona": test_case.persona.value,
            "final_state": current_state,
            "termination_reason": termination_reason,
            "turns": len(history),
            "memory_final": dict(memory),
        },
    }

    # Evaluate simulation
    evaluation_report = evaluate_simulation(
        result=result,
        scenario=scenario,
        test_case=test_case,
        use_llm_judge=use_llm_judge,
        tester_llm=tester,
    )
    result["summary"]["evaluation"] = evaluation_report.to_dict()
    result["evaluation_report"] = evaluation_report

    return result
