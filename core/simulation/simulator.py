from __future__ import annotations

import re
from typing import Any

from ai_engine.evaluation.engine import evaluate_simulation


MAX_TURNS = 7
STAGNATION_WINDOW = 3
NO_PROGRESS_LIMIT = 3


def _normalize_slot_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def extract_slots_from_text(text: str, expected_slots: list[str], memory: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    expected = {_normalize_slot_name(slot): slot for slot in expected_slots}

    # Generic key/value capture: "slot_name: value" or "slot_name = value"
    for key, value in re.findall(r"([a-zA-Z_][a-zA-Z0-9_ ]{1,40})\s*[:=]\s*([^\n,;]+)", text):
        normalized_key = _normalize_slot_name(key)
        if normalized_key in expected:
            extracted[expected[normalized_key]] = value.strip()

    # Generic direct mention: "<slot> is <value>" / "<slot> <value>"
    lowered_text = text.lower()
    for normalized_key, original_key in expected.items():
        if original_key in extracted or original_key in memory:
            continue
        pattern = rf"\b{re.escape(normalized_key).replace('_', r'[_\s]+')}\b(?:\s+is|\s*=|\s*:)?\s+([^\n,;]+)"
        match = re.search(pattern, lowered_text)
        if match:
            extracted[original_key] = match.group(1).strip()

    # If exactly one required slot is missing, use full answer as fallback value.
    missing = [slot for slot in expected_slots if slot not in memory and slot not in extracted]
    if len(missing) == 1 and text.strip():
        extracted[missing[0]] = text.strip()

    return extracted


def _value_matches_condition(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return value in expected
    if isinstance(expected, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"} if expected else value.strip().lower() in {"false", "0", "no"}
        return bool(value) is expected
    return str(value).strip().lower() == str(expected).strip().lower()


def resolve_next_state(current_state: str, memory: dict[str, Any], scenario) -> str:
    state_map = {state.state_id: state for state in scenario.states}
    state = state_map[current_state]

    # Transition gate: required slots must be filled first.
    if not all(slot in memory and str(memory[slot]).strip() for slot in state.required_slots):
        return current_state

    transitions = [t for t in scenario.transitions if t.from_state == current_state]
    for transition in transitions:
        conditions = transition.conditions or {}
        if all(_value_matches_condition(memory.get(key), expected) for key, expected in conditions.items()):
            return transition.to_state

    # Deterministic fallback if transitions have no matching conditions.
    if state.next_states:
        return state.next_states[0]

    return current_state


def is_stuck(history: list[dict], window: int = STAGNATION_WINDOW) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    states = [entry["state"] for entry in recent]
    same_state = len(set(states)) == 1
    memory_snapshots = [entry.get("memory_snapshot", {}) for entry in recent]
    no_memory_change = all(snapshot == memory_snapshots[0] for snapshot in memory_snapshots[1:])
    return same_state or no_memory_change


def run_simulation(
    scenario,
    test_case,
    llm_client,
    build_agent_prompt,
    build_user_prompt,
    max_turns: int = MAX_TURNS,
):
    state_map = {s.state_id: s for s in scenario.states}
    all_slots = [slot.name for slot in scenario.slots]

    current_state = scenario.states[0].state_id
    memory: dict[str, Any] = dict(test_case.initial_slots or {})
    history: list[dict] = []
    termination_reason = "max_turns"

    no_progress_counter = 0
    mutation_turns_used = 0

    for turn_index in range(max_turns):
        state = state_map[current_state]

        # Agent turn (text only; no control logic).
        agent_msg = llm_client.generate(
            build_agent_prompt(
                scenario=scenario,
                state=state,
                memory=memory,
                history=history,
            )
        ).strip()
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

        mutation = test_case.mutation
        apply_mutation = bool(
            mutation
            and mutation.turn_index <= turn_index < (mutation.turn_index + 2)
            and mutation_turns_used < 2
        )
        if apply_mutation:
            mutation_turns_used += 1

        user_msg = llm_client.generate(
            build_user_prompt(
                scenario=scenario,
                state=state,
                memory=memory,
                history=history,
                mutation=mutation if apply_mutation else None,
                persona=test_case.persona,
            )
        ).strip()

        memory_before = dict(memory)
        required_for_state = list(state.required_slots)
        extracted = extract_slots_from_text(user_msg, expected_slots=all_slots, memory=memory)
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

        next_state = resolve_next_state(current_state=current_state, memory=memory, scenario=scenario)
        transitioned = next_state != current_state
        memory_changed = memory != memory_before
        current_state = next_state

        required_missing = [slot for slot in required_for_state if slot not in memory or not str(memory[slot]).strip()]
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

    result = {
        "history": history,
        "summary": {
            "final_state": current_state,
            "termination_reason": termination_reason,
            "turns": len(history),
            "memory_final": dict(memory),
        },
    }
    result["summary"]["evaluation"] = evaluate_simulation(result, scenario, test_case)
    return result