from __future__ import annotations


def _all_valid_edges(history: list[dict], scenario) -> bool:
    valid_edges = {(t.from_state, t.to_state) for t in scenario.transitions}
    if not valid_edges:
        return True
    agent_states = [entry["state"] for entry in history if entry.get("speaker") == "agent"]
    if len(agent_states) < 2:
        return True
    for index in range(1, len(agent_states)):
        previous_state = agent_states[index - 1]
        current_state = agent_states[index]
        if previous_state != current_state and (previous_state, current_state) not in valid_edges:
            return False
    return True


def evaluate_simulation(result, scenario, test_case):
    history = result["history"]
    summary = result["summary"]

    initial_state = scenario.states[0].state_id if scenario.states else None
    final_state = summary.get("final_state")
    terminal_states = {state.state_id for state in scenario.states if state.is_terminal}

    state_progression = 1 if final_state != initial_state else 0
    completion = 1 if final_state in terminal_states else 0
    transition_validity = 1 if _all_valid_edges(history, scenario) else 0

    termination_reason = summary.get("termination_reason")
    safe_termination_reasons = {"terminal_state", "stagnation", "no_progress", "max_turns"}
    if test_case.mutation:
        recovered = completion == 1 or termination_reason in safe_termination_reasons
        recovery_score = 1 if recovered else 0
    else:
        recovery_score = 1

    stagnation = 0 if termination_reason == "stagnation" else 1

    metrics = {
        "state_progression": state_progression,
        "completion": completion,
        "transition_validity": transition_validity,
        "recovery_score": recovery_score,
        "stagnation": stagnation,
    }
    metrics["final_score"] = round(sum(metrics.values()) / len(metrics), 2)
    return metrics