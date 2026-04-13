from dataclasses import dataclass, field
from typing import List, Set

from ai_engine.utils.models import Scenario


@dataclass
class ValidationResult:
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def fail(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def validate_scenario(scenario: Scenario) -> ValidationResult:
    result = ValidationResult()

    state_ids: Set[str] = {s.state_id for s in scenario.states}
    slot_names: Set[str] = {s.name for s in scenario.slots}

    # initial state exists
    if not scenario.states:
        result.fail("Scenario has no states")

    # check transitions
    for t in scenario.transitions:
        if t.from_state not in state_ids:
            result.fail(f"Transition from unknown state {t.from_state}")
        if t.to_state not in state_ids:
            result.fail(f"Transition to unknown state {t.to_state}")

    # check slots
    for state in scenario.states:
        for slot in state.required_slots:
            if slot not in slot_names:
                result.fail(f"State {state.state_id} references unknown slot {slot}")

    # reachability
    reachable = _bfs(scenario)
    for s in state_ids:
        if s not in reachable:
            result.warn(f"Unreachable state {s}")

    # dead states (dead state : a state with no outgoing transition)
    for s in scenario.states:
        if not s.is_terminal:
            has_outgoing = any(t.from_state == s.state_id for t in scenario.transitions)
            if not has_outgoing:
                result.fail(f"Dead state {s.state_id}")

    return result


def _bfs(scenario: Scenario) -> Set[str]:
    visited = set()
    queue = [scenario.states[0].state_id]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        for t in scenario.transitions:
            if t.from_state == current:
                queue.append(t.to_state)

    return visited