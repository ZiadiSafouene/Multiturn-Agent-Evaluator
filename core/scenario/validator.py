"""
core.scenario.validator
~~~~~~~~~~~~~~~~~~~~~~~

Validates that a Scenario forms a sound deterministic state graph without
orphan states, dangling transitions, or dead ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from core.utils.models import Scenario


@dataclass
class ValidationResult:
    """Outcome of validating a scenario graph."""

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_scenario(scenario: Scenario) -> ValidationResult:
    """
    Validate scenario topology and slot references.

    Checks:
    - Non-empty states list
    - Transitions point to declared states
    - State required slots exist in declared slots
    - All non-terminal states have at least one outgoing transition
    - Reachability of states from start node
    """
    result = ValidationResult()

    if not scenario.states:
        result.fail("Scenario has no states defined.")
        return result

    state_ids: Set[str] = {s.state_id for s in scenario.states}
    slot_names: Set[str] = {s.name for s in scenario.slots}

    # Transitions validity
    for t in scenario.transitions:
        if t.from_state not in state_ids:
            result.fail(f"Transition references unknown source state: '{t.from_state}'")
        if t.to_state not in state_ids:
            result.fail(f"Transition references unknown destination state: '{t.to_state}'")

    # Slot references
    for state in scenario.states:
        for slot in state.required_slots:
            if slot not in slot_names:
                result.fail(f"State '{state.state_id}' requires undeclared slot: '{slot}'")

    # Reachability check
    reachable = _bfs(scenario)
    for s in state_ids:
        if s not in reachable:
            result.warn(f"State '{s}' is unreachable from initial state '{scenario.states[0].state_id}'")

    # Dead state detection
    for s in scenario.states:
        if not s.is_terminal:
            has_outgoing = any(t.from_state == s.state_id for t in scenario.transitions)
            if not has_outgoing and not s.next_states:
                result.fail(f"Non-terminal state '{s.state_id}' has no outgoing transitions (dead state).")

    return result


def _bfs(scenario: Scenario) -> Set[str]:
    """Traverse state graph starting from initial state."""
    visited: Set[str] = set()
    queue = [scenario.states[0].state_id]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        for t in scenario.transitions:
            if t.from_state == current and t.to_state not in visited:
                queue.append(t.to_state)

    return visited
