"""
core.scenario.state_machine
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic execution and state transition engine for conversation scenarios.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.utils.models import Scenario, State, StateTransition


class ScenarioStateMachine:
    """
    Manages deterministic state tracking, slot prerequisites, and condition evaluation.

    Parameters
    ----------
    scenario:
        The :class:`~core.utils.models.Scenario` graph being executed.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.state_map: Dict[str, State] = {s.state_id: s for s in scenario.states}
        self.current_state_id: str = (
            scenario.states[0].state_id if scenario.states else "start"
        )
        self.visited_states: List[str] = [self.current_state_id]

    @property
    def current_state(self) -> Optional[State]:
        return self.state_map.get(self.current_state_id)

    @property
    def is_terminal(self) -> bool:
        curr = self.current_state
        return curr.is_terminal if curr else True

    def get_missing_slots(self, memory: Dict[str, Any]) -> List[str]:
        """Return required slots for current state that are not yet filled."""
        curr = self.current_state
        if not curr:
            return []
        return [
            slot for slot in curr.required_slots
            if slot not in memory or memory[slot] is None or str(memory[slot]).strip() == ""
        ]

    def can_transition(self, memory: Dict[str, Any]) -> bool:
        """Check if all required slots for current state are populated."""
        return len(self.get_missing_slots(memory)) == 0

    def step(self, memory: Dict[str, Any], trigger_intent: Optional[str] = None) -> str:
        """
        Attempt to step the state machine forward based on memory and transition conditions.

        Returns
        -------
        str
            The next state ID (which may be the same if prerequisites are unfulfilled).
        """
        curr = self.current_state
        if not curr or curr.is_terminal:
            return self.current_state_id

        # If required slots are missing, cannot leave state
        if not self.can_transition(memory):
            return self.current_state_id

        # Search matching transition
        transitions = [t for t in self.scenario.transitions if t.from_state == self.current_state_id]
        for t in transitions:
            if trigger_intent and t.trigger_intent and t.trigger_intent.lower() != trigger_intent.lower():
                continue
            if self._matches_conditions(t.conditions, memory):
                self._move_to(t.to_state)
                return self.current_state_id

        # Fallback to first next_state if defined
        if curr.next_states:
            self._move_to(curr.next_states[0])
            return self.current_state_id

        return self.current_state_id

    def _move_to(self, new_state_id: str) -> None:
        if new_state_id in self.state_map:
            self.current_state_id = new_state_id
            self.visited_states.append(new_state_id)

    @staticmethod
    def _matches_conditions(conditions: Dict[str, Any], memory: Dict[str, Any]) -> bool:
        if not conditions:
            return True
        for key, expected in conditions.items():
            val = memory.get(key)
            if isinstance(expected, bool):
                if isinstance(val, str):
                    actual_bool = val.strip().lower() in {"true", "1", "yes", "oui", "نعم"}
                else:
                    actual_bool = bool(val)
                if actual_bool != expected:
                    return False
            elif isinstance(expected, list):
                if val not in expected:
                    return False
            else:
                if str(val).strip().lower() != str(expected).strip().lower():
                    return False
        return True
