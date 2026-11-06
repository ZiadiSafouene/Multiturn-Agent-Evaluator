"""
mutation/generator.py — production-grade mutation engine
"""

from __future__ import annotations
import random
from typing import List, Optional

from ai_engine.utils.models import (
    Scenario,
    Mutation,
    MutationType,
    TestCase,
    Dialect,
    Persona,
)

# ─────────────────────────────────────────────────────────────
# Core API
# ─────────────────────────────────────────────────────────────

def generate_test_cases(
    scenario: Scenario,
    num_cases: int = 5,
    dialects: Optional[List[Dialect]] = None,
    personas: Optional[List[Persona]] = None,
) -> List[TestCase]:

    dialects = dialects or list(Dialect)
    personas = personas or list(Persona)

    all_mutations = _generate_mutations_for_scenario(scenario)
    test_cases: List[TestCase] = []

    for i in range(num_cases):
        # Select 0–2 mutations (realistic)
        selected_mutations = random.sample(
            all_mutations,
            k=min(len(all_mutations), random.randint(0, 2))
        )

        mutation = selected_mutations[0] if selected_mutations else None

        expected_outcome = _infer_expected_outcome(mutation)

        test_case = TestCase(
            test_id=f"{scenario.scenario_id}_TC_{i}",
            scenario_id=scenario.scenario_id,
            mutation=mutation,
            dialect=random.choice(dialects),
            persona=random.choice(personas),
            description=_build_test_description(scenario, mutation),
            expected_outcome=expected_outcome,
            expected_slots_filled=[s.name for s in scenario.slots if s.required],
            injected_errors=[mutation.type.value] if mutation else [],
        )

        test_cases.append(test_case)

    return test_cases


# ─────────────────────────────────────────────────────────────
# Mutation generation (context-aware)
# ─────────────────────────────────────────────────────────────

def _generate_mutations_for_scenario(scenario: Scenario) -> List[Mutation]:
    mutations: List[Mutation] = []

    # ── Slot mutations (linked to states)
    for state in scenario.states:
        for slot in state.required_slots:
            mutations.append(_slot_missing_mutation(slot, state.state_id))
            mutations.append(_slot_invalid_mutation(slot, state.state_id))

    # ── State mutations (skip meaningful states only)
    for state in scenario.states:
        if not state.is_terminal and state.required_slots:
            mutations.append(_state_skip_mutation(state.state_id))

    # ── Intent mutations (from actual turns)
    for idx, turn in enumerate(scenario.turns):
        if turn.role == "user" and turn.expected_intent:
            mutations.append(_intent_switch_mutation(turn.expected_intent, idx))
            mutations.append(_ambiguity_mutation(turn.expected_intent, idx))

    # ── Policy violations (only if exist)
    for state in scenario.states:
        for policy in state.policy_checks:
            mutations.append(_policy_violation_mutation(policy, state.state_id))

    return mutations


# ─────────────────────────────────────────────────────────────
# Expected outcome logic (CRITICAL FIX)
# ─────────────────────────────────────────────────────────────

def _infer_expected_outcome(mutation: Optional[Mutation]) -> str:
    if not mutation:
        return "success"

    if mutation.type == MutationType.POLICY_VIOLATION:
        return "failure"

    if mutation.type in [
        MutationType.SLOT_MISSING,
        MutationType.SLOT_INVALID,
        MutationType.STATE_SKIP,
        MutationType.STATE_REPEAT,
        MutationType.AMBIGUITY,
        MutationType.INTENT_SWITCH,
    ]:
        return "recovery"

    return "success"


# ─────────────────────────────────────────────────────────────
# Mutation builders (IMPROVED)
# ─────────────────────────────────────────────────────────────

def _slot_missing_mutation(slot: str, state: str) -> Mutation:
    return Mutation(
        type=MutationType.SLOT_MISSING,
        target=slot,
        turn_index=1,
        description=f"User does not provide required slot '{slot}' in state '{state}'",
        expected_behavior="Agent must explicitly request the missing information",
    )


def _slot_invalid_mutation(slot: str, state: str) -> Mutation:
    return Mutation(
        type=MutationType.SLOT_INVALID,
        target=slot,
        turn_index=1,
        description=f"User provides invalid value for '{slot}' in state '{state}'",
        expected_behavior="Agent must detect invalid input and ask for correction",
    )


def _state_skip_mutation(state_id: str) -> Mutation:
    return Mutation(
        type=MutationType.STATE_SKIP,
        target=state_id,
        turn_index=2,
        description=f"User attempts to bypass state '{state_id}'",
        expected_behavior="Agent must enforce required step before proceeding",
    )


def _intent_switch_mutation(intent: str, turn_index: int) -> Mutation:
    return Mutation(
        type=MutationType.INTENT_SWITCH,
        target=intent,
        turn_index=turn_index,
        description=f"User switches intent unexpectedly from '{intent}'",
        expected_behavior="Agent should either redirect or handle new intent safely",
    )


def _ambiguity_mutation(intent: str, turn_index: int) -> Mutation:
    return Mutation(
        type=MutationType.AMBIGUITY,
        target=intent,
        turn_index=turn_index,
        description=f"User provides vague input instead of clear intent '{intent}'",
        expected_behavior="Agent should ask clarifying question",
    )


def _policy_violation_mutation(policy: str, state: str) -> Mutation:
    return Mutation(
        type=MutationType.POLICY_VIOLATION,
        target=policy,
        turn_index=2,
        description=f"Policy '{policy}' fails during state '{state}'",
        expected_behavior="Agent must block action and explain reason",
    )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _build_test_description(scenario: Scenario, mutation: Optional[Mutation]) -> str:
    if not mutation:
        return f"Happy path test for scenario '{scenario.title}'"

    return (
        f"Test scenario '{scenario.title}' with mutation: "
        f"{mutation.type.value} on '{mutation.target}'"
    )