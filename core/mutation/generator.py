"""
core.mutation.generator
~~~~~~~~~~~~~~~~~~~~~~~

Context-aware mutation and test case generation engine.

Produces diverse adversarial and edge-case test scenarios by injecting
slot omissions, invalid inputs, state skips, intent switches, ambiguities,
and policy violations paired with personas and target languages.
"""

from __future__ import annotations

import random
from typing import List, Optional

from core.utils.models import (
    Language,
    Mutation,
    MutationType,
    Persona,
    Scenario,
    TestCase,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core API
# ─────────────────────────────────────────────────────────────────────────────

def generate_test_cases(
    scenario: Scenario,
    num_cases: int = 5,
    languages: Optional[List[Language]] = None,
    personas: Optional[List[Persona]] = None,
) -> List[TestCase]:
    """
    Generate test cases for a given scenario by combining mutations, personas,
    and target languages (English, Arabic, French).

    Parameters
    ----------
    scenario:
        The baseline :class:`~core.utils.models.Scenario`.
    num_cases:
        Number of test cases to synthesize.
    languages:
        List of target languages to sample from. Defaults to all supported languages.
    personas:
        List of user personas to sample from. Defaults to all personas.

    Returns
    -------
    List[TestCase]
        A collection of fully constructed test cases.
    """
    target_languages = languages or list(Language)
    target_personas = personas or list(Persona)

    all_mutations = _generate_mutations_for_scenario(scenario)
    test_cases: List[TestCase] = []

    for i in range(num_cases):
        selected_mutations = random.sample(
            all_mutations,
            k=min(len(all_mutations), random.randint(0, 2)),
        ) if all_mutations else []

        mutation = selected_mutations[0] if selected_mutations else None
        expected_outcome = _infer_expected_outcome(mutation)
        chosen_language = target_languages[i % len(target_languages)]
        chosen_persona = target_personas[i % len(target_personas)]

        test_case = TestCase(
            test_id=f"{scenario.scenario_id}_TC_{i+1:02d}",
            scenario_id=scenario.scenario_id,
            mutation=mutation,
            language=chosen_language,
            persona=chosen_persona,
            description=_build_test_description(scenario, mutation, chosen_language, chosen_persona),
            expected_outcome=expected_outcome,
            expected_slots_filled=[s.name for s in scenario.slots if s.required],
            injected_errors=[mutation.type.value] if mutation else [],
        )

        test_cases.append(test_case)

    return test_cases


# ─────────────────────────────────────────────────────────────────────────────
# Mutation generation (context-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_mutations_for_scenario(scenario: Scenario) -> List[Mutation]:
    """Generate all candidate mutations applicable to the given scenario."""
    mutations: List[Mutation] = []

    # 1. Slot mutations (linked to states and required slots)
    for state in scenario.states:
        for slot in state.required_slots:
            mutations.append(_slot_missing_mutation(slot, state.state_id))
            mutations.append(_slot_invalid_mutation(slot, state.state_id))

    # 2. State mutations (skip meaningful non-terminal states)
    for state in scenario.states:
        if not state.is_terminal and state.required_slots:
            mutations.append(_state_skip_mutation(state.state_id))

    # 3. Intent mutations (derived from actual turns)
    for idx, turn in enumerate(scenario.turns):
        if turn.role == "user" and turn.expected_intent:
            mutations.append(_intent_switch_mutation(turn.expected_intent, idx))
            mutations.append(_ambiguity_mutation(turn.expected_intent, idx))

    # 4. Policy violations
    for state in scenario.states:
        for policy in state.policy_checks:
            mutations.append(_policy_violation_mutation(policy, state.state_id))

    return mutations


# ─────────────────────────────────────────────────────────────────────────────
# Expected outcome logic
# ─────────────────────────────────────────────────────────────────────────────

def _infer_expected_outcome(mutation: Optional[Mutation]) -> str:
    """Infer whether the test case is expected to succeed, recover, or fail."""
    if not mutation:
        return "success"

    if mutation.type == MutationType.POLICY_VIOLATION:
        return "failure"

    if mutation.type in {
        MutationType.SLOT_MISSING,
        MutationType.SLOT_INVALID,
        MutationType.STATE_SKIP,
        MutationType.STATE_REPEAT,
        MutationType.AMBIGUITY,
        MutationType.INTENT_SWITCH,
    }:
        return "recovery"

    return "success"


# ─────────────────────────────────────────────────────────────────────────────
# Mutation builders
# ─────────────────────────────────────────────────────────────────────────────

def _slot_missing_mutation(slot: str, state: str) -> Mutation:
    return Mutation(
        type=MutationType.SLOT_MISSING,
        target=slot,
        turn_index=1,
        description=f"User omits required slot '{slot}' in state '{state}'",
        expected_behavior="Agent must recognize missing information and explicitly request it",
    )


def _slot_invalid_mutation(slot: str, state: str) -> Mutation:
    return Mutation(
        type=MutationType.SLOT_INVALID,
        target=slot,
        turn_index=1,
        description=f"User provides an invalid value format for slot '{slot}' in state '{state}'",
        expected_behavior="Agent must validate input, reject invalid format, and ask for correction",
    )


def _state_skip_mutation(state_id: str) -> Mutation:
    return Mutation(
        type=MutationType.STATE_SKIP,
        target=state_id,
        turn_index=2,
        description=f"User attempts to skip prerequisite state '{state_id}'",
        expected_behavior="Agent must enforce prerequisites and prevent illegal state skip",
    )


def _intent_switch_mutation(intent: str, turn_index: int) -> Mutation:
    return Mutation(
        type=MutationType.INTENT_SWITCH,
        target=intent,
        turn_index=turn_index,
        description=f"User suddenly switches intent from '{intent}' mid-conversation",
        expected_behavior="Agent should acknowledge intent shift or safely guide user back",
    )


def _ambiguity_mutation(intent: str, turn_index: int) -> Mutation:
    return Mutation(
        type=MutationType.AMBIGUITY,
        target=intent,
        turn_index=turn_index,
        description=f"User gives vague or underspecified statement for intent '{intent}'",
        expected_behavior="Agent must disambiguate and request clarification before executing",
    )


def _policy_violation_mutation(policy: str, state: str) -> Mutation:
    return Mutation(
        type=MutationType.POLICY_VIOLATION,
        target=policy,
        turn_index=2,
        description=f"Security/Business policy '{policy}' violated during state '{state}'",
        expected_behavior="Agent must halt action, explain refusal, and offer safe alternative",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_test_description(
    scenario: Scenario,
    mutation: Optional[Mutation],
    language: Language,
    persona: Persona,
) -> str:
    lang_name = language.value.capitalize()
    persona_name = persona.value.capitalize()
    if not mutation:
        return (
            f"Nominal test for '{scenario.title}' "
            f"in {lang_name} with {persona_name} persona."
        )

    return (
        f"Mutation test for '{scenario.title}' in {lang_name} ({persona_name}): "
        f"{mutation.type.value} on '{mutation.target}'."
    )
