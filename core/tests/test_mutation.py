"""Unit tests for mutation synthesis and test case generation."""

import pytest
from core.mutation.generator import generate_test_cases
from core.scenario.storage import load_scenarios_from_json
from core.utils.models import Language, Persona, MutationType


@pytest.fixture
def banking_scenario():
    scenarios = load_scenarios_from_json("test_scenarios.json")
    return scenarios[0]


def test_generate_test_cases_defaults(banking_scenario):
    cases = generate_test_cases(banking_scenario, num_cases=6)
    assert len(cases) == 6

    # Verify that the 3 supported languages (English, Arabic, French) are used
    languages_seen = {c.language for c in cases}
    assert Language.ENGLISH in languages_seen
    assert Language.ARABIC in languages_seen
    assert Language.FRENCH in languages_seen

    # Verify test case structure
    for c in cases:
        assert c.scenario_id == banking_scenario.scenario_id
        assert isinstance(c.persona, Persona)
        assert isinstance(c.language, Language)
        assert c.expected_outcome in {"success", "recovery", "failure"}


def test_generate_test_cases_custom_languages(banking_scenario):
    cases = generate_test_cases(
        banking_scenario,
        num_cases=4,
        languages=[Language.FRENCH],
        personas=[Persona.ANGRY],
    )
    assert len(cases) == 4
    for c in cases:
        assert c.language == Language.FRENCH
        assert c.persona == Persona.ANGRY


def test_mutation_types_represented(banking_scenario):
    cases = generate_test_cases(banking_scenario, num_cases=12)
    mutations = [c.mutation for c in cases if c.mutation is not None]
    assert len(mutations) > 0
    mutation_types = {m.type for m in mutations}
    assert any(isinstance(mt, MutationType) for mt in mutation_types)
