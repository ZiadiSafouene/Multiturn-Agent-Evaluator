
from __future__ import annotations
import sys
from pathlib import Path
import json



# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_engine.llm.prompts import build_agent_prompt, build_user_prompt
from ai_engine.llm.client import LLMClient
from ai_engine.mutation.generator import generate_test_cases
from ai_engine.scenario.generator import generate_scenarios_from_description

description = """
A banking chatbot that helps users:
- check balance
- transfer money
- report fraud
"""

# scenarios, errors = generate_scenarios_from_description(description)

# from ai_engine.scenario.storage import save_scenarios_to_json

# save_scenarios_to_json(
#     scenarios=scenarios,
#     path="./test_scenarios.json"
# )
from ai_engine.scenario.storage import load_scenarios_from_json
from ai_engine.scenario.parser import parse_scenario

raw = load_scenarios_from_json("test_scenarios.json")

scenarios = [parse_scenario(s) for s in raw]

print("VALID:", len(scenarios))
# print("ERRORS:", errors)

# for s in scenarios:
#     print(s)

# print("\nERRORS:")
# for e in errors:
#     print(e)

# ------------- Mutations ---------------

from ai_engine.mutation.generator import generate_test_cases
from ai_engine.utils.models import Dialect, Persona

test_cases = []

for s in scenarios:
    temp_test_cases = generate_test_cases(
        scenario=s,
        num_cases=3,
        dialects=[Dialect.MSA, Dialect.MIXED],  
        personas=[
        Persona.COOPERATIVE,
        Persona.CONFUSED,
        Persona.IMPATIENT,
        Persona.ANGRY,
    ],  
    )

    test_cases.extend(temp_test_cases)

scenario = scenarios[0]
test_case = test_cases[0]

# print("\n Test Cases after mutation:")
# print("\n Nb Mutations : ",len(total_test_cases))

# for i in total_test_cases: 
#     print("\n",i)


# ─────────────────────────────────────────────
# 3. Init LLM
# ─────────────────────────────────────────────

llm = LLMClient()

# ─────────────────────────────────────────────
# 4. Run simulation
# ─────────────────────────────────────────────
from typing import Dict, List

from ai_engine.utils.models import Scenario, TestCase


MAX_TURNS = 5

def _resolve_next_state(scenario, current_state, memory):
    state_map = {s.state_id: s for s in scenario.states}
    state = state_map[current_state]

    # If all required slots are filled → move forward
    if all(slot in memory for slot in state.required_slots):
        for t in scenario.transitions:
            if t.from_state == current_state:
                return t.to_state

    return current_state

def run_simulation(
    scenario: Scenario,
    test_case: TestCase,
    llm_client,
    build_agent_prompt,
    build_user_prompt,
):
    """
    Runs controlled FSM-based simulation.

    - Scenario controls flow
    - LLM generates text ONLY
    - Mutations injected at specific turn
    """

    state_map = {s.state_id: s for s in scenario.states}
    current_state = scenario.states[0].state_id

    memory: Dict[str, str] = {}
    history: List[Dict] = []

    turn_index = 0

    while turn_index < MAX_TURNS:
        state = state_map[current_state]

        # ─────────────────────────────────────────────
        # AGENT TURN
        # ─────────────────────────────────────────────
        agent_prompt = build_agent_prompt(
            scenario=scenario,
            state=state,
            memory=memory,
            history=history,
        )

        agent_msg = llm_client.generate(agent_prompt)

        history.append({
            "turn": turn_index,
            "speaker": "agent",
            "state": current_state,
            "message": agent_msg,
        })

        # ─────────────────────────────────────────────
        # TERMINATION CHECK
        # ─────────────────────────────────────────────
        if state.is_terminal:
            break

        # ─────────────────────────────────────────────
        # USER TURN (WITH MUTATION)
        # ─────────────────────────────────────────────
        mutation = test_case.mutation

        inject_mutation = (
            mutation is not None and
            mutation.turn_index == turn_index
        )

        user_prompt = build_user_prompt(
            scenario=scenario,
            state=state,
            memory=memory,
            history=history,
            mutation=mutation if inject_mutation else None,
            persona=test_case.persona,
        )

        user_msg = llm_client.generate(user_prompt)

        history.append({
            "turn": turn_index,
            "speaker": "user",
            "state": current_state,
            "message": user_msg,
            "mutation": inject_mutation,
        })

        # ─────────────────────────────────────────────
        # STATE TRANSITION (DETERMINISTIC)
        # ─────────────────────────────────────────────
        next_state = _resolve_next_state(
            scenario,
            current_state,
            memory
        )

        if next_state == current_state:
            # prevent infinite loops
            turn_index += 1
            continue

        current_state = next_state
        turn_index += 1

    return history

# ─────────────────────────────────────────────
# 5. Print conversation
# ─────────────────────────────────────────────
def print_simulation_log(history):
    print("\n=== SIMULATION LOG ===\n")

    for step in history:
        print(f"[Turn {step['turn']}] ({step['state']}) {step['speaker'].upper()}:")
        print(step["message"])

        if step.get("mutation"):
            print("⚠️ Mutation injected")

        print("-" * 50)


history = run_simulation(
    scenario,
    test_case,
    llm,
    build_agent_prompt,
    build_user_prompt,
)

print_simulation_log(history)