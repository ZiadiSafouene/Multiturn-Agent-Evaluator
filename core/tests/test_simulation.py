
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
from ai_engine.simulation.simulator import run_simulation

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

# ─────────────────────────────────────────────
# 5. Print conversation
# ─────────────────────────────────────────────
def print_simulation_log(result):
    print("\n=== SIMULATION LOG ===\n")
    history = result["history"]

    for step in history:
        print(f"[Turn {step['turn']}] ({step['state']}) {step['speaker'].upper()}:")
        print(step["message"])
        print(f"Memory Snapshot: {step.get('memory_snapshot', {})}")
        print(f"Mutation Applied: {step.get('mutation_applied', False)}")
        print("-" * 50)

    print("\n=== FINAL REPORT ===\n")
    print(json.dumps(result["summary"], indent=2))

result = run_simulation(
    scenario,
    test_case,
    llm,
    build_agent_prompt,
    build_user_prompt,
    max_turns=7,
)

print_simulation_log(result)