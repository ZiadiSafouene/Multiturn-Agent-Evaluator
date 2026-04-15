
import sys
from pathlib import Path

# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_engine.llm.client import LLMClient
from ai_engine.mutation.generator import generate_test_cases
from ai_engine.simulation.simulator import simulate_conversation
from ai_engine.scenario.generator import generate_scenarios_from_description

description = """
A banking chatbot that helps users:
- check balance
- transfer money
- report fraud
"""

scenarios, errors = generate_scenarios_from_description(description)

print("VALID:", len(scenarios))
print("ERRORS:", errors)

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

conversation = simulate_conversation(
    scenario=scenario,
    test_case=test_case,
    llm_client=llm,
    max_turns=10
)

# ─────────────────────────────────────────────
# 5. Print conversation
# ─────────────────────────────────────────────

print("\n===== CONVERSATION =====\n")

for turn in conversation.turns:
    role = "USER" if turn.role == "user" else "AGENT"
    print(f"{role}: {turn.message}")

print("\n===== TEST CASE =====")
print(test_case)

print("\n===== FINAL STATE =====")
print(conversation.memory.current_state)
print("Status:", conversation.status)