import sys
from pathlib import Path

# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

total_test_cases = []

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

    total_test_cases.extend(temp_test_cases)

print("\n Test Cases after mutation:")
print("\n Nb Mutations : ",len(total_test_cases))

for i in total_test_cases: 
    print("\n",i)
