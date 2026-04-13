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

for s in scenarios:
    print(s)

print("\nERRORS:")
for e in errors:
    print(e)