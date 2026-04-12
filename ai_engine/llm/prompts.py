def build_user_prompt(context: dict) -> str:
    """
    Builds a strict prompt for user message generation.
    Output MUST be natural language only.
    """

    return f"""
You are simulating a USER in a conversation.

CONTEXT:
- Intent: {context.get("intent")}
- State: {context.get("state")}
- Persona: {context.get("persona")}
- Dialect: {context.get("dialect")}
- Filled slots: {context.get("filled_slots")}
- Mutation: {context.get("mutation")}

INSTRUCTIONS:
- Speak like a real human
- Do NOT explain anything
- Do NOT output JSON
- Keep it short and natural
- Follow persona behavior strictly

Generate ONLY the user message.
"""

def build_scenario_generation_prompt(description: str) -> str:
    return f"""
You are an expert system designer.

Your task:
Generate 2–3 structured conversation testing scenarios.

INPUT DESCRIPTION:
{description}

LANGUAGE RULE (STRICT):
- ALL text MUST be in ENGLISH
- DO NOT use Arabic
- DO NOT include Arabic words in any field
- arabic_aliases must be EMPTY lists []

OUTPUT RULES:
- Output ONLY valid JSON
- No explanations
- JSON must be a LIST of scenarios

SCHEMA:

Each scenario must contain:

- scenario_id
- title
- description
- goal
- domain

- slots: [
    {{
      "name": "...",
      "slot_type": "string|number|enum",
      "required": true,
      "arabic_aliases": []
    }}
]

- states: [
    {{
      "state_id": "...",
      "required_slots": [...],
      "next_states": [...],
      "is_terminal": false
    }}
]

- transitions: [
    {{
      "from_state": "...",
      "to_state": "...",
      "trigger_intent": "..."
    }}
]

- turns: [
    {{
      "state": "...",
      "expected_intent": "...",
      "agent_action": "ask|confirm|execute|fallback",
      "role": "user",
      "intent": "..."
    }}
]

IMPORTANT:
- States must form a valid flow
- Include terminal state
- Include at least 3 states
- Slots must match states
- expected_intent MUST NOT be empty
"""