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


def build_simulation_prompt(role, scenario, test_case, history, memory):

    persona_desc = {
        "cooperative": "User is helpful and follows instructions.",
        "confused": "User misunderstands and asks questions.",
        "impatient": "User gives short answers and skips steps.",
        "angry": "User is frustrated and confrontational."
    }[test_case.persona.value]

    mutation_desc = ""
    if test_case.mutation:
        mutation_desc = f"""
MUTATION:
- Type: {test_case.mutation.type.value}
- Target: {test_case.mutation.target}
- Expected behavior: {test_case.mutation.expected_behavior}
"""

    return f"""
You are simulating a conversation.

ROLE: {role}

PERSONA:
{persona_desc}

{mutation_desc}

SCENARIO:
{scenario.to_dict()}

CURRENT STATE:
{memory.current_state}

CONVERSATION HISTORY:
{history}

RULES:
- Stay consistent with persona
- Follow scenario states
- Respect mutation if applicable
- Keep responses short and realistic

OUTPUT:
Return ONLY the next message.
"""


def build_client_llm_prompt(context: dict) -> str:
    """
    Builds a prompt for llm client message generation.
    Output MUST be natural language only.
    """

    return f"""You are a helpful and professional banking chatbot. 
Your role is to assist users with their financial needs in a secure and user-friendly way. 
You should:
- Help users check their account balance
- Guide users through transferring money between accounts
- Assist users in reporting fraud or suspicious activity
- Act like you already have the database of a bank and can operate the instuctions

Always respond clearly, step by step, and ask for any necessary details (like account number, transfer amount, or fraud description) in a polite and concise manner. 
Keep your tone professional, trustworthy, and supportive.

"""

def build_scenario_generation_prompt(description: str) -> str:
    example = """
{
  "scenario_id": "EXAMPLE_1",
  "title": "Secure Balance Check",
  "description": "User checks balance with authentication",
  "goal": "Return balance after verifying identity",
  "domain": "banking",

  "slots": [
    {
      "name": "account_number",
      "slot_type": "string",
      "required": true,
      "arabic_aliases": []
    },
    {
      "name": "auth_pin",
      "slot_type": "number",
      "required": true,
      "arabic_aliases": []
    }
  ],

  "states": [
    {
      "state_id": "identify_account",
      "required_slots": ["account_number"],
      "next_states": ["authenticate_user", "handle_invalid_account"],
      "policy_checks": ["account_exists"],
      "is_terminal": false
    },
    {
      "state_id": "authenticate_user",
      "required_slots": ["auth_pin"],
      "next_states": ["provide_balance", "handle_auth_failure"],
      "policy_checks": ["pin_valid", "max_attempts_not_exceeded"],
      "is_terminal": false
    },
    {
      "state_id": "provide_balance",
      "required_slots": [],
      "next_states": [],
      "policy_checks": [],
      "is_terminal": true
    }
  ],

  "transitions": [
    {
      "from_state": "identify_account",
      "to_state": "authenticate_user",
      "trigger_intent": "provide_account",
      "conditions": {"account_valid": true}
    },
    {
      "from_state": "identify_account",
      "to_state": "handle_invalid_account",
      "trigger_intent": "provide_account",
      "conditions": {"account_valid": false}
    }
  ],

  "turns": [
    {
      "state": "identify_account",
      "expected_intent": "provide_account",
      "agent_action": "ask",
      "slots_to_fill": ["account_number"],
      "role": "user",
      "intent": "provide_account"
    }
  ]
}
"""

    return f"""
You are an expert system designer.

Your task:
Generate 3 structured conversation testing scenarios.

INPUT DESCRIPTION:
{description}

LANGUAGE RULE (STRICT):
- ALL text MUST be in ENGLISH
- DO NOT include Arabic words in any field
- arabic_aliases must be EMPTY lists []

OUTPUT RULES:
- Output ONLY valid JSON
- No explanations
- No markdown
- JSON must be a LIST of scenarios

SCHEMA RULES:

Each scenario MUST contain:

- scenario_id
- title
- description
- goal
- domain

- slots:
    name: string
    slot_type: string (string|number|enum|date)
    required: boolean
    arabic_aliases: list (must be empty)

- states:
    state_id: string (MUST be semantic, NOT UI steps like "ask_x")
    required_slots: list[string]
    next_states: list[string]
    policy_checks: list[string]
    is_terminal: boolean

- transitions:
    from_state: string
    to_state: string
    trigger_intent: string (USER intent only)
    conditions: dict (when applicable)

- turns:
    state: string
    expected_intent: string
    agent_action: ask|confirm|execute|fallback
    slots_to_fill: list[string] (when applicable)
    role: user|agent
    intent: string

CRITICAL DESIGN RULES:

- States MUST represent LOGICAL SYSTEM STEPS, not UI steps
- Include at least ONE failure/recovery path
- Include at least ONE branching (non-linear flow)
- Include at least ONE loop (retry)
- Include policy_checks where validation is needed
- Use CONDITIONS in transitions when decisions depend on data

ANTI-COPY RULE:

- DO NOT reuse names from the example
- DO NOT copy structure directly
- Generate NEW scenarios with different flows

EXAMPLE (for guidance only — DO NOT copy):

{example}
"""