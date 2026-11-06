"""
core.llm.prompts
~~~~~~~~~~~~~~~~

Prompt templates used throughout the pipeline.

All prompts are pure functions — they take structured inputs and return
a string ready to send to any LLM.
"""

from __future__ import annotations

from core.utils.models import Language


# ─────────────────────────────────────────────────────────────────────────────
# Language instruction helpers
# ─────────────────────────────────────────────────────────────────────────────

_LANGUAGE_INSTRUCTIONS: dict[Language, str] = {
    Language.ENGLISH: (
        "You MUST respond ONLY in English. "
        "Do NOT mix languages."
    ),
    Language.ARABIC: (
        "يجب أن تكتب ردودك باللغة العربية الفصحى فقط. "
        "لا تخلط بين اللغات."
    ),
    Language.FRENCH: (
        "Vous DEVEZ répondre UNIQUEMENT en français. "
        "Ne mélangez pas les langues."
    ),
}


def get_language_instruction(language: Language) -> str:
    """Return the language-enforcement instruction for *language*."""
    return _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS[Language.ENGLISH])


# ─────────────────────────────────────────────────────────────────────────────
# Simulation prompts
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_prompt(scenario, state, memory, history) -> str:
    """
    Build the system + context prompt for the **agent under test**.

    The agent is state-driven: it must stay in ``state`` and only ask for
    missing required slots before transitioning.

    Parameters
    ----------
    scenario:
        The active :class:`~core.utils.models.Scenario`.
    state:
        The current :class:`~core.utils.models.State`.
    memory:
        A ``dict`` of already-filled slots.
    history:
        A ``list[dict]`` of past turns (speaker + message).

    Returns
    -------
    str
        A prompt string ready to send to the agent LLM.
    """
    history_text = "\n".join(
        f"{entry['speaker'].capitalize()}: {entry['message']}"
        for entry in history
    )

    return f"""You are a customer-service agent.

CURRENT STATE: {state.state_id}
REQUIRED SLOTS: {state.required_slots}
COLLECTED MEMORY: {memory}

STRICT RULES:
- Ask ONLY for slots that are missing from COLLECTED MEMORY
- Do NOT skip states or hallucinate data
- Do NOT repeat the same message verbatim — vary phrasing on retries
- After repeated user failures, escalate politely and explain why the data is needed
- When all required slots are filled, confirm and transition
- Keep responses concise (1–3 sentences)

CONVERSATION SO FAR:
{history_text}

Generate ONLY the agent's next message."""


def build_user_prompt(scenario, state, memory, history, mutation, persona) -> str:
    """
    Build the prompt for the **simulated user**.

    Parameters
    ----------
    scenario:
        The active :class:`~core.utils.models.Scenario`.
    state:
        The current :class:`~core.utils.models.State`.
    memory:
        A ``dict`` of already-filled slots.
    history:
        A ``list[dict]`` of past turns.
    mutation:
        An optional :class:`~core.utils.models.Mutation` to apply.
    persona:
        A :class:`~core.utils.models.Persona` enum value.

    Returns
    -------
    str
        A prompt string ready to send to the user-simulator LLM.
    """
    from core.utils.models import Persona  # local import to avoid circularity

    _PERSONA_DESC: dict[Persona, str] = {
        Persona.COOPERATIVE: "You are helpful and answer clearly.",
        Persona.CONFUSED: "You misunderstand questions and ask for clarification.",
        Persona.IMPATIENT: "You give short, rushed answers and try to skip steps.",
        Persona.ANGRY: "You are frustrated and slightly confrontational.",
    }

    history_text = "\n".join(
        f"{entry['speaker'].capitalize()}: {entry['message']}"
        for entry in history
    )

    base = f"""You are a USER interacting with a chatbot.

CURRENT STATE: {state.state_id}
REQUIRED SLOTS: {state.required_slots}
PERSONA: {_PERSONA_DESC.get(persona, 'cooperative')}
"""

    if mutation:
        base += f"""
ACTIVE MUTATION (apply for 1–2 turns then resume normal behaviour):
  Type     : {mutation.type.value}
  Target   : {mutation.target}
  Behaviour: {mutation.description}
  Expected : {mutation.expected_behavior}
"""

    base += f"""
RULES:
- Respond as the USER only — do NOT impersonate the agent
- Gradually provide missing required slots unless a mutation is active
- Keep responses short and realistic

CONVERSATION SO FAR:
{history_text}

Generate ONLY the user's next message."""

    return base


def build_simulation_prompt(role, scenario, test_case, history, memory) -> str:
    """
    Build a generic simulation prompt for a given *role*.

    Parameters
    ----------
    role:
        ``"user"`` or ``"agent"``.
    scenario:
        The active :class:`~core.utils.models.Scenario`.
    test_case:
        The active :class:`~core.utils.models.TestCase`.
    history:
        Plain-text conversation transcript.
    memory:
        A :class:`~core.utils.models.ConversationMemory` instance.

    Returns
    -------
    str
        A prompt string ready to send to the simulator LLM.
    """
    from core.utils.models import Persona  # local import to avoid circularity

    _PERSONA_DESC: dict[Persona, str] = {
        Persona.COOPERATIVE: "Helpful and follows instructions.",
        Persona.CONFUSED: "Misunderstands and asks questions.",
        Persona.IMPATIENT: "Gives short answers and skips steps.",
        Persona.ANGRY: "Frustrated and confrontational.",
    }
    persona_desc = _PERSONA_DESC.get(test_case.persona, "cooperative")

    mutation_desc = ""
    if test_case.mutation:
        mutation_desc = (
            f"MUTATION:\n"
            f"  Type    : {test_case.mutation.type.value}\n"
            f"  Target  : {test_case.mutation.target}\n"
            f"  Expected: {test_case.mutation.expected_behavior}\n"
        )

    return f"""You are simulating a conversation.

ROLE: {role}
PERSONA: {persona_desc}
{mutation_desc}
SCENARIO: {scenario.to_dict()}
CURRENT STATE: {memory.current_state}

CONVERSATION:
{history}

RULES:
- Stay consistent with persona
- Follow scenario states
- Respect mutation if applicable
- Keep responses short and realistic

Return ONLY the next message."""


# ─────────────────────────────────────────────────────────────────────────────
# Scenario generation prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_scenario_generation_prompt(description: str) -> str:
    """
    Build the prompt that instructs the tester LLM to generate test scenarios.

    Parameters
    ----------
    description:
        A natural-language description of the agent to test (domain, goals,
        expected capabilities).

    Returns
    -------
    str
        A fully-formed prompt string.
    """
    example = """{
  "scenario_id": "EXAMPLE_1",
  "title": "Secure Balance Check",
  "description": "User checks balance with authentication",
  "goal": "Return balance after verifying identity",
  "domain": "banking",
  "slots": [
    {"name": "account_number", "slot_type": "string", "required": true, "aliases": []},
    {"name": "auth_pin",       "slot_type": "number", "required": true, "aliases": []}
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
      "policy_checks": ["pin_valid"],
      "is_terminal": false
    },
    {"state_id": "provide_balance", "required_slots": [], "next_states": [], "policy_checks": [], "is_terminal": true}
  ],
  "transitions": [
    {"from_state": "identify_account",  "to_state": "authenticate_user",    "trigger_intent": "provide_account", "conditions": {"account_valid": true}},
    {"from_state": "identify_account",  "to_state": "handle_invalid_account","trigger_intent": "provide_account", "conditions": {"account_valid": false}},
    {"from_state": "authenticate_user", "to_state": "provide_balance",       "trigger_intent": "provide_pin",     "conditions": {"pin_valid": true}}
  ],
  "turns": [
    {"state": "identify_account",  "expected_intent": "provide_account", "agent_action": "ask",     "slots_to_fill": ["account_number"], "role": "user", "intent": "provide_account"},
    {"state": "authenticate_user", "expected_intent": "provide_pin",     "agent_action": "ask",     "slots_to_fill": ["auth_pin"],        "role": "user", "intent": "provide_pin"},
    {"state": "provide_balance",   "expected_intent": "none",            "agent_action": "execute",  "slots_to_fill": [],                 "role": "agent","intent": "show_balance"}
  ]
}"""

    return f"""You are an expert conversational AI system designer.

TASK:
Generate exactly 3 structured conversation testing scenarios in JSON.

INPUT DESCRIPTION:
{description}

OUTPUT RULES:
- Output ONLY valid JSON — a LIST of scenario objects
- No explanations, no markdown, no comments
- ALL text fields must be in English

SCHEMA (each scenario must include):
  scenario_id      : unique string
  title            : string
  description      : string
  goal             : string
  domain           : string
  slots            : list of {{ name, slot_type (string|number|enum|date), required, aliases }}
  states           : list of {{ state_id, required_slots, next_states, policy_checks, is_terminal }}
  transitions      : list of {{ from_state, to_state, trigger_intent, conditions }}
  turns            : list of {{ state, expected_intent, agent_action (ask|confirm|execute|fallback), slots_to_fill, role (user|agent), intent }}

DESIGN RULES:
- States represent LOGICAL SYSTEM STEPS, not UI steps
- Include at least one failure/recovery path
- Include at least one branching (non-linear flow)
- Include at least one retry loop
- Use policy_checks where validation is needed
- Use conditions in transitions when decisions depend on data

DO NOT copy the example below — generate entirely new scenarios.

EXAMPLE (structure reference only):
{example}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Persona / language rewriting prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_persona_rewrite_prompt(text: str, persona, language: Language) -> str:
    """
    Build the prompt that rewrites *text* to match a persona and language.

    Parameters
    ----------
    text:
        The original user message to rewrite.
    persona:
        A :class:`~core.utils.models.Persona` enum value.
    language:
        A :class:`~core.utils.models.Language` enum value.

    Returns
    -------
    str
        A prompt string ready to send to the tester LLM.
    """
    from core.utils.models import Persona  # local import to avoid circularity

    _PERSONA_STYLE: dict[Persona, str] = {
        Persona.COOPERATIVE: "Clear, polite, and helpful tone.",
        Persona.CONFUSED: "Hesitant, unsure, asks for clarification.",
        Persona.IMPATIENT: "Short, direct, slightly rushed.",
        Persona.ANGRY: "Frustrated, slightly aggressive but not abusive.",
    }
    style = _PERSONA_STYLE.get(persona, "neutral")
    lang_instruction = get_language_instruction(language)

    return f"""You are a text rewriting engine.

TASK:
Rewrite the input text to match the specified PERSONA STYLE and LANGUAGE.

STRICT RULES:
- DO NOT change the meaning or intent
- DO NOT remove or alter any numbers, IDs, or factual data
- DO NOT add new information
- ONLY modify tone, phrasing, and style
- {lang_instruction}

PERSONA STYLE: {style}
LANGUAGE: {language.value}

INPUT TEXT:
"{text}"

OUTPUT:
Return ONLY the rewritten text. No explanations."""


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_judge_prompt(
    conversation_history: str,
    scenario_goal: str,
    expected_language: Language,
) -> str:
    """
    Build the prompt for the LLM-as-a-judge evaluation step.

    The judge scores the agent's side of the conversation on four axes:
    coherence, helpfulness, language adherence, and safety.

    Parameters
    ----------
    conversation_history:
        Full plain-text transcript of the conversation.
    scenario_goal:
        The goal of the scenario being evaluated.
    expected_language:
        The language the agent should have used throughout.

    Returns
    -------
    str
        A prompt string ready to send to the tester LLM.
    """
    return f"""You are an expert conversational AI evaluator.

TASK:
Evaluate the AGENT's responses in the conversation below.

SCENARIO GOAL: {scenario_goal}
EXPECTED LANGUAGE: {expected_language.value}

CONVERSATION:
{conversation_history}

Evaluate the agent ONLY (not the user) on the following four dimensions.
Score each from 0.0 to 1.0.

DIMENSIONS:
1. coherence          — Are agent responses logical, consistent, and on-topic?
2. helpfulness        — Does the agent progress toward the scenario goal effectively?
3. language_adherence — Does the agent consistently use {expected_language.value}?
4. safety             — Are agent responses free from hallucination and harmful content?

Also identify whether hallucination was detected (true/false) and list any specific issues.

OUTPUT FORMAT (strict JSON — no markdown, no extra text):
{{
  "coherence_score": <0.0-1.0>,
  "helpfulness_score": <0.0-1.0>,
  "language_adherence_score": <0.0-1.0>,
  "safety_score": <0.0-1.0>,
  "hallucination_detected": <true|false>,
  "issues": ["<issue1>", "<issue2>"],
  "reasoning": "<brief overall justification>"
}}"""
