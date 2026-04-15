from ai_engine.utils.models import Conversation
from ai_engine.simulation.agent import run_turn


def simulate_conversation(scenario, test_case, llm_client, max_turns=10):
    conversation = Conversation(
        conversation_id=f"conv_{test_case.test_id}",
        test_case_id=test_case.test_id,
        scenario_id=scenario.scenario_id,
    )

    memory = conversation.memory
    memory.current_state = "start"

    for step in range(max_turns):
        state = memory.current_state

        # ── Agent turn ─────────────────────
        agent_msg = run_turn(
            role="agent",
            scenario=scenario,
            test_case=test_case,
            conversation=conversation,
            llm_client=llm_client,
        )
        conversation.add_turn("agent", agent_msg)

        # ── User turn ─────────────────────
        user_msg = run_turn(
            role="user",
            scenario=scenario,
            test_case=test_case,
            conversation=conversation,
            llm_client=llm_client,
        )
        conversation.add_turn("user", user_msg)

        # ── Stop if terminal ──────────────
        if _is_terminal_state(scenario, memory.current_state):
            conversation.status = "completed"
            break

    return conversation


def _is_terminal_state(scenario, state_id):
    for s in scenario.states:
        if s.state_id == state_id:
            return s.is_terminal
    return False