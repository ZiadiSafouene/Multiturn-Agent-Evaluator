

from ai_engine.llm.prompts import build_client_llm_prompt, build_simulation_prompt

def run_turn(role, scenario, test_case, conversation, llm_client):
    prompt = build_simulation_prompt(
        role=role,
        scenario=scenario,
        test_case=test_case,
        history=conversation.get_history_text(),
        memory=conversation.memory,
    )

    response = llm_client.generate(prompt)

    return response.strip()