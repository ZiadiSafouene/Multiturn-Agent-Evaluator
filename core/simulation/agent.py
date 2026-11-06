"""
core.simulation.agent
~~~~~~~~~~~~~~~~~~~~~

Agent turn execution wrapper for conversational simulations.
"""

from __future__ import annotations

from typing import Any

from core.llm.client import LLMClient
from core.llm.prompts import build_simulation_prompt
from core.utils.models import Conversation, Scenario, TestCase


def run_turn(
    role: str,
    scenario: Scenario,
    test_case: TestCase,
    conversation: Conversation,
    llm_client: LLMClient,
) -> str:
    """
    Execute a single turn for either the agent or the simulated user.

    Parameters
    ----------
    role:
        ``"agent"`` or ``"user"``.
    scenario:
        Active :class:`~core.utils.models.Scenario`.
    test_case:
        Active :class:`~core.utils.models.TestCase`.
    conversation:
        Active :class:`~core.utils.models.Conversation` instance.
    llm_client:
        Client instance used to generate the turn message.

    Returns
    -------
    str
        The generated message text.
    """
    prompt = build_simulation_prompt(
        role=role,
        scenario=scenario,
        test_case=test_case,
        history=conversation.get_history_text(),
        memory=conversation.memory,
    )

    response = llm_client.generate(prompt)
    return response.strip()
