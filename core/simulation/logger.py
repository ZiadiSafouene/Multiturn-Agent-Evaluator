"""
core.simulation.logger
~~~~~~~~~~~~~~~~~~~~~~

Conversation trace analysis and formatted logging utilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.utils.models import Conversation, Scenario

logger = logging.getLogger(__name__)


def log_conversation_analysis(conversation: Conversation, scenario: Scenario) -> None:
    """Print an ASCII breakdown of a completed conversation trace."""
    print("\n" + "=" * 60)
    print("           CONVERSATION TRACE ANALYSIS")
    print("=" * 60)
    print(f"Scenario ID: {scenario.scenario_id} - {scenario.title}")
    print(f"Total Turns: {len(conversation.turns)}")
    print("-" * 60)

    for turn in conversation.turns:
        speaker_tag = turn.role.upper().ljust(6)
        print(f"[{turn.turn_id:02d}] {speaker_tag} | {turn.message}")
        if turn.slots_extracted:
            print(f"     [+] Slots Extracted: {turn.slots_extracted}")
        if turn.intent_detected:
            print(f"     [*] Intent Detected: {turn.intent_detected}")

    print("-" * 60)
    print("FINAL MEMORY SNAPSHOT:")
    for k, v in conversation.memory.filled_slots.items():
        print(f"  * {k}: {v}")
    print(f"Final State: {conversation.memory.current_state}")
    print("=" * 60 + "\n")


def log_simulation_summary(result: Dict[str, Any]) -> None:
    """Log a concise summary of simulation execution."""
    summary = result.get("summary", {})
    history: List[Dict[str, Any]] = result.get("history", [])

    logger.info(
        "Simulation ended: reason=%s, turns=%d, final_state=%s",
        summary.get("termination_reason"),
        len(history),
        summary.get("final_state"),
    )
