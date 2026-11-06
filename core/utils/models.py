"""
core.utils.models
~~~~~~~~~~~~~~~~~

Core domain models shared across all modules.

All structures use Python dataclasses for determinism and serializability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class Language(str, Enum):
    """Supported conversation languages for persona simulation."""

    ENGLISH = "english"
    ARABIC = "arabic"
    FRENCH = "french"


class Persona(str, Enum):
    """User persona archetypes used during simulation."""

    COOPERATIVE = "cooperative"   # Clear answers, follows flow
    CONFUSED = "confused"         # Misunderstands, asks for clarification
    IMPATIENT = "impatient"       # Short answers, rushes, skips steps
    ANGRY = "angry"               # Confrontational, escalates quickly


class MutationType(str, Enum):
    """Types of adversarial mutations injected into test cases."""

    SLOT_MISSING = "slot_missing"
    SLOT_INVALID = "slot_invalid"
    SLOT_CONFLICTING = "slot_conflicting"
    STATE_SKIP = "state_skip"
    STATE_REPEAT = "state_repeat"
    INTENT_SWITCH = "intent_switch"
    AMBIGUITY = "ambiguity"
    POLICY_VIOLATION = "policy_violation"


class EvaluationStatus(str, Enum):
    """High-level outcome status of an evaluation report."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SlotDefinition:
    """Definition of a conversational slot (a piece of data to collect)."""

    name: str
    slot_type: str = "string"       # string | date | number | enum
    required: bool = True
    pattern: Optional[str] = None
    minvalue: Optional[int] = None
    maxvalue: Optional[int] = None
    valid_values: list[str] = field(default_factory=list)
    description: str = ""
    aliases: list[str] = field(default_factory=list)  # language-agnostic aliases


@dataclass
class Mutation:
    """An adversarial deviation injected into a test case."""

    type: MutationType
    target: str
    turn_index: int
    description: str
    expected_behavior: str


@dataclass
class StateTransition:
    """An edge in the scenario state machine."""

    from_state: str
    to_state: str
    trigger_intent: str
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnTemplate:
    """A template describing a single expected conversational turn."""

    state: str
    expected_intent: str
    agent_action: str               # ask | confirm | execute | fallback
    role: Literal["user", "agent"]
    intent: str
    slots_to_fill: list[str] = field(default_factory=list)


@dataclass
class State:
    """A node in the scenario state machine."""

    state_id: str
    required_slots: list[str] = field(default_factory=list)
    expected_actions: list[str] = field(default_factory=list)
    next_states: list[str] = field(default_factory=list)
    policy_checks: list[str] = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class Scenario:
    """A complete test scenario definition with states, slots, and transitions."""

    scenario_id: str
    title: str
    description: str
    goal: str
    domain: str
    slots: list[SlotDefinition] = field(default_factory=list)
    states: list[State] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    turns: list[TurnTemplate] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize the scenario to a plain dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "description": self.description,
            "goal": self.goal,
            "domain": self.domain,
            "slots": [vars(s) for s in self.slots],
            "states": [vars(s) for s in self.states],
            "transitions": [vars(t) for t in self.transitions],
            "turns": [vars(t) for t in self.turns],
            "edge_cases": self.edge_cases,
            "policies": self.policies,
        }

    def __str__(self) -> str:
        lines = [
            f"Scenario {self.scenario_id}: {self.title}",
            f"  Domain : {self.domain}",
            f"  Goal   : {self.goal}",
            f"  Slots  : {', '.join(s.name for s in self.slots)}",
            f"  States : {', '.join(s.state_id for s in self.states)}",
        ]
        return "\n".join(lines)


@dataclass
class Action:
    """An action taken by the agent during a turn."""

    name: str
    parameters: dict[str, Any]
    success: bool


# ─────────────────────────────────────────────────────────────────────────────
# Test Case Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TestCase:
    """A generated test case combining a scenario with persona, language, and mutation."""

    __test__ = False  # Prevent pytest from collecting this model as a test suite

    test_id: str
    scenario_id: str
    mutation: Optional[Mutation]
    language: Language
    persona: Persona
    description: str
    initial_slots: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""          # success | failure | recovery | escalation
    expected_slots_filled: list[str] = field(default_factory=list)
    injected_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"TestCase({self.test_id})\n"
            f"  Scenario : {self.scenario_id}\n"
            f"  Persona  : {self.persona.value}\n"
            f"  Language : {self.language.value}\n"
            f"  Mutation : {self.mutation.type.value if self.mutation else 'none'}\n"
            f"  Outcome  : {self.expected_outcome}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Turn:
    """A single conversational turn (one message from one speaker)."""

    turn_id: int
    role: str                           # user | agent
    message: str
    language: Optional[Language] = None
    intent_detected: Optional[str] = None
    slots_extracted: dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[Action] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ConversationMemory:
    """Tracks the evolving state of a conversation."""

    filled_slots: dict[str, Any] = field(default_factory=dict)
    current_state: str = "start"
    visited_states: list[str] = field(default_factory=list)
    intent_history: list[str] = field(default_factory=list)
    failed_attempts: dict[str, int] = field(default_factory=dict)


@dataclass
class Conversation:
    """A complete record of a simulated multi-turn conversation."""

    conversation_id: str
    test_case_id: str
    scenario_id: str
    turns: list[Turn] = field(default_factory=list)
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    status: str = "in_progress"         # in_progress | completed | failed
    total_turns: int = 0

    def add_turn(self, role: str, message: str, **kwargs: Any) -> Turn:
        """Append a new turn and return it."""
        turn = Turn(turn_id=len(self.turns), role=role, message=message, **kwargs)
        self.turns.append(turn)
        self.total_turns += 1
        return turn

    def get_history_text(self) -> str:
        """Return the full conversation as a plain-text transcript."""
        lines: list[str] = []
        for t in self.turns:
            prefix = "User" if t.role == "user" else "Agent"
            lines.append(f"{prefix}: {t.message}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DeterministicResult:
    """
    Rule-based evaluation result — no LLM involved.

    Scoring weights
    ---------------
    40 pts — task completion
    30 pts — slot fill accuracy
    15 pts — state transition correctness
    15 pts — policy compliance
    """

    task_completed: bool
    slot_accuracy: float                    # 0.0 – 1.0
    state_errors: list[str]
    policy_violations: list[str]
    missing_slots: list[str]
    invalid_slots: list[str]
    tool_calls_valid: bool
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute and cache the weighted deterministic score (0–100)."""
        score = 0.0
        score += 40.0 if self.task_completed else 0.0
        score += 30.0 * self.slot_accuracy
        score += 15.0 if not self.state_errors else max(0.0, 15.0 - 5.0 * len(self.state_errors))
        score += 15.0 if not self.policy_violations else max(0.0, 15.0 - 5.0 * len(self.policy_violations))
        self.score = round(score, 2)
        return self.score


@dataclass
class LLMJudgeResult:
    """
    LLM-as-a-judge evaluation result.

    Scoring weights
    ---------------
    35 pts — coherence
    35 pts — helpfulness
    15 pts — language adherence (correct language used throughout)
    15 pts — safety (no hallucination)
    """

    coherence_score: float              # 0.0 – 1.0
    helpfulness_score: float            # 0.0 – 1.0
    language_adherence_score: float     # 0.0 – 1.0
    safety_score: float                 # 0.0 – 1.0
    hallucination_detected: bool
    issues: list[str]
    reasoning: str
    score: float = 0.0

    def compute_score(self) -> float:
        """Compute and cache the weighted LLM judge score (0–100)."""
        hallucination_penalty = 0.0 if self.hallucination_detected else 15.0
        score = (
            self.coherence_score * 35
            + self.helpfulness_score * 35
            + self.language_adherence_score * 15
            + hallucination_penalty
        )
        self.score = round(score, 2)
        return self.score


@dataclass
class EvaluationReport:
    """Aggregated evaluation output for a single test-case run."""

    report_id: str
    conversation_id: str
    test_case_id: str
    scenario_id: str
    status: EvaluationStatus
    deterministic: DeterministicResult
    llm_judge: Optional[LLMJudgeResult]
    final_score: float = 0.0
    summary: str = ""
    language: Optional[Language] = None
    persona: Optional[Persona] = None
    mutation: Optional[Mutation] = None

    def compute_final_score(self) -> float:
        """
        Deterministic result carries 70 %, LLM judge 30 %.
        Falls back to 100 % deterministic when no LLM judge is available.
        """
        det_score = self.deterministic.score
        if self.llm_judge:
            self.final_score = round(det_score * 0.7 + self.llm_judge.score * 0.3, 2)
        else:
            self.final_score = round(det_score, 2)
        return self.final_score

    def to_dict(self) -> dict:
        """Serialize the report to a plain dictionary."""
        return {
            "report_id": self.report_id,
            "conversation_id": self.conversation_id,
            "test_case_id": self.test_case_id,
            "scenario_id": self.scenario_id,
            "status": self.status.value,
            "final_score": self.final_score,
            "summary": self.summary,
            "language": self.language.value if self.language else None,
            "persona": self.persona.value if self.persona else None,
            "mutation_type": self.mutation.type.value if self.mutation else None,
            "deterministic": {
                "task_completed": self.deterministic.task_completed,
                "slot_accuracy": self.deterministic.slot_accuracy,
                "state_errors": self.deterministic.state_errors,
                "policy_violations": self.deterministic.policy_violations,
                "missing_slots": self.deterministic.missing_slots,
                "score": self.deterministic.score,
            },
            "llm_judge": {
                "coherence_score": self.llm_judge.coherence_score,
                "helpfulness_score": self.llm_judge.helpfulness_score,
                "language_adherence_score": self.llm_judge.language_adherence_score,
                "safety_score": self.llm_judge.safety_score,
                "hallucination_detected": self.llm_judge.hallucination_detected,
                "issues": self.llm_judge.issues,
                "score": self.llm_judge.score,
            } if self.llm_judge else None,
        }
