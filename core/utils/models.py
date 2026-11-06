"""
Core domain models — shared across all modules.
All structures are dataclasses (deterministic, serializable).
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from enum import Enum
import uuid
from datetime import datetime


# ─── Enums ────────────────────────────────────────────────────────────────────

class Dialect(str, Enum):
    MSA = "msa"                    # Modern Standard Arabic
    EGYPTIAN = "egyptian"          # Egyptian Arabic (Masri)
    SAUDI = "saudi"                # Saudi Arabic (Najdi/Hejazi)
    MIXED = "mixed"                # Arabic-English code-switching


class Persona(str, Enum):
    COOPERATIVE = "cooperative"    # Clear answers, follows flow
    CONFUSED = "confused"          # Misunderstands, asks for clarification
    IMPATIENT = "impatient"        # Short answers, rushes, repeats
    ANGRY = "angry"                # Confrontational, escalates


class MutationType(str, Enum):
    SLOT_MISSING = "slot_missing"
    SLOT_INVALID = "slot_invalid"
    SLOT_CONFLICTING = "slot_conflicting"
    STATE_SKIP = "state_skip"
    STATE_REPEAT = "state_repeat"
    INTENT_SWITCH = "intent_switch"
    AMBIGUITY = "ambiguity"
    POLICY_VIOLATION = "policy_violation"


class EvaluationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"



# ─── Scenario Models ──────────────────────────────────────────────────────────

@dataclass
class SlotDefinition:
    name: str
    slot_type: str = "string"               # "string" | "date" | "number" | "enum"
    required: bool = True
    pattern: Optional[str] = None
    minvalue: Optional[int] = None
    maxvalue: Optional[int] = None
    valid_values: list[str] = field(default_factory=list)
    description: str = ""
    arabic_aliases: list[str] = field(default_factory=list)

@dataclass
class Mutation:
    type: MutationType
    target: str
    turn_index: int
    description: str
    expected_behavior: str


@dataclass
class StateTransition:
    from_state: str
    to_state: str
    trigger_intent: str
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnTemplate:
    state: str
    expected_intent: str
    agent_action: str                       # "ask" | "confirm" | "execute" | "fallback"
    role: Literal["user", "agent"]
    intent: str
    slots_to_fill: list[str] = field(default_factory=list)

@dataclass
class State:
    state_id: str
    required_slots: list[str] = field(default_factory=list)
    expected_actions: list[str] = field(default_factory=list)
    next_states: list[str] = field(default_factory=list)
    policy_checks: list[str] = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class Scenario:
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
    def __str__(self):
        lines = [
            f"Scenario {self.scenario_id}: {self.title}",
            f"  Domain: {self.domain}",
            f"  Goal: {self.goal}",
            f"  Description: {self.description}",
            f"  Slots:",
        ]
        for slot in self.slots:
            lines.append(f"    - {slot.name} ({slot.slot_type}, required={slot.required})")

        lines.append("  States:")
        for state in self.states:
            lines.append(f"    - {state.state_id} (terminal={state.is_terminal}, next={state.next_states})")

        lines.append("  Transitions:")
        for t in self.transitions:
            lines.append(f"    - {t.from_state} -> {t.to_state} (intent={t.trigger_intent})")

        lines.append("  Turns:")
        for turn in self.turns:
            lines.append(f"    - State={turn.state}, Intent={turn.intent}, Action={turn.agent_action}")

        return "\n".join(lines)


@dataclass
class Action:
    name: str
    parameters: dict[str, Any]
    success: bool

# ─── Test Case Models ─────────────────────────────────────────────────────────

@dataclass
class TestCase:
    test_id: str
    scenario_id: str
    mutation: Optional[Mutation] 
    dialect: Dialect
    persona: Persona
    description: str
    initial_slots: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""              # "success" | "failure" | "escalation"
    expected_slots_filled: list[str] = field(default_factory=list)
    injected_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"TestCase:\n"
            f"  Test ID: {self.test_id}\n"
            f"  Scenario ID: {self.scenario_id}\n"
            f"  Mutation: {self.mutation if self.mutation else 'None'}\n"
            f"  Dialect: {self.dialect}\n"
            f"  Persona: {self.persona}\n"
            f"  Description: {self.description}\n"
            f"  Initial Slots: {self.initial_slots if self.initial_slots else '{}'}\n"
            f"  Expected Outcome: {self.expected_outcome}\n"
            f"  Expected Slots Filled: {', '.join(self.expected_slots_filled) if self.expected_slots_filled else '[]'}\n"
            f"  Injected Errors: {', '.join(self.injected_errors) if self.injected_errors else '[]'}\n"
            f"  Metadata: {self.metadata if self.metadata else '{}'}\n"
        )


# ─── Conversation Models ──────────────────────────────────────────────────────

@dataclass
class Turn:
    turn_id: int
    role: str                               # "user" | "agent"
    message: str
    dialect: Optional[Dialect] = None
    intent_detected: Optional[str] = None
    slots_extracted: dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[Action] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConversationMemory:
    filled_slots: dict[str, Any] = field(default_factory=dict)
    current_state: str = "start"
    visited_states: list[str] = field(default_factory=list)
    intent_history: list[str] = field(default_factory=list)
    failed_attempts: dict[str, int] = field(default_factory=dict)


@dataclass
class Conversation:
    conversation_id: str
    test_case_id: str
    scenario_id: str
    turns: list[Turn] = field(default_factory=list)
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    status: str = "in_progress"            # "in_progress" | "completed" | "failed"
    total_turns: int = 0

    def add_turn(self, role: str, message: str, **kwargs) -> Turn:
        turn = Turn(turn_id=len(self.turns), role=role, message=message, **kwargs)
        self.turns.append(turn)
        self.total_turns += 1
        return turn

    def get_history_text(self) -> str:
        lines = []
        for t in self.turns:
            prefix = "User" if t.role == "user" else "Agent"
            lines.append(f"{prefix}: {t.message}")
        return "\n".join(lines)


# ─── Evaluation Models ────────────────────────────────────────────────────────

@dataclass
class DeterministicResult:
    task_completed: bool
    slot_accuracy: float                    # 0.0 – 1.0
    state_errors: list[str] 
    policy_violations: list[str]
    missing_slots: list[str]
    invalid_slots: list[str]
    tool_calls_valid: bool
    score: float = 0.0

    def compute_score(self) -> float:
        score = 0.0
        score += 40.0 if self.task_completed else 0.0
        score += 30.0 * self.slot_accuracy
        score += 15.0 if not self.state_errors else max(0, 15 - 5 * len(self.state_errors))
        score += 15.0 if not self.policy_violations else max(0, 15 - 5 * len(self.policy_violations))
        self.score = round(score, 2)
        return self.score


@dataclass
class LLMJudgeResult:
    coherence_score: float                  # 0.0 – 1.0
    helpfulness_score: float
    hallucination_detected: bool
    issues: list[str]
    reasoning: str
    score: float = 0.0

    def compute_score(self) -> float:
        score = (self.coherence_score * 40 + self.helpfulness_score * 40
                 + (0 if self.hallucination_detected else 20))
        self.score = round(score, 2)
        return self.score


@dataclass
class EvaluationReport:
    report_id: str
    conversation_id: str
    test_case_id: str
    scenario_id: str
    status: EvaluationStatus
    deterministic: DeterministicResult
    llm_judge: Optional[LLMJudgeResult]
    final_score: float = 0.0
    summary: str = ""
    dialect: Optional[Dialect] = None
    persona: Optional[Persona] = None
    mutation: Optional[Mutation] = None

    def compute_final_score(self) -> float:
        """Deterministic result carries 70%, LLM judge 30%."""
        det_score = self.deterministic.score
        if self.llm_judge:
            llm_score = self.llm_judge.score
            self.final_score = round(det_score * 0.7 + llm_score * 0.3, 2)
        else:
            self.final_score = round(det_score, 2)
        return self.final_score

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "conversation_id": self.conversation_id,
            "test_case_id": self.test_case_id,
            "scenario_id": self.scenario_id,
            "status": self.status.value,
            "final_score": self.final_score,
            "summary": self.summary,
            "dialect": self.dialect.value if self.dialect else None,
            "persona": self.persona.value if self.persona else None,
            "mutation_type": self.mutation.type if self.mutation else None,
            "deterministic": {
                "task_completed": self.deterministic.task_completed,
                "slot_accuracy": self.deterministic.slot_accuracy,
                "state_transition_correct": self.deterministic.state_transition_correct,
                "policy_violations": self.deterministic.policy_violations,
                "missing_slots": self.deterministic.missing_slots,
                "score": self.deterministic.score,
            },
            "llm_judge": {
                "coherence_score": self.llm_judge.coherence_score,
                "helpfulness_score": self.llm_judge.helpfulness_score,
                "hallucination_detected": self.llm_judge.hallucination_detected,
                "issues": self.llm_judge.issues,
                "score": self.llm_judge.score,
            } if self.llm_judge else None,
        }

