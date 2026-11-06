from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum

# --- Enums for restricted values ---

class SlotType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    ENUM = "enum"

class AgentAction(str, Enum):
    ASK = "ask"
    CONFIRM = "confirm"
    EXECUTE = "execute"
    FALLBACK = "fallback"

# --- Sub-Models ---

class Slot(BaseModel):
    name: str
    slot_type: SlotType
    required: bool = True
    arabic_aliases: List[str] = Field(default_factory=list)

class State(BaseModel):
    state_id: str
    required_slots: List[str]
    next_states: List[str]
    is_terminal: bool = False

class Transition(BaseModel):
    from_state: str
    to_state: str
    trigger_intent: str

class Turn(BaseModel):
    state: str
    expected_intent: str
    agent_action: AgentAction
    role: Literal["user", "agent"]  # Added "agent" as a logical option
    intent: str

# --- Main Model ---

class Scenario(BaseModel):
    scenario_id: str
    title: str
    description: str
    goal: str
    domain: str
    slots: List[Slot]
    states: List[State]
    transitions: List[Transition]
    turns: List[Turn]
