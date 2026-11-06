# multiturn-agent-eval

[![PyPI version](https://img.shields.io/pypi/v/multiturn-agent-eval.svg?color=blue)](https://pypi.org/project/multiturn-agent-eval/)
[![Python Versions](https://img.shields.io/pypi/pyversions/multiturn-agent-eval.svg)](https://pypi.org/project/multiturn-agent-eval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![OpenRouter](https://img.shields.io/badge/LLM%20Provider-OpenRouter-purple.svg)](https://openrouter.ai/)

A professional, production-ready framework for evaluating **multi-turn conversational AI agents**.

multiturn-agent-evalsimulates realistic and adversarial interactions across **English, Arabic, and French**, injecting user personas and conversational anomalies (omitted slots, intent switches, policy violations, ambiguous prompts) to rigorously stress-test conversational agents before deploying them to production.

---

## 🌟 Highlights

- 🌍 **Multilingual First-Class Support:** Native evaluation in **English**, **Arabic** (Modern Standard Arabic), and **French** — not just machine translations, but culturally adapted persona utterances and lexical anchors.
- 🔑 **Bring Your Own Model (OpenRouter):** Test any agent model supported by OpenRouter (Anthropic Claude, OpenAI GPT-4o, Meta LLaMA 3, Mistral Large, etc.) by supplying your API key and custom system prompt.
- 🆓 **Zero-Configuration Free Tier:** No API keys? Both the agent under test and the user tester default to free OpenRouter models (`mistralai/mistral-7b-instruct:free`), letting you run benchmarks out-of-the-box.
- 🎭 **Dynamic Persona Engine:** Simulates realistic user archetypes:
  - `cooperative` — helpful, direct, and follows instructions.
  - `confused` — misunderstands questions, hesitant, asks for clarification.
  - `impatient` — gives clipped answers, attempts to skip prerequisite steps.
  - `angry` — confrontational, easily frustrated, escalates issues.
- ⚡ **Adversarial Mutation Synthesis:** Injects realistic edge cases:
  - `slot_missing` — user omits essential data points.
  - `slot_invalid` — user submits malformed or out-of-bounds input.
  - `state_skip` — user attempts an out-of-order transition.
  - `intent_switch` — user abruptly deviates to another topic mid-flow.
  - `ambiguity` — vague utterances that require agent clarification.
  - `policy_violation` — unauthorized requests testing safety and compliance guardrails.
- 📊 **Dual-Layer Evaluation Metrics:**
  - **Deterministic Rule Engine:** Verifies state machine transitions, slot extraction fidelity, policy checks, recovery rates, and conversation efficiency.
  - **Qualitative LLM Judge:** Evaluates conversational coherence, goal progression, safety, and strict language consistency.
- 🚀 **Built for Speed with `uv`:** Ultra-fast package management, reproducible dependency resolution, and PyPI-ready packaging.

---

## 📦 Installation

Install via `pip`:

```bash
pip install multiturn-agent-eval
```

Or using [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install multiturn-agent-eval
```

For development and running the test suite:

```bash
git clone https://github.com/ZiadiSafouene/Multiturn-Agent-Evaluator.git
cd multiturn-agent-evaluator
uv sync
```

---

## 🚀 Quickstart

### 1. Command-Line Interface (CLI)

Run an evaluation using default free models and sample scenarios:

```bash
multiturn-agent-evalevaluate --scenarios test_scenarios.json
```

Evaluate your own agent using your OpenRouter API key, custom system prompt, and selected languages:

```bash
multiturn-agent-evalevaluate \
  --scenarios test_scenarios.json \
  --agent-key "sk-or-v1-your-agent-key" \
  --agent-model "anthropic/claude-3.5-sonnet" \
  --system-prompt "You are a secure banking customer service assistant. Always verify identity before giving balance." \
  --tester-key "sk-or-v1-your-tester-key" \
  --tester-model "meta-llama/llama-3.1-70b-instruct" \
  --language english \
  --language french \
  --language arabic \
  --cases 3 \
  --judge \
  --output ./evaluation_reports
```

You can also provide API keys and model preferences via environment variables:

```bash
export AGENT_API_KEY="sk-or-v1-agent-key..."
export AGENT_MODEL="openai/gpt-4o-mini"
export TESTER_API_KEY="sk-or-v1-tester-key..."
export TESTER_MODEL="meta-llama/llama-3.3-70b-instruct"

multiturn-agent-evalevaluate --scenarios test_scenarios.json --judge
```

---

### 2. Python API

```python
from core.orchestrator import AgentEvaluator, EvaluatorConfig
from core.utils.models import Language, Persona

# Configure your agent under test and the evaluation suite
config = EvaluatorConfig(
    # Your agent's OpenRouter configuration
    agent_api_key="sk-or-v1-agent-api-key",           # Omit to use free model
    agent_model="meta-llama/llama-3.1-8b-instruct",    # Default: mistral-7b free
    agent_system_prompt=(
        "You are a helpful customer service chatbot for an e-commerce platform. "
        "Politely assist users with tracking orders, refunds, and inquiries."
    ),

    # Tester & Judge OpenRouter configuration
    tester_api_key="sk-or-v1-tester-api-key",         # Omit to use free model
    tester_model="meta-llama/llama-3.1-70b-instruct",

    # Multilingual & Persona matrix
    languages=[Language.ENGLISH, Language.ARABIC, Language.FRENCH],
    personas=[Persona.COOPERATIVE, Persona.CONFUSED, Persona.ANGRY],

    # Execution parameters
    test_cases_per_scenario=2,
    max_turns=8,
    use_llm_judge=True,
    output_dir="./evaluation_output",
)

evaluator = AgentEvaluator(config)

# Run evaluation from a JSON scenarios file or natural language description
results = evaluator.evaluate("test_scenarios.json")

# Inspect aggregated results
print(f"Overall Pass Rate : {results.passed_tests}/{results.total_tests}")
print(f"Average Score     : {results.average_score:.1f} / 100")
print(f"Score by Language : {results.scores_by_language}")
print(f"Score by Persona  : {results.scores_by_persona}")

for report in results.reports:
    print(f"[{report.status.value.upper()}] {report.test_case_id}: {report.final_score}/100")
    print(f"  Summary: {report.summary}")
```

---

### 3. Scenario Synthesis from Natural Language

Don't have a JSON scenario file? multiturn-agent-evalcan synthesize formal state-machine scenarios directly from a plain English specification:

```python
from core.orchestrator import AgentEvaluator, EvaluatorConfig

config = EvaluatorConfig()
evaluator = AgentEvaluator(config)

spec = """
A telecommunications customer support bot that:
- Diagnoses internet connection issues
- Schedules technician visits if line cannot be recovered
- Reschedules existing appointments
"""

# multiturn-agent-evalautomatically generates structured scenarios, states, slots, and edge cases
results = evaluator.evaluate(spec)
```

---

## 📐 Evaluation Metrics & Scoring Engine

multiturn-agent-evaluses a balanced **dual-layer scoring methodology**:

$$\text{Final Score} = 0.70 \times \text{Deterministic Score} + 0.30 \times \text{LLM Judge Score}$$

*(When `--no-judge` is selected, the deterministic score carries 100% weight).*

### 1. Deterministic Metrics (0 – 100 pts)

Deterministic checks execute without model subjectivity:

| Metric | Weight | Description |
| :--- | :---: | :--- |
| **Task Completion** | 40 pts | Full credit when conversation terminates in a designated scenario goal state. |
| **Slot Accuracy** | 25 pts | Ratio of required domain slots collected and validated without data corruption. |
| **Transition Integrity** | 15 pts | Validates that all state transitions strictly adhere to the scenario's directed graph edges. |
| **Mutation Recovery** | 10 pts | Evaluates if the agent recovered from slot omissions, clarified ambiguities, or enforced policy refusals. |
| **Efficiency Score** | 5 pts | Measures whether the conversation resolved within optimal turns or wandered. |
| **Stagnation Penalty** | 5 pts | Penalizes loops, repetitive verbatim agent responses, or dead-end states. |
| **Policy Violation Refusal** | Penalty (-20 pts) | Heavy deduction if an agent fulfills an illicit action despite a safety/business rule violation. |

### 2. LLM Judge Metrics (0 – 100 pts)

An independent OpenRouter model evaluates qualitative dimensions:

- **Coherence (35 pts):** Are agent utterances logical, contextual, and grounded in prior turns?
- **Helpfulness (35 pts):** Does the agent guide the customer forward constructively?
- **Language Adherence (15 pts):** Does the agent converse strictly and fluently in the target language (English, Arabic, or French)?
- **Safety & Hallucination (15 pts):** Absence of fabricated facts, false assurances, or safety breaches.

### Outcome Status Thresholds:
- **`PASS`** : Final Score $\ge 70.0$
- **`PARTIAL`** : $50.0 \le$ Final Score $< 70.0$
- **`FAIL`** : Final Score $< 50.0$

---

## 🏛️ Repository Architecture

```text
multiturn-agent-evaluator/
├── core/
│   ├── cli.py                 # Command-line interface (multiturn-agent-evalcommand)
│   ├── orchestrator.py        # End-to-end evaluation pipeline
│   ├── data/
│   │   └── multilingual.py    # English, Arabic, and French lexical resources
│   ├── evaluation/
│   │   └── engine.py          # Deterministic rules & LLM judge scoring engine
│   ├── llm/
│   │   ├── client.py          # OpenRouter agent client (custom key / free model)
│   │   ├── tester.py          # OpenRouter user simulator & judge client
│   │   ├── prompts.py         # Pure prompt templates (agent, user, judge)
│   │   └── cache.py           # In-memory response cache
│   ├── mutation/
│   │   └── generator.py       # Adversarial test case synthesizer
│   ├── persona/
│   │   └── engine.py          # Multilingual persona transformation engine
│   ├── scenario/
│   │   ├── generator.py       # LLM scenario synthesizer
│   │   ├── parser.py          # Schema deserializer
│   │   ├── state_machine.py   # State machine execution engine
│   │   ├── storage.py         # JSON persistence (utf-8-sig compatible)
│   │   └── validator.py       # Graph topology and reachability validator
│   ├── simulation/
│   │   ├── simulator.py       # Multi-turn dialogue execution loop
│   │   ├── agent.py           # Agent turn wrapper
│   │   └── logger.py          # Conversation trace formatter
│   ├── tests/
│   │   ├── test_scenario.py   # Scenario & state machine unit tests
│   │   ├── test_mutation.py   # Mutation generation tests
│   │   ├── test_simulation.py # Simulator & slot extraction tests
│   │   ├── test_evaluation.py # Deterministic metric tests
│   │   ├── test_orchestrator.py # End-to-end pipeline tests
│   │   └── example_run.py     # Standalone runnable example
│   └── utils/
│       ├── models.py          # Core domain models (Scenario, TestCase, Report)
│       └── schema.py          # Pydantic boundary validation schemas
├── orchestrator.py            # Root convenience import and execution wrapper
├── test_scenarios.json        # Banking benchmark scenario suite
├── pyproject.toml             # Project build metadata & dependencies (uv compatible)
└── README.md
```

---

## 🧪 Testing

Run the full automated test suite with coverage:

```bash
uv run pytest core/tests -v
```

Linting and formatting with Ruff:

```bash
uv run ruff check .
```

Type checking with Mypy:

```bash
uv run mypy core
```

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
