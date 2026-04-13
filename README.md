# AI Engine Architecture

This repository implements a structured pipeline for scenario‑based AI testing and evaluation.  
The flow moves from specification to analytics in a modular fashion.

---

## 🔄 Pipeline Flow

1. **Agent Spec**  
   Define the agent’s capabilities, intents, and expected behaviors.

2. **Scenario Generator (LLM → structure)**  
   Convert natural language descriptions into structured scenarios using LLMs.

3. **Test Case Generator (mutations + personas + dialects)**  
   Generate diverse test cases by mutating scenarios, applying personas, and adding dialect variations.

4. **Simulator (multi‑turn execution)**  
   Run simulations of multi‑turn conversations between agents and users.

5. **Logging Layer (full trace)**  
   Capture detailed logs of each interaction for debugging and analysis.

6. **Deterministic Evaluation**  
   Apply rule‑based checks and deterministic metrics to validate scenario outcomes.

7. **LLM Judge (optional)**  
   Use an LLM to provide qualitative judgments on conversation quality.

8. **Aggregation**  
   Collect results across scenarios and evaluations.

9. **Analytics Dashboard**  
   Visualize aggregated metrics and insights in a dashboard.

---

## 📂 Repository Structure

```text
ai_engine/
├── llm/
│   ├── tester.py          # LLM-based testing utilities
│   ├── prompts.py         # Prompt templates for scenario generation
│   └── cache.py           # Cache layer for LLM responses
├── utils/
│   └── models.py          # Shared data models
├── scenario/
│   ├── generator.py       # Scenario generation logic
│   ├── parser.py          # Scenario parsing utilities
│   ├── validator.py       # Scenario validation checks
│   └── state_machine.py   # State machine implementation
├── persona/
│   └── engine.py          # Persona and dialect handling
├── mutation/
│   ├── generator.py       # Mutation strategies for test cases
│   └── validator.py       # Mutation validation
├── simulation/
│   ├── agent.py           # Agent simulation logic
│   ├── simulator.py       # Multi-turn simulator
│   └── extractor.py       # Extractor for simulation traces
├── evaluation/
│   ├── engine.py          # Evaluation engine
│   └── metrics/           # Deterministic metrics
├── orchestrator.py        # Orchestration of pipeline stages
└── tests/                 # Unit and integration tests

readme.md
requirements.txt
