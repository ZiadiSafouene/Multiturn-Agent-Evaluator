1. Agent Spec
      ↓
2. Scenario Generator (LLM → structure)
      ↓
3. Test Case Generator (mutations + personas + dialects)
      ↓
4. Simulator (multi-turn execution)
      ↓
5. Logging Layer (full trace)
      ↓
6. Deterministic Evaluation
      ↓
7. LLM Judge (optional)
      ↓
8. Aggregation
      ↓
9. Analytics Dashboard

-----------------------------


ai_engine/
├── llm/
│   ├── client.py
│   ├── prompts.py
│   └── cache.py
├── utils/
│   └── models.py
├── scenario/
│   ├── generator.py
│   ├── parser.py
│   ├── validator.py      
│   └── state_machine.py  
├── persona/
│   └── engine.py
├── mutation/
│   ├── generator.py
│   └── validator.py     
├── simulation/
│   ├── agent.py
│   ├── simulator.py
│   └── extractor.py      
├── evaluation/
│   ├── engine.py
│   └── metrics/          
├── orchestrator.py
└── tests/            
readme.md
requirements.txt
