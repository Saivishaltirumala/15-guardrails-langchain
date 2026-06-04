# Guardrails with LangChain

A progressive, hands-on collection of projects exploring guardrails patterns with LangChain — input validation, output parsing, content moderation, hallucination detection, and safety layers for LLM applications.

## The Stack

```
┌─────────────────────────────────────────────────┐
│  Guardrails AI / NeMo    ← guardrail frameworks │
│  LangChain 1.x           ← LLM orchestration    │
│  Groq                    ← LLM provider (fast)   │
│  Python                  ← runtime               │
└─────────────────────────────────────────────────┘
```

## Projects

| # | Project | Concepts Covered | Location |
|---|---------|-----------------|----------|
| 01 | Deterministic Guardrails | Keyword blocking, PII regex detection, input/output filtering, redaction, agent flow | `project-01-deterministic-guardrails/` |
| 02 | Model-Based Guardrail | LLM-as-judge safety classifier (SAFE/UNSAFE), gate before main LLM call | `project-02-model-based-guardrail/` |
| 03 | PII Middleware | Official LangChain PIIMiddleware (redact/mask/block strategies), create_agent, custom regex detector | `project-03-pii-middleware/` |
| 04 | Human-in-the-Loop | HumanInTheLoopMiddleware, interrupt/resume flow, approve/reject decisions, InMemorySaver checkpointer | `project-04-human-in-loop/` |
| 05 | Custom Guardrails | Custom AgentMiddleware with before_agent (keyword blocking) + after_agent (output truncation, forbidden phrase filter) | `project-05-custom-guardrails/` |
| 06 | Combined Guardrails | 5-layer production stack: ContentFilter + PII mask + HITL + PII redact output + model-based safety | `project-06-combined-guardrails/` |

## Setup

```bash
cd 15-guardrails-langchain
python3.10 -m venv venv   # Requires Python 3.10+ (langchain 1.x)
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```
