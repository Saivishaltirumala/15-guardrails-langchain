# Guardrails with LangChain

A progressive, hands-on collection of projects exploring guardrails patterns with LangChain — input validation, output parsing, content moderation, hallucination detection, and safety layers for LLM applications.

## The Stack

```
┌─────────────────────────────────────────────────┐
│  Guardrails AI / NeMo    ← guardrail frameworks │
│  LangChain               ← LLM orchestration    │
│  Groq                    ← LLM provider (fast)   │
│  Python                  ← runtime               │
└─────────────────────────────────────────────────┘
```

## Projects

| # | Project | Concepts Covered | Location |
|---|---------|-----------------|----------|
| 01 | Deterministic Guardrails | Keyword blocking, PII regex detection, input/output filtering, redaction, agent flow | `project-01-deterministic-guardrails/` |
| 02 | Model-Based Guardrail | LLM-as-judge safety classifier (SAFE/UNSAFE), gate before main LLM call | `project-02-model-based-guardrail/` |
| 03 | PII Middleware | Reusable PIIMiddleware class (redact/mask/block strategies), middleware stack, agent integration | `project-03-pii-middleware/` |

## Setup

```bash
cd 15-guardrails-langchain
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```
