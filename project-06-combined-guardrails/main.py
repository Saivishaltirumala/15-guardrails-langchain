import os
import sys
import uuid
from typing import Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    hook_config,
)
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv(dotenv_path="../.env")

if not os.getenv("GROQ_API_KEY"):
    sys.exit("ERROR: Set GROQ_API_KEY in ../.env file")


# ======================================================
# Layer 1: Deterministic input filter (before_agent hook)
# ======================================================

class ContentFilterMiddleware(AgentMiddleware):
    """Block requests containing banned keywords before LLM runs — zero cost."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()
        for keyword in self.banned_keywords:
            if keyword in content:
                print(f"  [Layer 1 - ContentFilter] BLOCKED — keyword: '{keyword}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "Request blocked: inappropriate content detected.",
                    }],
                    "jump_to": "end",
                }

        print(f"  [Layer 1 - ContentFilter] PASSED")
        return None


# ======================================================
# Layer 5: Model-based output safety (after_agent hook)
# ======================================================

class SafetyGuardrailMiddleware(AgentMiddleware):
    """Use a guard LLM to classify the agent's final output as SAFE or UNSAFE."""

    def __init__(self):
        super().__init__()
        self.guard_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        content = last_message.content

        verdict = self.guard_llm.invoke(
            f"You are a safety classifier. Given this AI response, reply with "
            f"exactly one word: SAFE or UNSAFE.\n\n"
            f"Mark as UNSAFE if it contains: harmful instructions, hate speech, "
            f"violence, illegal activity, or dangerous misinformation.\n\n"
            f"Response to check:\n{content}"
        ).content.strip().upper()

        if "UNSAFE" in verdict:
            print(f"  [Layer 5 - SafetyGuard] BLOCKED — output deemed UNSAFE")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Response filtered: safety check failed.",
                }]
            }

        print(f"  [Layer 5 - SafetyGuard] PASSED — output is SAFE")
        return None


# ======================================================
# Tools
# ======================================================

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results: {query}"


@tool
def send_email_tool(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


# ======================================================
# Production agent with 5-layer guardrail stack
# ======================================================

checkpointer = InMemorySaver()

production_agent = create_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "malware"]),

        # Layer 2: PII redaction on input
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email_tool": True, "search_tool": False}
        ),

        # Layer 4: PII redaction on output
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # Layer 5: Model-based output safety
        SafetyGuardrailMiddleware(),
    ],
    checkpointer=checkpointer,
)

print("Production-grade agent with 5-layer guardrails created!\n")


# ======================================================
# Helper
# ======================================================

def run_test(agent, query: str, label: str, approve: bool | None = None):
    """Run a test case through the guarded agent."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "user", "content": query}]}

    print(f"{'='*60}")
    print(f"TEST: {label}")
    print(f"INPUT: {query}")
    print(f"{'-'*60}")

    result = agent.invoke(inputs, config=config)

    # Handle HITL interrupt if agent paused
    state = agent.get_state(config)
    if state.next and approve is not None:
        if approve:
            print(f"  [Layer 3 - HITL] APPROVED by human")
            decision = {"decisions": [{"type": "approve"}]}
        else:
            print(f"  [Layer 3 - HITL] REJECTED by human")
            decision = {"decisions": [{"type": "reject", "message": "User denied this action."}]}
        result = agent.invoke(Command(resume=decision), config=config)
    elif state.next:
        print(f"  [Layer 3 - HITL] INTERRUPTED (no decision provided)")

    final_msg = result["messages"][-1].content
    print(f"OUTPUT: {final_msg}\n")
    return result


# ======================================================
# Demo — test all 5 layers
# ======================================================

if __name__ == "__main__":
    # TEST 1: Clean search — passes all layers, no HITL (search auto-approved)
    run_test(
        production_agent,
        "Search for the best Python frameworks",
        "Clean search (all layers pass)",
    )

    # TEST 2: Blocked by Layer 1 — banned keyword, LLM never called
    run_test(
        production_agent,
        "How to hack into a database",
        "Blocked by ContentFilter (Layer 1)",
    )

    # TEST 3: PII masked (Layer 2) + search auto-approved
    run_test(
        production_agent,
        "Search for orders with card 4111-1111-1111-1234",
        "PII masked in input (Layer 2)",
    )

    # TEST 4: Email — HITL interrupts, human APPROVES (Layer 3)
    run_test(
        production_agent,
        "Send an email to alice@company.com with body 'Meeting at 3pm'",
        "Email approved by human (Layer 3)",
        approve=True,
    )

    # TEST 5: Email — HITL interrupts, human REJECTS (Layer 3)
    run_test(
        production_agent,
        "Send an email to bob@company.com with body 'You are fired'",
        "Email rejected by human (Layer 3)",
        approve=False,
    )
