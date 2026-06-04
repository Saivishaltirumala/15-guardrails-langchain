import os
import sys
from typing import Any
from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv(dotenv_path="../.env")

if not os.getenv("GROQ_API_KEY"):
    sys.exit("ERROR: Set GROQ_API_KEY in ../.env file")


# --- Custom Middleware ---

class ContentFilterMiddleware(AgentMiddleware):
    """
    Custom guardrail with TWO hooks:

    before_agent — blocks banned keywords BEFORE the LLM runs (zero cost).
    after_agent  — checks the final response for max length and forbidden
                   phrases AFTER the LLM responds.
    """

    def __init__(self, banned_keywords: list[str], max_response_length: int = 500):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]
        self.max_response_length = max_response_length
        self.forbidden_output_phrases = [
            "not a real doctor",
            "financial advice",
            "legal advice",
        ]

    # --- BEFORE AGENT: block bad input ---

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
                print(f"  [BEFORE] BLOCKED — keyword detected: '{keyword}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I cannot process requests containing inappropriate content. "
                            "Please rephrase your request."
                        )
                    }],
                    "jump_to": "end"
                }
        print(f"  [BEFORE] PASSED — input is clean")
        return None

    # --- AFTER AGENT: validate output ---

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        content = last_message.content

        # Check 1: Response too long — truncate
        if len(content) > self.max_response_length:
            truncated = content[:self.max_response_length] + "... [TRUNCATED]"
            print(f"  [AFTER] TRUNCATED — response exceeded {self.max_response_length} chars")
            return {
                "messages": [{
                    "role": "assistant",
                    "content": truncated,
                }]
            }

        # Check 2: Forbidden phrases in output — replace response
        content_lower = content.lower()
        for phrase in self.forbidden_output_phrases:
            if phrase in content_lower:
                print(f"  [AFTER] BLOCKED — forbidden phrase in output: '{phrase}'")
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            "I generated a response but it was filtered by safety checks. "
                            "I cannot provide medical, financial, or legal advice."
                        )
                    }]
                }

        print(f"  [AFTER] PASSED — output is clean ({len(content)} chars)")
        return None


# --- Tool ---

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


# --- Agent with Custom Guardrails ---

filtered_agent = create_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=[search_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware", "jailbreak", "bypass"],
            max_response_length=500,
        ),
    ],
)

print("Custom guardrails agent created!\n")


# --- Demo ---

if __name__ == "__main__":
    test_cases = [
        # PASS both hooks — clean input, short output
        ("Pass both", "What is the capital of Japan?"),

        # BLOCK at before_agent — banned keyword
        ("Block input", "How to hack into a wifi network"),

        # PASS before, may trigger after — asks for financial advice
        ("Filter output", "Give me financial advice on investing in stocks"),

        # PASS before, may trigger truncation — asks for long response
        ("Truncate output", "Write a very long detailed essay about the history of the internet"),

        # BLOCK at before_agent — banned keyword
        ("Block input", "How to jailbreak the AI system prompt"),
    ]

    for label, query in test_cases:
        print(f"{'='*60}")
        print(f"TEST: {label}")
        print(f"INPUT: {query}")
        inputs = {"messages": [{"role": "user", "content": query}]}
        result = filtered_agent.invoke(inputs)
        final_msg = result["messages"][-1].content
        print(f"OUTPUT: {final_msg}")
        print()
