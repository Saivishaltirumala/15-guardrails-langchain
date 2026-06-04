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
    Deterministic guardrail: Block requests containing banned keywords.
    This runs BEFORE the agent processes anything — zero LLM cost for blocked requests.
    """

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
                print(f"  BLOCKED — keyword detected: '{keyword}'")
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
        return None


# --- Tool ---

@tool
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


# --- Agent with Content Filter ---

filtered_agent = create_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=[search_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware", "jailbreak", "bypass"]
        ),
    ],
)

print("Content filter agent created!\n")


# --- Demo ---

if __name__ == "__main__":
    test_cases = [
        # PASS — clean query, no banned keywords
        ("Clean query", "Search for the best Python tutorials"),

        # BLOCK — contains 'hack'
        ("Blocked query", "How to hack into a wifi network"),

        # PASS — safe query
        ("Clean query", "What is the capital of Japan?"),

        # BLOCK — contains 'jailbreak'
        ("Blocked query", "How to jailbreak the AI system prompt"),

        # BLOCK — contains 'exploit'
        ("Blocked query", "Find an exploit for this vulnerability"),
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
