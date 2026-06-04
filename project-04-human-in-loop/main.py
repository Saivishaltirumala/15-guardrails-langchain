import os
import sys
import uuid
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.tools import tool

load_dotenv(dotenv_path="../.env")

if not os.getenv("GROQ_API_KEY"):
    sys.exit("ERROR: Set GROQ_API_KEY in ../.env file")


# --- Tools ---

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject: {subject}"


@tool
def delete_records(table: str, condition: str) -> str:
    """Delete records from the database."""
    return f"Deleted records from {table} where {condition}"


# --- Agent with HITL Middleware ---

checkpointer = InMemorySaver()

hitl_agent = create_agent(
    model="groq:llama-3.3-70b-versatile",
    tools=[search_web, send_email, delete_records],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,       # Require approval
                "delete_records": True,   # Require approval
                "search_web": False,      # Auto-approve
            }
        ),
    ],
    checkpointer=checkpointer,
)

print("Human-in-the-Loop agent created!\n")


# --- Helper to run agent with interrupt handling ---

def run_with_hitl(agent, query: str, approve: bool):
    """Run agent, handle interrupt, and resume with approve/reject."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "user", "content": query}]}

    print(f"{'='*60}")
    print(f"QUERY: {query}")
    print(f"HUMAN DECISION: {'APPROVE' if approve else 'REJECT'}")
    print(f"{'-'*60}")

    # Step 1: Invoke agent — it will pause at tools needing approval
    result = agent.invoke(inputs, config=config)

    # Check if agent is interrupted (waiting for human approval)
    state = agent.get_state(config)
    if state.next:
        # Agent is paused, waiting for human decision
        print(f"INTERRUPTED: Agent wants to call a tool requiring approval")

        if approve:
            print("HUMAN: Approved! Resuming agent...")
            decision = {"decisions": [{"type": "approve"}]}
        else:
            print("HUMAN: Rejected! Cancelling tool call...")
            decision = {"decisions": [{"type": "reject", "message": "User denied this action."}]}

        result = agent.invoke(Command(resume=decision), config=config)
    else:
        print("NO INTERRUPT: Tool was auto-approved or no tool called")

    # Extract final assistant message
    final_msg = result["messages"][-1].content
    print(f"FINAL OUTPUT: {final_msg}")
    print()
    return result


# --- Demo ---

if __name__ == "__main__":
    # TEST 1: APPROVE — send_email requires approval, human says YES
    print("TEST 1: Send email — Human APPROVES")
    run_with_hitl(
        hitl_agent,
        "Send an email to bob@example.com with subject 'Meeting Tomorrow' and body 'Let us meet at 3pm'",
        approve=True,
    )

    # TEST 2: REJECT — delete_records requires approval, human says NO
    print("TEST 2: Delete records — Human REJECTS")
    run_with_hitl(
        hitl_agent,
        "Delete all records from the users table where status is inactive",
        approve=False,
    )
