import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from pii_middleware import PIIMiddleware, apply_middlewares

load_dotenv(dotenv_path="../.env")


# --- Tool ---

@tool
def customer_lookup(query: str) -> str:
    """Look up customer information."""
    return f"Customer record found for query: {query}"


# --- Middleware Stack ---

middlewares = [
    # Redact emails in user input before sending to model
    PIIMiddleware(
        "email",
        strategy="redact",
        apply_to_input=True,
    ),
    # Mask credit cards in user input
    PIIMiddleware(
        "credit_card",
        strategy="mask",
        apply_to_input=True,
    ),
    # Block API keys — raise error if detected
    PIIMiddleware(
        "api_key",
        detector=r"sk-[a-zA-Z0-9]{32}",
        strategy="block",
        apply_to_input=True,
    ),
]


# --- Agent ---

def build_agent() -> AgentExecutor:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer support assistant. Use the tools available to help users."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    tools = [customer_lookup]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Guarded Runner ---

def run_with_pii_middleware(agent: AgentExecutor, user_input: str) -> str:
    print(f"\n{'='*60}")
    print(f"USER INPUT: {user_input}")

    # Apply middleware stack
    try:
        sanitized = apply_middlewares(user_input, middlewares)
    except ValueError as e:
        print(f"  {e}")
        return str(e)

    if sanitized != user_input:
        print(f"  [SANITIZED]: {sanitized}")

    # Call agent with sanitized input
    result = agent.invoke({"input": sanitized})
    print(f"\nRESPONSE: {result['output']}")
    return result["output"]


# --- Demo ---

if __name__ == "__main__":
    agent = build_agent()

    test_cases = [
        # PASS — clean query, no PII
        "Look up the account for John Smith",

        # REDACT — email gets replaced
        "Find the customer with email alice@company.com",

        # MASK — credit card partially hidden
        "Check order for card 4111-1111-1111-1234",

        # BLOCK — API key triggers hard block
        "Use this key sk-abcdefghijklmnopqrstuvwxyz123456 to authenticate",

        # MIXED — email redacted + still works
        "Look up orders for bob@shop.io placed last week",
    ]

    for tc in test_cases:
        run_with_pii_middleware(agent, tc)
        print()
