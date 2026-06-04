import os
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.tools import tool

load_dotenv(dotenv_path="../.env")

if not os.getenv("GROQ_API_KEY"):
    sys.exit("ERROR: Set GROQ_API_KEY in ../.env file")


# --- Tool ---

@tool
def customer_lookup(query: str) -> str:
    """Look up customer information."""
    return f"Customer record found for query: {query}"


# --- Agent with PII Middleware ---

agent = create_agent(
    "groq:llama-3.3-70b-versatile",
    tools=[customer_lookup],
    middleware=[
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
    ],
)

print("Agent with PII middleware created successfully!\n")


# --- Demo ---

if __name__ == "__main__":
    test_cases = [
        # PASS — clean query, no PII
        ("Clean input", "Look up the account for John Smith"),

        # REDACT — email gets replaced before reaching LLM
        ("Email redact", "Find the customer with email alice@company.com"),

        # MASK — credit card partially hidden
        ("Card mask", "Check order for card 4111-1111-1111-1234"),

        # BLOCK — API key triggers hard block, agent never called
        ("API key block", "Use this key sk-abcdefghijklmnopqrstuvwxyz123456 to authenticate"),
    ]

    for label, query in test_cases:
        print(f"{'='*60}")
        print(f"TEST: {label}")
        print(f"INPUT: {query}")
        try:
            result = agent.invoke(query)
            print(f"OUTPUT: {result}")
        except Exception as e:
            print(f"BLOCKED: {e}")
        print()
