import re


BLOCKED_INPUT_KEYWORDS = [
    "hack", "exploit", "bypass security", "steal data",
    "make a bomb", "illegal", "crack password",
]

BLOCKED_OUTPUT_KEYWORDS = [
    "as an ai", "i cannot", "i'm sorry but",
    "confidential", "classified",
]

PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b",
}


def check_input(user_message: str) -> dict:
    message_lower = user_message.lower()

    for keyword in BLOCKED_INPUT_KEYWORDS:
        if keyword in message_lower:
            return {
                "allowed": False,
                "reason": f"Blocked keyword detected: '{keyword}'",
                "stage": "input",
            }

    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, user_message):
            return {
                "allowed": False,
                "reason": f"PII detected in input: {pii_type}",
                "stage": "input",
            }

    return {"allowed": True, "reason": None, "stage": "input"}


def check_output(llm_response: str) -> dict:
    response_lower = llm_response.lower()

    for keyword in BLOCKED_OUTPUT_KEYWORDS:
        if keyword in response_lower:
            return {
                "allowed": False,
                "reason": f"Blocked phrase in output: '{keyword}'",
                "stage": "output",
            }

    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, llm_response):
            return {
                "allowed": False,
                "reason": f"PII leaked in output: {pii_type}",
                "stage": "output",
            }

    return {"allowed": True, "reason": None, "stage": "output"}


def redact_pii(text: str) -> str:
    redacted = text
    for pii_type, pattern in PII_PATTERNS.items():
        redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted
