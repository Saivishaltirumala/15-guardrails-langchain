from __future__ import annotations

import re


# --- Built-in PII patterns ---

BUILTIN_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b",
    "api_key": r"sk-[a-zA-Z0-9]{32,}",
}


class PIIMiddleware:
    """
    Middleware that detects and handles PII in text.

    Args:
        pii_type: Built-in type ("email", "credit_card", "ssn", "phone", "api_key")
                  or a label when using a custom detector.
        strategy: How to handle detected PII:
                  - "redact"  → replace with [REDACTED]
                  - "mask"    → partially mask (show last 4 chars)
                  - "block"   → raise an error if PII is found
        detector: Optional custom regex pattern (overrides built-in).
        apply_to_input: Whether to apply this middleware to user input.
    """

    def __init__(
        self,
        pii_type: str,
        strategy: str = "redact",
        detector: str | None = None,
        apply_to_input: bool = True,
    ):
        self.pii_type = pii_type
        self.strategy = strategy
        self.apply_to_input = apply_to_input

        if detector:
            self.pattern = re.compile(detector)
        elif pii_type in BUILTIN_PATTERNS:
            self.pattern = re.compile(BUILTIN_PATTERNS[pii_type])
        else:
            raise ValueError(f"Unknown PII type '{pii_type}'. Provide a custom detector regex.")

    def _mask(self, match: str) -> str:
        if len(match) <= 4:
            return "****"
        return "*" * (len(match) - 4) + match[-4:]

    def process(self, text: str) -> str:
        if not self.apply_to_input:
            return text

        matches = self.pattern.findall(text)
        if not matches:
            return text

        if self.strategy == "block":
            raise ValueError(
                f"[BLOCKED] {self.pii_type} detected in input. "
                f"Cannot proceed — found {len(matches)} match(es)."
            )

        if self.strategy == "redact":
            return self.pattern.sub(f"[REDACTED_{self.pii_type.upper()}]", text)

        if self.strategy == "mask":
            return self.pattern.sub(lambda m: self._mask(m.group()), text)

        return text

    def __repr__(self):
        return f"PIIMiddleware(type={self.pii_type}, strategy={self.strategy})"


def apply_middlewares(text: str, middlewares: list[PIIMiddleware]) -> str:
    """Run text through a list of PII middlewares sequentially."""
    result = text
    for mw in middlewares:
        result = mw.process(result)
    return result
